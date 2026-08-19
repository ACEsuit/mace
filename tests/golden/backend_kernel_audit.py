"""Did the vendor kernel actually run?

A backend golden compares numbers, and numbers alone cannot tell an
accelerated kernel from the plain e3nn path that produced the reference in
the first place -- the two are *supposed* to agree. So a value-only assertion
is green in exactly the case the golden exists to catch: the conversion
quietly produced something that is not accelerated at all. Two ways that
happens here, both silent, both observed rather than imagined:

* **cuEquivariance downgrades and only warns.** ``cuet.SegmentedPolynomial``
  falls back to ``SegmentedPolynomialNaive`` when ``cuequivariance_ops_torch``
  cannot be imported (an ops wheel whose CUDA major does not match torch's, or
  a version pin below the one that carries the fused kernels). It sets
  ``.method`` to ``"naive"``, logs a warning, and returns a module that
  computes the right answer with no vendor kernel behind it.
  ``CueqConvFusionWrapper`` refuses that today, but the refusal lives in the
  code under test: a golden that relies on it is trusting the thing it is
  supposed to be checking.
* **The converter was handed the wrong device.** ``convert_e3nn_cueq.run``
  sets ``conv_fusion=(device == "cuda")``. Converting on ``"cpu"`` and moving
  the model afterwards yields a cueq model whose conv path is the *unfused*
  ``ChannelWiseTensorProduct`` -- still cueq, still correct, and not the kernel
  anybody meant to pin. Nothing in the outputs says so.

This module answers the structural question ("is the accelerated
implementation the one in place?") and the dynamic one ("did it execute?"),
and it is written so both can be exercised without a GPU: everything is read
off ordinary attributes, so a stand-in module tree reproduces either failure
on a laptop. See ``tests/golden/test_backend_parity_golden.py``.

Nothing here imports ``cuequivariance`` or ``openequivariance``. It must be
importable on a host that has neither, because the test that proves the audit
*rejects* an unaccelerated model is exactly the test that runs there.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence

CUEQ = "cueq"
OEQ = "oeq"
BACKENDS = (CUEQ, OEQ)

#: Root packages a backend's modules must come from. cuEquivariance splits
#: itself over two distributions (``cuequivariance`` for the descriptors,
#: ``cuequivariance_torch`` for the modules), hence the prefix match.
_PACKAGE_PREFIX = {CUEQ: "cuequivariance", OEQ: "openequivariance"}

#: The method ``cuet.SegmentedPolynomial`` reports when the fused uniform_1d
#: kernels are in use. Any other value means the fallback -- see the module
#: docstring. This is the same attribute ``mace/modules/wrapper_ops.py`` reads
#: to build its own guard, so the two cannot disagree about where to look.
_CUEQ_FUSED_METHOD = "uniform_1d"


@dataclass(frozen=True)
class KernelSite:
    """One place in the model where a vendor kernel is supposed to be.

    ``module`` is the object whose ``__call__`` has to fire, which is not
    always the object holding the implementation: MACE wraps the vendor's
    module to adapt its calling convention, and the wrapper is what the
    interaction block invokes.
    """

    path: str
    role: str
    module: Any
    implementation: str
    complaints: tuple = ()

    @property
    def healthy(self) -> bool:
        return not self.complaints


@dataclass
class _SiteDraft:
    path: str
    role: str
    module: Any
    implementation: str = "<none>"
    complaints: List[str] = field(default_factory=list)

    def freeze(self) -> KernelSite:
        return KernelSite(
            path=self.path,
            role=self.role,
            module=self.module,
            implementation=self.implementation,
            complaints=tuple(self.complaints),
        )


def qualname(obj: Any) -> str:
    """``module.QualName`` of an object's class, for messages and records."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _from_package(obj: Any, prefix: str) -> bool:
    return type(obj).__module__.split(".")[0].startswith(prefix)


def _cueq_conv_site(index: int, interaction: Any) -> _SiteDraft:
    # Imported here rather than at module scope only to keep the import graph
    # of this file to the standard library plus MACE's own wrapper types.
    from mace.modules.wrapper_ops import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        CueqConvFusionWrapper,
    )

    conv = interaction.conv_tp
    draft = _SiteDraft(
        path=f"interactions.{index}.conv_tp",
        role="channelwise tensor product (fused conv)",
        module=conv,
        implementation=qualname(conv),
    )
    if not hasattr(interaction, "conv_fusion"):
        draft.complaints.append(
            "the interaction block carries no `conv_fusion` attribute, so its "
            "forward takes the unfused branch (blocks.py checks "
            "`hasattr(self, 'conv_fusion')`). convert_e3nn_cueq sets "
            "conv_fusion=(device == 'cuda'): converting on 'cpu' and moving "
            "the model afterwards lands exactly here"
        )
    if not isinstance(conv, CueqConvFusionWrapper):
        draft.complaints.append(
            f"conv_tp is {qualname(conv)}, not the fused CueqConvFusionWrapper"
            + (
                ""
                if _from_package(conv, _PACKAGE_PREFIX[CUEQ])
                else " -- and it is not a cuEquivariance module at all, so this "
                "model was never converted (or the conversion silently fell "
                "back to e3nn)"
            )
        )
        return draft
    inner = conv.conv_tp
    draft.implementation = qualname(inner)
    method = getattr(inner, "method", None)
    if method != _CUEQ_FUSED_METHOD:
        draft.complaints.append(
            f"cuequivariance selected method={method!r} instead of "
            f"{_CUEQ_FUSED_METHOD!r}: the fused kernels are not in use"
        )
    implementation = getattr(inner, "m", None)
    if implementation is None:
        draft.complaints.append(
            "the SegmentedPolynomial exposes no `.m`, so which implementation "
            "cuequivariance selected cannot be read; treat this as a failure "
            "rather than a pass -- an audit that cannot see the kernel is not "
            "an audit"
        )
    else:
        name = qualname(implementation)
        draft.implementation = name
        if "naive" in name.lower():
            draft.complaints.append(
                f"the implementation is {name}: cuequivariance downgraded to "
                "its naive path, which it does with a warning and no error "
                "when cuequivariance_ops_torch cannot be imported"
            )
    return draft


def _cueq_symmetric_site(index: int, product: Any) -> _SiteDraft:
    contraction = product.symmetric_contractions
    draft = _SiteDraft(
        path=f"products.{index}.symmetric_contractions",
        role="symmetric contraction",
        module=contraction,
        implementation=qualname(contraction),
    )
    if not _from_package(contraction, _PACKAGE_PREFIX[CUEQ]):
        draft.complaints.append(
            f"the symmetric contraction is {qualname(contraction)}, which is "
            "MACE's own e3nn implementation and not cuEquivariance's"
        )
        return draft
    # Only checked when the attribute exists: unlike the conv path above,
    # nothing in MACE reads a `method` off this module, so its absence is a
    # fact about a cuequivariance version rather than evidence of a fallback.
    method = getattr(contraction, "method", None)
    if isinstance(method, str) and "naive" in method.lower():
        draft.complaints.append(
            f"the symmetric contraction reports method={method!r}: the fused "
            "kernels are not in use"
        )
    return draft


def _oeq_conv_site(index: int, interaction: Any) -> _SiteDraft:
    conv = interaction.conv_tp
    draft = _SiteDraft(
        path=f"interactions.{index}.conv_tp",
        role="channelwise tensor product (fused conv)",
        module=conv,
        implementation=qualname(conv),
    )
    if not _from_package(conv, _PACKAGE_PREFIX[OEQ]):
        draft.complaints.append(
            f"conv_tp is {qualname(conv)}, not an OpenEquivariance module: "
            "this model was never converted, or the conversion fell back to "
            "e3nn because openequivariance was not importable (OEQConfig "
            "silently sets enabled=False in that case)"
        )
        return draft
    if not hasattr(interaction, "conv_fusion"):
        draft.complaints.append(
            "the interaction block carries no `conv_fusion` attribute, so its "
            "forward takes the unfused branch"
        )
    if not hasattr(conv, "original_forward"):
        draft.complaints.append(
            "the conv_tp was not wrapped by with_oeq_conv_fusion, so MACE's "
            "calling convention was never adapted to the fused kernel"
        )
    return draft


def kernel_sites(model: Any, backend: str) -> List[KernelSite]:
    """Every place ``backend`` is supposed to have put a vendor kernel.

    Sites are returned whether they are healthy or not; the complaints ride
    along, so a caller can report all of them at once instead of failing on
    the first layer.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; the backends are {BACKENDS}")
    interactions = getattr(model, "interactions", None)
    if interactions is None:
        raise TypeError(
            f"{qualname(model)} has no `interactions`; this audit reads the "
            f"MACE module tree and was handed something else"
        )
    drafts: List[_SiteDraft] = []
    for index, interaction in enumerate(interactions):
        if backend == CUEQ:
            drafts.append(_cueq_conv_site(index, interaction))
        else:
            drafts.append(_oeq_conv_site(index, interaction))
    if backend == CUEQ:
        # The symmetric contraction is cuEquivariance's only under cueq: the
        # oeq converter leaves it on MACE's own implementation (see
        # SymmetricContractionWrapper, which takes an oeq_config and ignores
        # it), so demanding a vendor module there would fail every oeq run.
        for index, product in enumerate(getattr(model, "products", ()) or ()):
            drafts.append(_cueq_symmetric_site(index, product))
    for draft in drafts:
        if not hasattr(draft.module, "register_forward_hook"):
            draft.complaints.append(
                f"{qualname(draft.module)} is not an nn.Module, so whether it "
                f"ran cannot be observed"
            )
    return [draft.freeze() for draft in drafts]


def environment_report(backend: str) -> str:
    """What the host has, in the words a failure needs to be actionable.

    Appended to every failure message rather than asserted on its own: the
    module names below are cuEquivariance's business and could be renamed by
    an upgrade, and a diagnostic that turns into a false failure is worse than
    no diagnostic. The one thing it must never do is stay silent about the
    root cause, which for the naive downgrade is an ops wheel that will not
    import.
    """
    lines = []
    try:
        import torch  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        lines.append(
            f"torch {torch.__version__} (cuda={torch.version.cuda}, "
            f"hip={torch.version.hip}, cuda.is_available="
            f"{torch.cuda.is_available()})"
        )
    except Exception as error:  # pylint: disable=broad-except
        lines.append(f"torch could not be inspected: {error!r}")
    probes = {
        CUEQ: ("cuequivariance", "cuequivariance_torch", "cuequivariance_ops_torch"),
        OEQ: ("openequivariance",),
    }[backend]
    for name in probes:
        try:
            module = __import__(name)
            lines.append(f"{name} {getattr(module, '__version__', '?')}")
        except Exception as error:  # pylint: disable=broad-except
            lines.append(f"{name} does NOT import: {error!r}")
    return "\n    ".join(lines)


def assert_vendor_kernels(model: Any, backend: str) -> List[KernelSite]:
    """Assert the converted model is accelerated where it claims to be.

    Returns the sites, so the caller can hand them straight to :func:`watch`
    and then to :func:`assert_every_site_ran` -- the structural check and the
    "it executed" check are two halves of one claim and share their subject.
    """
    sites = kernel_sites(model, backend)
    if not sites:
        raise AssertionError(
            f"{qualname(model)} exposes no place a {backend} kernel could sit, "
            f"so this golden would pin nothing about {backend}"
        )
    bad = [site for site in sites if not site.healthy]
    if not bad:
        return sites
    detail = "\n".join(
        f"  {site.path} ({site.role}) -> {site.implementation}\n"
        + "\n".join(f"      - {complaint}" for complaint in site.complaints)
        for site in bad
    )
    raise AssertionError(
        f"the model does not run {backend} kernels at {len(bad)} of "
        f"{len(sites)} site(s), so a value comparison here would pass without "
        f"any vendor kernel having run:\n"
        f"{detail}\n"
        f"  environment:\n    {environment_report(backend)}"
    )


@contextlib.contextmanager
def watch(sites: Sequence[KernelSite]) -> Iterator[Dict[str, int]]:
    """Count how many times each site's module is called, inside the block."""
    counts: Dict[str, int] = {site.path: 0 for site in sites}
    handles = []

    def make_hook(path: str):
        def hook(module, args, output):  # pylint: disable=unused-argument
            counts[path] += 1

        return hook

    for site in sites:
        handles.append(site.module.register_forward_hook(make_hook(site.path)))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def assert_every_site_ran(
    sites: Sequence[KernelSite], counts: Dict[str, int], backend: str
) -> None:
    """Assert every audited site was actually executed.

    The structural check says the kernel is installed; this one says the
    forward went through it. They are not the same claim -- an interaction
    block that stopped calling its ``conv_tp`` (a refactor, a LAMMPS branch, a
    compiled copy of the model that left the original untouched) would keep
    the structure and lose the kernel.
    """
    silent = sorted(site.path for site in sites if counts.get(site.path, 0) == 0)
    if silent:
        raise AssertionError(
            f"{len(silent)} {backend} site(s) were never called during the "
            f"evaluation, so the numbers just compared did not come out of "
            f"them: {silent}. Call counts: {dict(sorted(counts.items()))}"
        )
