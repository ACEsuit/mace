"""The accelerated backends reproduce the committed CPU e3nn references.

`tests/backends/backend_parity.py` already checks that a freshly built model
survives the round trip e3nn -> backend -> e3nn *in one process, on one
machine, against a model built beside it*. That is a self-consistency check:
if the e3nn side moved, both sides move together and the comparison still
passes. What was missing is the cross-machine half -- the accelerated model
evaluated on a GPU against numbers that were committed to this repository
from a CPU e3nn run and cannot move without somebody rewriting a file.

So the assertion path here is deliberately: committed checkpoint -> convert ->
evaluate on the accelerator -> compare against the committed JSON. No e3nn
model is built in this process, because a fresh one would re-derive the very
oracle the comparison is supposed to be independent of.

Three things about this file are not obvious and each was paid for once:

* **The conversion must be told it is targeting a GPU.**
  ``convert_e3nn_cueq.run`` sets ``conv_fusion=(device == "cuda")``, so
  converting with the default ``device="cpu"`` and moving the model
  afterwards produces a cueq model whose conv path is the *unfused*
  ``ChannelWiseTensorProduct``. It still gives the right numbers, so a golden
  taken that way passes for years while pinning a kernel nobody runs.
* **Matching the reference is not evidence that a vendor kernel ran.** The
  reference was produced by the plain e3nn path, so the plain e3nn path
  reproduces it perfectly -- and cuEquivariance downgrades to
  ``SegmentedPolynomialNaive`` with a warning and no error when its ops wheel
  cannot be imported. Every case therefore audits the module tree and counts
  the calls into it (``tests/golden/backend_kernel_audit.py``); the CPU tests
  at the bottom of this file exercise both halves of that audit on a host
  with no GPU, including against a real degraded cuEquivariance module.
* **Parity runs on the ScaleShiftMACE anchor, not the plain-MACE one.** Both
  converters route through ``extract_config_mace_model``, whose whitelist does
  not include a plain ``MACE``; converting one tests the refusal, not the
  kernels. The refusal is pinned as a contract instead.

This file commits no new reference and regenerates none.
"""
# pylint: disable=wrong-import-position
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import pytest
import torch

from mace.tools import torch_tools
from tests.golden import backend_kernel_audit as audit
from tests.golden import harness

ANCHOR_MODEL = harness.MODELS_DIR / "tiny_scaleshift.model"
ANCHOR_REFERENCE = harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"
PLAIN_ANCHOR_MODEL = harness.MODELS_DIR / "tiny_mace.model"
#: Committed by the foundation-model goldens, which are a separate change.
#: Absent here means "not merged yet", not "not pinned": see _reference_for.
MP_SMALL_REFERENCE = harness.REFERENCES_DIR / "mp_small_e3nn_cpu_fp64.json"

#: Tests never encode a vendor. "cuda" is what a ROCm build answers to as
#: well, and which vendor a job is on is decided by its marker expression.
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# The models under test and how each is reached
# ---------------------------------------------------------------------------


def _load_anchor(path=ANCHOR_MODEL, device="cpu"):
    return torch.load(path, weights_only=False, map_location=device).to(torch.float64)


def _load_mp_small(device):
    """MACE-MP-0 small, as the foundation-model golden loads it.

    Same loader and same dtype; only the device differs, and it has to --
    the reference is the CPU evaluation this run is being compared against.
    """
    from mace.calculators.foundations_models import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        mace_mp,
    )

    calc = mace_mp(model="small", default_dtype="float64", device=device)
    models = getattr(calc, "models", None)
    if not models:
        raise AssertionError(
            f"mace_mp returned a {type(calc).__name__} with no `models`; the "
            f"loader used to hand back a plain MACECalculator and this golden "
            f"converts the module it holds"
        )
    return models[0].to(torch.float64)


MODELS = {
    "tiny_scaleshift": {
        "load": lambda device: _load_anchor(ANCHOR_MODEL, device),
        "reference": ANCHOR_REFERENCE,
        "provided_by": "",
    },
    "mp_small": {
        "load": _load_mp_small,
        "reference": MP_SMALL_REFERENCE,
        "provided_by": "the foundation-model goldens",
    },
}


def _reference_for(name):
    """The committed reference, or a skip that says who owns the missing file.

    A capability that is promised and absent must fail rather than skip, which
    is what ``MACE_REQUIRE_CAPS`` is for. This is not that case: the file
    below is another change's artifact, so its absence is a merge order and
    resolves itself, and the skip reason names the owner so it cannot quietly
    become permanent. Everything else here fails.
    """
    spec = MODELS[name]
    path = spec["reference"]
    if path.is_file():
        return harness.load_reference(path)
    pytest.skip(
        f"{path.name} is not in this tree; it is committed by "
        f"{spec['provided_by']}, and this golden asserts against it rather "
        f"than generating its own reference"
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _convert(model, backend, device):
    """Convert through the shipped entry points, with the device declared.

    The ``default_dtype`` scope is not decoration: ``run`` calls
    ``torch.set_default_dtype`` from the source model's parameters and never
    puts it back, and the accelerated modules read the process-wide default
    at construction time. Leaving that to escape would silently re-dtype
    every later test in the session.
    """
    with torch_tools.default_dtype("float64"):
        if backend == audit.CUEQ:
            from mace.cli.convert_e3nn_cueq import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
                run as run_e3nn_to_cueq,
            )

            return run_e3nn_to_cueq(model, device=device)
        from mace.cli.convert_e3nn_oeq import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
            run as run_e3nn_to_oeq,
        )

        return run_e3nn_to_oeq(model, device=device)


def _calculator(model, device):
    from mace.calculators import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        MACECalculator,
    )

    return MACECalculator(models=[model], device=device, default_dtype="float64")


# ---------------------------------------------------------------------------
# The goldens
#
# mp_small is a cueq case only, and that is a decision rather than an
# oversight: an oeq case carrying `network` would also be collected by the AMD
# job (its expression is "gpu and not cueq"), which would put a download into
# a job whose required-capability list says it has no network test to reach.
# The anchor covers oeq on both vendors; MACE-MP-0 small adds a published,
# many-element model on the one vendor that can run both backends.
# ---------------------------------------------------------------------------

CASES = [
    pytest.param(
        "tiny_scaleshift", audit.CUEQ, marks=pytest.mark.cueq, id="tiny_scaleshift-cueq"
    ),
    pytest.param(
        "tiny_scaleshift", audit.OEQ, marks=pytest.mark.oeq, id="tiny_scaleshift-oeq"
    ),
    pytest.param(
        "mp_small",
        audit.CUEQ,
        marks=[pytest.mark.cueq, pytest.mark.network],
        id="mp_small-cueq",
    ),
]



def _fixtures_for(model):
    """The fixtures this model can evaluate at all.

    Derived from the model rather than hardcoded, because the manifest is
    shared with every other golden family: a bare ``load_fixtures()`` hands
    whichever structures the next family commits to a model that has no
    z-table entry for their elements, which fails as a KeyError rather than
    as a tolerance miss. The tiny anchor is H/C/O; a published foundation
    model covers far more, and both get exactly their own subset.
    """
    return harness.load_fixtures(
        elements=[int(z) for z in model.atomic_numbers]
    )

@pytest.mark.gpu
@pytest.mark.parametrize("model_name,backend", CASES)
def test_converted_model_reproduces_the_committed_cpu_reference(model_name, backend):
    reference = _reference_for(model_name)

    source = MODELS[model_name]["load"](DEVICE)
    fixtures = _fixtures_for(source)
    source_class = type(source).__name__
    converted = _convert(source, backend, DEVICE)

    assert type(converted).__name__ == source_class, (
        "the conversion returned a different model class; the golden would be "
        "comparing a different architecture against the reference"
    )
    calc = _calculator(converted, DEVICE)
    running = calc.models[0]
    assert running is converted, (
        "the calculator did not keep the converted module, so the audit below "
        "would be watching an object that never runs"
    )
    assert next(running.parameters()).device.type == "cuda", (
        "the evaluated model is not on the accelerator"
    )

    sites = audit.assert_vendor_kernels(running, backend)
    with audit.watch(sites) as counts:
        snapshot = harness.snapshot_outputs(
            calc,
            fixtures,
            dtype="float64",
            device=DEVICE,
            backend=backend,
            metadata={"converted_from": source_class, "n_kernel_sites": len(sites)},
        )
    audit.assert_every_site_ran(sites, counts, backend)

    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_ACCELERATED_BACKEND.name
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(audit.CUEQ, marks=pytest.mark.cueq, id="cueq"),
        pytest.param(audit.OEQ, marks=pytest.mark.oeq, id="oeq"),
    ],
)
def test_the_calculators_own_backend_flag_reaches_the_same_kernels(backend):
    """The route a user takes, held to the same standard as the converter.

    ``MACECalculator(enable_cueq=True)`` does the conversion itself and passes
    its own ``device`` through (mace/calculators/mace.py:359-369). That is the
    shipped path, and it would be perfectly possible for the golden above to
    pin fused kernels while the calculator flag quietly produced unfused ones.
    """
    from mace.calculators import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        MACECalculator,
    )

    flag = {audit.CUEQ: "enable_cueq", audit.OEQ: "enable_oeq"}[backend]
    calc = MACECalculator(
        models=[_load_anchor(ANCHOR_MODEL, DEVICE)],
        device=DEVICE,
        default_dtype="float64",
        **{flag: True},
    )
    sites = audit.assert_vendor_kernels(calc.models[0], backend)
    with audit.watch(sites) as counts:
        harness.snapshot_outputs(
            calc,
            harness.load_fixtures(["water_cluster"]),
            dtype="float64",
            device=DEVICE,
            backend=backend,
        )
    audit.assert_every_site_ran(sites, counts, backend)


# ---------------------------------------------------------------------------
# The audit, exercised without a GPU
#
# These are the tests that keep the oracle above honest, and they were written
# to run on a laptop on purpose: the failure they guard against is a *passing*
# test, so it can never be discovered by watching a GPU job stay green.
# ---------------------------------------------------------------------------


def test_a_value_comparison_alone_cannot_tell_an_unaccelerated_model_apart():
    """Why every case above audits the module tree.

    The plain e3nn anchor is the model the references were taken from, so it
    reproduces them at the *accelerated* tolerance row with room to spare --
    while running no vendor kernel whatsoever. Any backend golden that
    asserted values only would be green on this model.
    """
    model = _load_anchor()
    snapshot = harness.snapshot_outputs(
        _calculator(model, "cpu"),
        _fixtures_for(model),
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(
        snapshot,
        harness.load_reference(ANCHOR_REFERENCE),
        row=harness.FP64_ACCELERATED_BACKEND.name,
    )

    for backend in audit.BACKENDS:
        with pytest.raises(AssertionError) as complaint:
            audit.assert_vendor_kernels(model, backend)
        message = str(complaint.value)
        assert "interactions.0.conv_tp" in message
        assert "e3nn" in message


@pytest.mark.cueq
def test_converting_for_the_cpu_leaves_cueq_unfused_and_the_audit_says_so():
    """The first of the two silent failures, reproduced end to end.

    ``convert_e3nn_cueq.run`` ties conv fusion to ``device == "cuda"``. This
    is the same call the goldens make with the wrong device, and the model it
    returns is a real cueq model -- ``ChannelWiseTensorProduct`` in the conv
    slot, ``cuet.SymmetricContraction`` in the products -- which is exactly
    why nothing else in the tree would notice. Measured here: it reproduces
    the committed reference to 1.7e-16, eleven orders of magnitude inside the
    1e-5 accelerated row, so there is no tightening of a tolerance that would
    catch this. Only the structure says which kernel is in place.
    """
    converted = _convert(_load_anchor(), audit.CUEQ, "cpu")
    conv = converted.interactions[0].conv_tp
    assert audit.qualname(conv).startswith("cuequivariance"), (
        f"expected a cuEquivariance module in the conv slot, got "
        f"{audit.qualname(conv)}; this test is no longer reproducing the "
        f"unfused-but-converted case it exists for"
    )
    assert not hasattr(converted.interactions[0], "conv_fusion")

    snapshot = harness.snapshot_outputs(
        _calculator(converted, "cpu"),
        _fixtures_for(converted),
        dtype="float64",
        device="cpu",
        backend="cueq",
    )
    harness.compare_to_reference(
        snapshot,
        harness.load_reference(ANCHOR_REFERENCE),
        row=harness.FP64_ACCELERATED_BACKEND.name,
    )

    with pytest.raises(AssertionError) as complaint:
        audit.assert_vendor_kernels(converted, audit.CUEQ)
    message = str(complaint.value)
    assert "conv_fusion" in message
    assert "device == 'cuda'" in message


@pytest.mark.cueq
def test_the_audits_verdict_tracks_whether_the_fused_ops_are_installed():
    """The second silent failure, against the real cuEquivariance object.

    ``cuet.SegmentedPolynomial(..., method="uniform_1d")`` does not fail when
    ``cuequivariance_ops_torch`` is missing: it warns, sets ``.method`` to
    ``"naive"`` and hands back a working module. This test asks for the fused
    kernels and then asserts the audit's verdict matches what the host can
    actually provide -- rejection where the ops are absent (a plain
    ``[cueq]`` install, which is what the CPU backend job has), acceptance
    where they are present (a ``[cueq-cuda-*]`` install). Both directions
    matter: an audit that always failed would be just as useless as one that
    always passed.

    MACE's own wrapper refuses the naive downgrade at construction time. This
    goes around that guard deliberately -- the guard is code under test, and a
    golden that depends on it is trusting the thing it is checking.
    """
    import cuequivariance as cue  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    import cuequivariance_torch as cuet  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    from mace.tools.cg import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        O3_e3nn,
    )

    polynomial = (
        cue.descriptors.channelwise_tensor_product(
            cue.Irreps(O3_e3nn, "16x0e + 16x1o"),
            cue.Irreps(O3_e3nn, "0e + 1o + 2e"),
            cue.Irreps(O3_e3nn, "16x0e + 16x1o"),
        )
        .flatten_coefficient_modes()
        .squeeze_modes()
        .polynomial
    )
    segmented = cuet.SegmentedPolynomial(
        polynomial, math_dtype=torch.float64, method="uniform_1d"
    )
    fused_ops_available = getattr(segmented, "method", None) == "uniform_1d"

    model = _StandInModel([_StandInInteraction(_unguarded_fusion_wrapper(segmented))])
    if fused_ops_available:
        sites = audit.assert_vendor_kernels(model, audit.CUEQ)
        assert len(sites) == 1
        assert "naive" not in sites[0].implementation.lower()
        return
    with pytest.raises(AssertionError) as complaint:
        audit.assert_vendor_kernels(model, audit.CUEQ)
    message = str(complaint.value)
    assert "naive" in message
    # The failure has to name the cause, not only the symptom: "method is
    # naive" sends somebody reading cuequivariance's source, while the ops
    # wheel that will not import is the thing to fix.
    assert "cuequivariance_ops_torch" in message


def test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that():
    """A positive control for the branch no CPU host can reach for real.

    OpenEquivariance JIT-compiles its kernels and does not install without a
    toolkit, so nothing on a laptop -- or on any non-GPU CI job -- ever hands
    the oeq branch a healthy model. Without this, a mistake in that branch
    would first surface as a red vendor job, and the reading would be
    ambiguous: broken audit or broken conversion? The stand-in carries exactly
    the three things the audit reads, so it also fails if the audit starts
    reading something else.
    """
    healthy = _StandInModel([_StandInInteraction(_StandInOeqConv(fused=True))])
    sites = audit.assert_vendor_kernels(healthy, audit.OEQ)
    assert [site.path for site in sites] == ["interactions.0.conv_tp"]

    unwrapped = _StandInModel([_StandInInteraction(_StandInOeqConv(fused=False))])
    with pytest.raises(AssertionError) as complaint:
        audit.assert_vendor_kernels(unwrapped, audit.OEQ)
    assert "with_oeq_conv_fusion" in str(complaint.value)

    unfused_block = _StandInModel(
        [_StandInInteraction(_StandInOeqConv(fused=True), fused=False)]
    )
    with pytest.raises(AssertionError) as complaint:
        audit.assert_vendor_kernels(unfused_block, audit.OEQ)
    assert "conv_fusion" in str(complaint.value)


def test_the_audit_fails_a_site_that_is_installed_and_never_called():
    """Structure is not execution, and the audit checks both separately."""
    kernels = [_fake_fused_conv(), _fake_fused_conv()]
    model = _StandInModel([_StandInInteraction(k) for k in kernels])
    sites = audit.assert_vendor_kernels(model, audit.CUEQ)

    with audit.watch(sites) as counts:
        kernels[0](torch.zeros(1))
    assert counts == {"interactions.0.conv_tp": 1, "interactions.1.conv_tp": 0}

    with pytest.raises(AssertionError) as complaint:
        audit.assert_every_site_ran(sites, counts, audit.CUEQ)
    assert "interactions.1.conv_tp" in str(complaint.value)

    with audit.watch(sites) as counts:
        for kernel in kernels:
            kernel(torch.zeros(1))
    audit.assert_every_site_ran(sites, counts, audit.CUEQ)

    # The hooks must not outlive the block, or one test's audit would count
    # calls made by the next one.
    kernels[0](torch.zeros(1))
    assert counts == {"interactions.0.conv_tp": 1, "interactions.1.conv_tp": 1}


def test_the_conversion_whitelist_refuses_the_plain_anchor_and_both_converters_stop():
    """Why parity uses the ScaleShiftMACE anchor.

    ``extract_config_mace_model`` returns an ``{"error": ...}`` payload for a
    plain ``MACE`` rather than a config, and neither converter checks for it.
    What matters for a golden is that the refusal is a *stop*: both converters
    die on the malformed config instead of returning a model that would then
    be compared against a reference. Pinned so the choice of anchor is a
    stated contract rather than a workaround for a surprise.
    """
    from mace.tools.scripts_utils import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        extract_config_mace_model,
    )

    plain = _load_anchor(PLAIN_ANCHOR_MODEL)
    refused = extract_config_mace_model(plain)
    assert isinstance(refused, dict) and set(refused) == {"error"}

    for backend in audit.BACKENDS:
        with pytest.raises((TypeError, KeyError)):
            _convert(plain, backend, "cpu")

    accepted = extract_config_mace_model(_load_anchor())
    assert "error" not in accepted


def test_the_tolerance_row_is_the_shared_accelerated_one():
    """No second table, and no local number.

    Named explicitly because this file is the only consumer of the
    accelerated row: if it ever stops importing it, nothing else fails.
    """
    row = harness.FP64_ACCELERATED_BACKEND
    assert harness.TOLERANCES[row.name] is row
    assert (row.atol, row.rtol) == (1e-5, 0.0)
    assert row.atol > harness.FP64_CPU_REFERENCE.atol, (
        "the accelerated row exists because the comparison crosses a kernel "
        "and a device; if it is no looser than the CPU row, say so there"
    )


# ---------------------------------------------------------------------------
# Stand-ins for the CPU tests above
#
# Small on purpose: they carry only what the audit reads, so a change to the
# audit that stops reading something fails here rather than passing quietly.
# ---------------------------------------------------------------------------


class _FakeUniform1dKernel(torch.nn.Module):
    """Stands in for the fused implementation object cuEquivariance selects."""


class _FakeSegmentedPolynomial(torch.nn.Module):
    """Stands in for ``cuet.SegmentedPolynomial`` with the fused method."""

    def __init__(self):
        super().__init__()
        self.method = "uniform_1d"
        self.m = _FakeUniform1dKernel()

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return None


def _unguarded_fusion_wrapper(segmented):
    """A ``CueqConvFusionWrapper`` built without its constructor's guard.

    The guard is what MACE does about the naive downgrade today. Going around
    it is the point: the audit has to hold on its own, so that removing or
    weakening the guard shows up as a failing test rather than as a golden
    that silently starts pinning the naive path.
    """
    from mace.modules.wrapper_ops import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        CueqConvFusionWrapper,
    )

    class _Unguarded(CueqConvFusionWrapper):  # pylint: disable=too-few-public-methods
        def __init__(self, conv_tp):  # pylint: disable=super-init-not-called
            torch.nn.Module.__init__(self)
            self.conv_tp = conv_tp

        def forward(self, *args, **kwargs):
            # The real wrapper's forward takes MACE's four conv arguments and
            # rearranges them for cuEquivariance. Nothing here is testing that
            # rearrangement, only that a call is observable, so this passes
            # whatever it is given straight through.
            return self.conv_tp(*args, **kwargs)

    return _Unguarded(segmented)


def _fake_fused_conv():
    return _unguarded_fusion_wrapper(_FakeSegmentedPolynomial())


class _StandInOeqConv(torch.nn.Module):
    """Stands in for ``oeq.TensorProductConv`` after ``with_oeq_conv_fusion``.

    ``__module__`` is rewritten below because the audit identifies a vendor
    module by the package it comes from, which is the one property a fake
    cannot have by accident.
    """

    def __init__(self, fused=True):
        super().__init__()
        if fused:
            self.original_forward = self.forward

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return None


_StandInOeqConv.__module__ = "openequivariance.implementations.convolution"


class _StandInInteraction(torch.nn.Module):
    def __init__(self, conv_tp, fused=True):
        super().__init__()
        self.conv_tp = conv_tp
        if fused:
            self.conv_fusion = True


class _StandInModel(torch.nn.Module):
    def __init__(self, interactions):
        super().__init__()
        self.interactions = torch.nn.ModuleList(interactions)
        self.products = torch.nn.ModuleList([])
