"""The neutral weights format: safetensors for the numbers, JSON for their meaning.

A neutral artifact is two files side by side, ``<name>.safetensors`` and
``<name>.json``, and nothing in either is pickled. It is what a legacy
checkpoint becomes before any v1 code touches it, and it is written by a
script that runs against the legacy package and imports nothing from this one,
so the JSON below is the contract and this module is its reader and its
validator rather than the only thing that can produce it.

**The tensors** are flat-keyed ``<op>::<name>``, every one of them declared by
exactly one op in the sidecar. A tensor in the file that no op declares, or an
op naming a tensor the file lacks, is refused on read.

**The sidecar** holds:

* ``format`` and ``version``, so a reader refuses a file it was not written for.
* ``provenance``: which file it came from and its SHA-256, the class it was, the
  version of the package that pickled it and of the converter, and whether the
  source recorded its heads at all.
* ``family``: what kind of model it is, in this format's own words: ``plain``
  (an energy model with no scale and shift), ``scale_shift`` (one with them) or
  ``dielectric`` (dipoles and polarizabilities). A reader decides what to build
  from this and never from the source's class name.
* ``config``: everything it takes to rebuild the model, as the source recorded
  it, with classes and irreps as their names and arrays as lists.
* ``heads``, in the order their rows appear in every per-head tensor.
* ``dtype``: the float type the weights were trained in.
* ``ops``: one entry per operator, keyed by its place in the model, each an
  :class:`OpSpec`.
* ``derived``: every source tensor deliberately not carried, with why. A tensor
  the source held appears either under an op or here, so a reader can audit
  that nothing was dropped rather than trusting it.

**Op names** describe the model's structure, not any implementation's module
tree: ``node_embedding``, ``interactions.<i>.linear_up``,
``interactions.<i>.radial``, ``interactions.<i>.linear``,
``interactions.<i>.skip``, ``products.<i>.contraction.<irrep>``,
``products.<i>.linear``, ``readouts.<i>`` or ``readouts.<i>.first`` and
``.second``, ``energy.atomic_energies``, ``energy.scale_shift``,
``radial_basis``, ``cutoff``, ``pair_repulsion``.

**Weights are in the canonical layout** of :mod:`mace_core.kernels.canonical`:
``mul_ir``, output copies outermost, every normalization folded into the
number. The exception is the symmetric contraction, which an artifact may carry
in the basis it was trained in. That op says which with
``clebsch_gordan_basis`` and carries the basis itself as tensors, path axis
first, because weights are only meaningful against the basis they were trained
with and two full bases of the same space need not be the same basis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "FORMAT",
    "SEPARATOR",
    "VERSION",
    "NeutralArtifact",
    "NeutralFormatError",
    "NeutralSidecar",
    "OpSpec",
    "Provenance",
    "read_neutral",
    "write_neutral",
]

#: What the sidecar says it is.
FORMAT = "mace-neutral"

#: Bumped when the meaning of a field changes. A reader refuses any other.
VERSION = 1

#: Between an op's name and its tensor's. A dot would be ambiguous, since op
#: names are dotted paths themselves.
SEPARATOR = "::"

OpKind = Literal[
    "linear",
    "element_linear",
    "radial_mlp",
    "symmetric_contraction",
    "interaction",
    "atomic_energies",
    "scale_shift",
    "bessel_basis",
    "polynomial_cutoff",
    "zbl",
]


class NeutralFormatError(ValueError):
    """The files are not a neutral artifact this reader can trust."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OpSpec(_Strict):
    """One operator: what it is, how its tensors are to be read, and which.

    Attributes:
        op_kind: What the operator computes.
        schema_version: The version of the canonical layout its tensors are in,
            :data:`mace_core.kernels.canonical.KERNEL_SPEC_VERSION` when
            written.
        clebsch_gordan_basis: For a symmetric contraction, the basis its
            weights are written against.
        parametrization: For a symmetric contraction, ``"nested"`` when each
            body order has its own weight tensor, ``"fused"`` when they are one
            flat array.
        proj: The name of a tensor holding a change of parametrization, when
            the weights need one to mean what the basis says.
        descriptor: What the op needs to be rebuilt: irreps, widths, orders.
        tensors: The names of its tensors, each stored as ``<op>::<name>``.
    """

    op_kind: OpKind
    schema_version: str
    clebsch_gordan_basis: Literal["full", "reduced"] | None = None
    parametrization: Literal["nested", "fused"] | None = None
    proj: str | None = None
    descriptor: dict[str, Any] = {}
    tensors: tuple[str, ...] = ()


class Provenance(_Strict):
    """Where an artifact came from.

    Attributes:
        source_file: The source's file name.
        source_sha256: The SHA-256 of its bytes.
        source_class: The class the source was, by name.
        source_version: The version of the package that wrote the source, as
            that package reports it.
        converter_version: The version of what wrote this artifact.
        headless: The source recorded no heads, so the one head here was named
            by the converter. Tells a single-head model from one that predates
            heads.
    """

    source_file: str
    source_sha256: str
    source_class: str
    source_version: str
    converter_version: str
    headless: bool


class NeutralSidecar(_Strict):
    """The JSON half. See the module docstring for each field."""

    format: Literal["mace-neutral"]
    version: Literal[1]
    provenance: Provenance
    family: Literal["plain", "scale_shift", "dielectric"]
    config: dict[str, Any]
    heads: tuple[str, ...]
    dtype: Literal["float32", "float64"]
    ops: dict[str, OpSpec]
    derived: dict[str, str] = {}


@dataclass(frozen=True)
class NeutralArtifact:
    """Both halves, checked against each other."""

    sidecar: NeutralSidecar
    tensors: dict[str, np.ndarray]

    def tensor(self, op: str, name: str) -> np.ndarray:
        """One op's tensor.

        Raises:
            NeutralFormatError: Naming the op and the tensor, when either is
                absent.
        """
        spec = self.sidecar.ops.get(op)
        if spec is None:
            raise NeutralFormatError(
                f"the artifact has no op {op!r}. It has {sorted(self.sidecar.ops)}."
            )
        if name not in spec.tensors:
            raise NeutralFormatError(
                f"op {op!r} declares {list(spec.tensors)} and not {name!r}."
            )
        return self.tensors[f"{op}{SEPARATOR}{name}"]


def _paths(path: str | Path) -> tuple[Path, Path]:
    base = Path(path)
    if base.suffix in {".json", ".safetensors"}:
        base = base.with_suffix("")
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def write_neutral(
    path: str | Path, sidecar: NeutralSidecar, tensors: dict[str, np.ndarray]
) -> Path:
    """Write both halves, after checking they agree.

    Returns:
        The sidecar's path.

    Raises:
        NeutralFormatError: If the tensors and the declared ops disagree.
    """
    from safetensors.numpy import save_file

    _check_declared(sidecar, set(tensors))
    weights, document = _paths(path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: np.ascontiguousarray(value) for key, value in tensors.items()},
        str(weights),
    )
    document.write_text(
        json.dumps(sidecar.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    return document


def read_neutral(path: str | Path) -> NeutralArtifact:
    """Read and validate an artifact.

    Args:
        path: Either half, or the name without a suffix.

    Raises:
        NeutralFormatError: If a half is missing, the sidecar is not this
            format or version or does not validate, or the tensors and the
            declared ops disagree.
    """
    from pydantic import ValidationError
    from safetensors.numpy import load_file

    weights, document = _paths(path)
    for half in (weights, document):
        if not half.exists():
            raise NeutralFormatError(
                f"no {half.name} beside {path}. A neutral artifact is two "
                f"files, the tensors and the JSON that says what they are."
            )
    raw = json.loads(document.read_text())
    if raw.get("format") != FORMAT:
        raise NeutralFormatError(
            f"{document} says its format is {raw.get('format')!r}, and this "
            f"reads {FORMAT!r}."
        )
    if raw.get("version") != VERSION:
        raise NeutralFormatError(
            f"{document} is version {raw.get('version')!r}, and this reader "
            f"was written for {VERSION}."
        )
    try:
        sidecar = NeutralSidecar.model_validate(raw)
    except ValidationError as error:
        raise NeutralFormatError(f"{document} does not validate: {error}") from error
    tensors = load_file(str(weights))
    _check_declared(sidecar, set(tensors))
    return NeutralArtifact(sidecar=sidecar, tensors=tensors)


def _check_declared(sidecar: NeutralSidecar, present: set[str]) -> None:
    declared = {
        f"{op}{SEPARATOR}{name}"
        for op, spec in sidecar.ops.items()
        for name in spec.tensors
    }
    if declared != present:
        raise NeutralFormatError(
            f"the ops declare {len(declared)} tensor(s) and the file holds "
            f"{len(present)}. In the file and declared by no op: "
            f"{sorted(present - declared) or 'none'}. Declared and not in the "
            f"file: {sorted(declared - present) or 'none'}."
        )
