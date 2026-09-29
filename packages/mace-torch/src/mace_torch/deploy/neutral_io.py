"""A neutral artifact into a v1 model, through the ordinary checkpoint path.

The artifact says what the model was, in the terms its source used. This
module turns that into a v1 configuration, builds the model the configuration
describes with the same function a training run uses, fills every operator's
canonical tensors from the artifact, and loads them the way a checkpoint is
loaded. Nothing here reads a pickle or imports the legacy package.

Three rules make it a conversion rather than an approximation:

* **Every recorded setting is read.** A configuration key this module does not
  map is an error, and so is a value the model stage cannot build. Building the
  nearest model instead would carry the weights into a different architecture.
* **The contraction is projected against the basis it was trained with**,
  which the artifact carries. The projection is exact; it is checked on every
  call and its result must have exactly the path counts the model expects.
* **Constants the v1 model computes rather than stores are compared**, not
  trusted: the radial basis frequencies, the cutoff order and the repulsion's
  screening constants. A source whose constants differ is refused, since v1
  would silently use its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mace_core.clebsch_gordan.conversion import BasisConversionError, full_to_reduced
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.metadata import (
    ConfigRecord,
    HeadSummary,
    ModelMetadata,
    ParentModel,
    Provenance,
)
from mace_core.observables import ObservableCatalogue
from mace_core.weights.neutral_format import NeutralArtifact, read_neutral
from torch import Tensor

from mace_torch import __version__
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import canonical_state, load_canonical_state

__all__ = [
    "ImportedModel",
    "NeutralImportError",
    "import_neutral",
    "resolved_config",
]

#: The model spelling each family of artifact becomes.
_SPELLINGS = {"plain": "plain", "scale_shift": "scale_shift"}

#: Why a family that extracted cleanly still cannot be built here.
_UNBUILT_FAMILIES = {
    "dielectric": ("its dipole and polarizability readouts have no v1 counterpart yet"),
}

#: What the model reads out once converted. Stress needs nothing stored; it is
#: the energy's strain derivative, declared so the converted model can report
#: it and the verification can compare it.
_OBSERVABLES = ("energy", "forces", "stress")

#: The readout the last layer must have been for the v1 readout to be it.
_LAST_READOUT = "NonLinearReadoutBlock"

#: Relative tolerance on constants v1 recomputes instead of storing.
_CONSTANT_TOLERANCE = 1e-12


class NeutralImportError(RuntimeError):
    """The artifact describes a model this cannot build faithfully."""


@dataclass(frozen=True)
class ImportedModel:
    """A converted model, ready to be written or used.

    Attributes:
        engine: The model in its derivative engine, every weight loaded.
        config: The v1 configuration it was built from.
        metadata: The record a v1 checkpoint carries, naming the source.
        heads: The head names, in the order of every per-head tensor.
        z_table: The elements, in the order of every per-element tensor.
    """

    engine: DerivativeEngine
    config: ResolvedConfig
    metadata: ModelMetadata
    heads: tuple[str, ...]
    z_table: AtomicNumberTable


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def resolved_config(artifact: NeutralArtifact) -> ResolvedConfig:
    """The v1 configuration of the model the artifact records.

    Raises:
        NeutralImportError: If a recorded setting has no v1 field, or a value
            the configuration cannot hold, naming each.
    """
    sidecar = artifact.sidecar
    family = sidecar.family
    if family in _UNBUILT_FAMILIES:
        raise NeutralImportError(
            f"the artifact is a {family} model, which cannot be built here: "
            f"{_UNBUILT_FAMILIES[family]}."
        )
    recorded = dict(sidecar.config)
    read: set[str] = set()

    def take(key: str) -> Any:
        if key not in recorded:
            raise NeutralImportError(
                f"the artifact's configuration has no {key!r}, which it takes "
                f"to rebuild the model. It was written by something that did "
                f"not record the whole model."
            )
        read.add(key)
        return recorded[key]

    refusals: list[str] = []
    hidden = Irreps.parse(str(take("hidden_irreps")))
    multiplicities = {multiplicity for multiplicity, _ in hidden.terms}
    if len(multiplicities) != 1:
        refusals.append(
            f"hidden_irreps is {hidden}, with different channel counts per irrep"
        )
    for key, allowed in (
        ("use_so3", False),
        ("embedding_specs", None),
    ):
        if take(key) != allowed:
            refusals.append(f"{key} is {recorded[key]!r}, which v1 has no field for")
    if take("readout_cls") != _LAST_READOUT:
        refusals.append(
            f"the last readout is a {recorded['readout_cls']}, and v1 builds a "
            f"{_LAST_READOUT} there"
        )
    # A backend setting of the source, not part of the model it computes.
    take("cueq_config")
    # Checked against the ops rather than trusted: each contraction records the
    # basis it was actually trained in.
    take("use_reduced_cg")
    # Rebuilt from the tensors and the element table, not configured.
    for key in (
        "num_elements",
        "atomic_numbers",
        "atomic_energies",
        "atomic_inter_scale",
        "atomic_inter_shift",
        "heads",
        "avg_num_neighbors",
    ):
        if key in recorded:
            take(key)
    if refusals:
        raise NeutralImportError(
            "the artifact records a model v1 cannot build: " + "; ".join(refusals) + "."
        )

    transform = take("distance_transform")
    model = {
        "model": _SPELLINGS[family],
        "observables": list(_OBSERVABLES),
        "r_max": float(take("r_max")),
        "num_interactions": int(take("num_interactions")),
        "num_channels": multiplicities.pop(),
        "hidden_irreps": "+".join(str(irrep) for _, irrep in hidden.terms),
        "max_ell": int(take("max_ell")),
        "correlation": int(take("correlation")),
        "interaction": take("interaction_cls"),
        "interaction_first": take("interaction_cls_first"),
        "use_agnostic_product": bool(take("use_agnostic_product")),
        "radial_type": take("radial_type"),
        "num_radial_basis": int(take("num_bessel")),
        "num_cutoff_basis": int(take("num_polynomial_cutoff")),
        "distance_transform": "None" if transform is None else str(transform),
        "apply_cutoff": bool(take("apply_cutoff")),
        "radial_mlp": [int(width) for width in take("radial_MLP")],
        "pair_repulsion": bool(take("pair_repulsion")),
        "edge_irreps": take("edge_irreps"),
        "use_edge_irreps_first": bool(take("use_edge_irreps_first")),
        "clebsch_gordan_basis": "reduced",
        "readout": {
            "mlp_irreps": str(take("MLP_irreps")),
            "gate": str(take("gate")),
            "last_only": bool(take("use_last_readout_only")),
            "from_embedding": bool(take("use_embedding_readout")),
        },
    }
    if family == "plain":
        model["scaling"] = "none"
    unread = sorted(set(recorded) - read)
    if unread:
        raise NeutralImportError(
            f"the artifact records settings this does not map: {unread}. "
            f"Converting without them would build a model that ignores them."
        )
    # Each head declared with the energies it was trained with, as a table: they
    # are an answer already, and a checkpoint read back takes its heads and
    # their energies from here.
    heads = {
        head: {"e0s": {"table": {"values": values}}}
        for head, values in isolated_atom_energies(artifact).items()
    }
    return ResolvedConfig.model_validate({"model": model, "data": {"heads": heads}})


def isolated_atom_energies(artifact: NeutralArtifact) -> dict[str, dict[int, float]]:
    """Each head's isolated-atom energies, by atomic number.

    Raises:
        NeutralImportError: If the table's shape does not match the heads and
            the element table.
    """
    heads = tuple(artifact.sidecar.heads)
    numbers = [int(z) for z in artifact.sidecar.config["atomic_numbers"]]
    table = artifact.tensor("energy.atomic_energies", "values").astype(np.float64)
    if table.shape != (len(heads), len(numbers)):
        raise NeutralImportError(
            f"the isolated-atom table is {table.shape}, and {len(heads)} head(s) "
            f"over {len(numbers)} element(s) needs {(len(heads), len(numbers))}."
        )
    return {
        head: {z: float(table[row][column]) for column, z in enumerate(numbers)}
        for row, head in enumerate(heads)
    }


# ---------------------------------------------------------------------------
# The tensors
# ---------------------------------------------------------------------------


def _as(value: np.ndarray, like: Tensor) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(value)).to(like.dtype)
    if tensor.shape != like.shape:
        raise NeutralImportError(
            f"a tensor of shape {tuple(tensor.shape)} arrived where the model "
            f"holds {tuple(like.shape)}."
        )
    return tensor


def _contraction(
    artifact: NeutralArtifact, index: int, expected: dict[str, Tensor]
) -> dict[str, Tensor]:
    """One product block's contractions as the one fused canonical array.

    Body order outermost, output irreps in declaration order within it, which
    is the order the model's own canonical form has them.
    """
    names = sorted(
        (
            name
            for name in artifact.sidecar.ops
            if name.startswith(f"products.{index}.contraction.")
        ),
        key=lambda name: list(artifact.sidecar.ops).index(name),
    )
    specs = [artifact.sidecar.ops[name] for name in names]
    correlations = {int(spec.descriptor["correlation"]) for spec in specs}
    if len(correlations) != 1:
        raise NeutralImportError(
            f"product {index} mixes correlations {sorted(correlations)}."
        )
    correlation = correlations.pop()
    pieces: list[np.ndarray] = []
    for order in range(1, correlation + 1):
        for name, spec in zip(names, specs, strict=True):
            target = str(spec.descriptor["target"])
            irreps_in = str(spec.descriptor["irreps_in"])
            if spec.descriptor.get("zeroed", {}).get(str(order)):
                raise NeutralImportError(
                    f"{name} fixes body order {order} at zero. v1 has no frozen "
                    f"body order: the converted model would compute the same "
                    f"and a fine-tune of it would train that order."
                )
            weights = artifact.tensor(name, f"weights.{order}")
            source = artifact.tensor(name, f"basis.{order}")
            full = full_symmetric_tensor_product_basis(irreps_in, order, target)[
                target
            ].shape[0]
            reduced = reduced_symmetric_tensor_product_basis(irreps_in, order, target)[
                target
            ].shape[0]
            recorded = weights.shape[1]
            wanted = {"full": full, "reduced": reduced}[spec.clebsch_gordan_basis or ""]
            if recorded != wanted:
                raise NeutralImportError(
                    f"{name} says its body order {order} is in the "
                    f"{spec.clebsch_gordan_basis} basis, which has {wanted} "
                    f"path(s) for {target!r} over {irreps_in!r}, and it holds "
                    f"{recorded}."
                )
            try:
                pieces.append(
                    full_to_reduced(
                        weights.astype(np.float64),
                        irreps_in,
                        order,
                        target,
                        source=source,
                    )
                )
            except BasisConversionError as error:
                raise NeutralImportError(
                    f"{name}, body order {order}: {error}"
                ) from error
    counts = [piece.shape[1] for piece in pieces]
    if counts != [int(n) for n in expected["path_counts"]]:
        raise NeutralImportError(
            f"product {index} converts to {counts} path(s) per body order and "
            f"irrep, and the model expects {expected['path_counts'].tolist()}. "
            f"A mismatch would load the right number of weights into the "
            f"wrong paths."
        )
    return {
        "weight": _as(np.concatenate(pieces, axis=1), expected["weight"]),
        "path_counts": expected["path_counts"].clone(),
    }


def _linear(artifact: NeutralArtifact, op: str, expected: dict[str, Tensor]):
    return {
        "weight": _as(artifact.tensor(op, "weight"), expected["weight"]),
        "bias": _as(artifact.tensor(op, "bias"), expected["bias"]),
    }


def _state(
    artifact: NeutralArtifact, engine: DerivativeEngine
) -> dict[str, dict[str, Tensor]]:
    """Every canonical tensor the model holds, filled from the artifact."""
    expected = canonical_state(engine)
    backbone = "backbone.backbone."
    outputs = "backbone.outputs."
    ops = artifact.sidecar.ops
    state: dict[str, dict[str, Tensor]] = {}
    for path, tensors in expected.items():
        if path == f"{backbone}node_embedding":
            state[path] = _linear(artifact, "node_embedding", tensors)
        elif path.startswith(f"{backbone}interactions."):
            index, _, rest = path.removeprefix(f"{backbone}interactions.").partition(
                "."
            )
            source = f"interactions.{index}"
            if rest == "body":
                value = ops[source].descriptor["avg_num_neighbors"]
                state[path] = {
                    "neighbours": torch.tensor(float(value)).to(
                        tensors["neighbours"].dtype
                    )
                }
            elif rest in {"body.linear_up", "body.linear"}:
                state[path] = _linear(
                    artifact, f"{source}.{rest.removeprefix('body.')}", tensors
                )
            elif rest == "body.radial":
                state[path] = {
                    name: _as(artifact.tensor(f"{source}.radial", name), value)
                    for name, value in tensors.items()
                }
            elif rest == "skip":
                state[path] = {
                    "weight": _as(
                        artifact.tensor(f"{source}.skip", "weight"), tensors["weight"]
                    )
                }
            else:
                raise NeutralImportError(f"the model holds {path}, which no op fills.")
        elif path.startswith(f"{backbone}products."):
            index, _, rest = path.removeprefix(f"{backbone}products.").partition(".")
            if rest == "contraction":
                state[path] = _contraction(artifact, int(index), tensors)
            elif rest == "linear":
                state[path] = _linear(artifact, f"products.{index}.linear", tensors)
            else:
                raise NeutralImportError(f"the model holds {path}, which no op fills.")
        elif path == f"{outputs}energy_head":
            state[path] = _energy_constants(artifact, tensors)
        elif path.startswith(f"{outputs}heads.energy.readouts."):
            state[path] = _linear(
                artifact,
                "readouts." + path.removeprefix(f"{outputs}heads.energy.readouts."),
                tensors,
            )
        else:
            raise NeutralImportError(f"the model holds {path}, which no op fills.")
    return state


def _energy_constants(artifact: NeutralArtifact, expected: dict[str, Tensor]):
    table = artifact.tensor("energy.atomic_energies", "values").astype(np.float64)
    if "energy.scale_shift" in artifact.sidecar.ops:
        scale = artifact.tensor("energy.scale_shift", "scale")
        shift = artifact.tensor("energy.scale_shift", "shift")
    else:
        scale = np.ones(expected["scale"].shape)
        shift = np.zeros(expected["shift"].shape)
    return {
        "e0_table": torch.from_numpy(np.ascontiguousarray(table)),
        "scale": _as(
            np.broadcast_to(scale, expected["scale"].shape), expected["scale"]
        ),
        "shift": _as(
            np.broadcast_to(shift, expected["shift"].shape), expected["shift"]
        ),
    }


def _check_constants(artifact: NeutralArtifact, config: ResolvedConfig, engine) -> None:
    """The constants v1 recomputes, compared with the ones the source used."""
    ops = artifact.sidecar.ops
    refusals: list[str] = []

    frequencies = artifact.tensor("radial_basis", "weights").astype(np.float64)
    analytic = math.pi * np.arange(1, frequencies.size + 1) / config.model.r_max
    if not np.allclose(frequencies, analytic, rtol=_CONSTANT_TOLERANCE, atol=0.0):
        refusals.append(
            "the radial basis frequencies are not pi * n / r_max, so the source "
            "trained them, and v1 computes them from the cutoff"
        )
    order = int(ops["cutoff"].descriptor["p"])
    if order != config.model.num_cutoff_basis:
        refusals.append(
            f"the cutoff envelope has order {order} and the configuration "
            f"{config.model.num_cutoff_basis}"
        )
    if "pair_repulsion" in ops:
        repulsion = dict(engine.get_submodule("backbone.repulsion").named_buffers())
        pairs = {
            "c": "screening_coefficients",
            "covalent_radii": "covalent_radii",
            "a_exp": "screening_length_exponent",
            "a_prefactor": "screening_length_prefactor",
        }
        for source, name in pairs.items():
            # Flattened: a scalar can arrive as one element rather than as a
            # zero-dimensional array, and that is the same constant.
            recorded = artifact.tensor("pair_repulsion", source).astype(np.float64)
            recorded = recorded.reshape(-1)
            held = repulsion[name].detach().cpu().numpy().astype(np.float64)
            held = held.reshape(-1)
            if recorded.shape != held.shape or not np.allclose(
                recorded, held, rtol=_CONSTANT_TOLERANCE, atol=0.0
            ):
                refusals.append(f"the repulsion's {source} differs from v1's {name}")
        if int(ops["pair_repulsion"].descriptor["p"]) != order:
            refusals.append(
                "the repulsion's envelope order differs from the cutoff's, and "
                "v1 uses one order for both"
            )
    if refusals:
        raise NeutralImportError(
            "the artifact's constants are not the ones v1 would use: "
            + "; ".join(refusals)
            + "."
        )


# ---------------------------------------------------------------------------
# The whole import
# ---------------------------------------------------------------------------


def import_neutral(
    source: str | Path | NeutralArtifact,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig | None = None,
    basis: str = "reduced",
) -> ImportedModel:
    """Build the v1 model a neutral artifact records, with every weight loaded.

    Args:
        source: The artifact, or the path of either of its files.
        catalogue: The observable declarations.
        precision: What the model computes in. Defaults to the training
            default.
        basis: The Clebsch-Gordan basis to import into. Only ``"reduced"``:
            a v1 model holds that basis, and the other direction is refused.

    Raises:
        NeutralImportError: If the artifact records a model v1 cannot build,
            constants v1 would not use, or weights that do not fit.
    """
    from mace_torch.train.model_stage import DEFAULT_PRECISION, build_model

    if basis != "reduced":
        raise NeutralImportError(
            f"the import was asked for the {basis!r} basis, and a v1 model holds "
            f"the reduced one. Going from reduced to full is under-determined: "
            f"it picks one of infinitely many weight vectors that compute the "
            f"same function. Import into the reduced basis, and export the full "
            f"one separately if an external implementation needs it."
        )
    artifact = source if isinstance(source, NeutralArtifact) else read_neutral(source)
    config = resolved_config(artifact)
    heads = tuple(artifact.sidecar.heads)
    numbers = [int(z) for z in artifact.sidecar.config["atomic_numbers"]]
    e0s = isolated_atom_energies(artifact)
    scale = (
        artifact.tensor("energy.scale_shift", "scale")
        if "energy.scale_shift" in artifact.sidecar.ops
        else np.ones(1)
    )
    shift = (
        artifact.tensor("energy.scale_shift", "shift")
        if "energy.scale_shift" in artifact.sidecar.ops
        else np.zeros(1)
    )
    average = {
        float(spec.descriptor["avg_num_neighbors"])
        for name, spec in artifact.sidecar.ops.items()
        if spec.op_kind == "interaction"
    }
    if len(average) != 1:
        raise NeutralImportError(
            f"the interactions record different neighbour counts {sorted(average)}, "
            f"and v1 normalizes every layer by one."
        )
    engine, _ = build_model(
        config,
        catalogue,
        z_table=AtomicNumberTable(numbers),
        heads=heads,
        e0s=ResolvedE0s(e0s),
        statistics=DatasetStatistics(
            avg_num_neighbors=average.pop(),
            mean=float(np.asarray(shift).reshape(-1)[0]),
            std=float(np.asarray(scale).reshape(-1)[0]),
        ),
        precision=precision or DEFAULT_PRECISION,
        initialize=False,
    )
    _check_constants(artifact, config, engine)
    load_canonical_state(engine, _state(artifact, engine))
    return ImportedModel(
        engine=engine,
        config=config,
        metadata=_metadata(artifact, config, e0s),
        heads=heads,
        z_table=AtomicNumberTable(numbers),
    )


def _metadata(
    artifact: NeutralArtifact, config: ResolvedConfig, e0s: dict[str, dict[int, float]]
) -> ModelMetadata:
    from ase.data import chemical_symbols
    from mace_core.config.provenance import e0_details

    provenance = artifact.sidecar.provenance
    origin = "recorded no heads" if provenance.headless else "recorded its heads"
    return ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(code_version=__version__),
        heads={
            head: HeadSummary(
                e0=e0_details(
                    config.data.heads[head].e0s,
                    {chemical_symbols[z]: energy for z, energy in values.items()},
                )
            )
            for head, values in e0s.items()
        },
        notes=(
            f"Converted from a {provenance.source_class} checkpoint written by "
            f"version {provenance.source_version}, by converter version "
            f"{provenance.converter_version}. The source {origin}."
        ),
        parents=[
            ParentModel(
                role="initial_weights",
                name=f"{provenance.source_file} (sha256 {provenance.source_sha256})",
            )
        ],
    )
