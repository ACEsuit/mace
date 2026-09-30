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
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.canonical import linear_weight_table
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
from mace_torch.electrostatics.reference.gto_utils import gto_basis_kspace_cutoff
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import canonical_state, load_canonical_state

__all__ = [
    "ImportedModel",
    "NeutralImportError",
    "import_neutral",
    "resolved_config",
]

#: The model spelling each family of artifact becomes.
_SPELLINGS = {
    "plain": "plain",
    "scale_shift": "scale_shift",
    "polar": "polar",
    "dielectric": "dielectric",
}


#: What the model reads out once converted. Stress needs nothing stored; it is
#: the energy's strain derivative, declared so the converted model can report
#: it and the verification can compare it.
_OBSERVABLES = ("energy", "forces", "stress")

#: What a dielectric model reads out once converted.
_DIELECTRIC_OBSERVABLES = ("dipole", "polarizability")


def _dielectric_settings(take, readout_class: str, refusals: list[str]) -> None:
    """The dielectric model's two flags, and its readout, checked."""
    if not take("use_polarizability") or take("only_dipole"):
        refusals.append(
            "the dielectric model reads out a dipole alone, and v1 builds the "
            "one with its polarizability"
        )
    if readout_class != "NonLinearDipolePolarReadoutBlock":
        refusals.append(
            f"the dielectric model's last readout is a {readout_class}, and v1 "
            f"builds NonLinearDipolePolarReadoutBlock there"
        )


#: The readouts the last layer may have been, and whether each is the biased
#: one.
_LAST_READOUTS = {"NonLinearReadoutBlock": False, "NonLinearBiasReadoutBlock": True}


def _graph_features(specs: list[list[Any]]) -> list[dict[str, Any]]:
    """The frozen tree's embedding specs, ``[name, spec]`` pairs in the order
    it concatenates them, as the model's graph features."""
    return [
        {
            "name": name,
            "kind": spec["type"],
            "embedding_dim": int(spec["emb_dim"]),
            "num_classes": int(spec.get("num_classes", 0)),
            "input_dim": int(spec.get("in_dim", 1)),
            "per": "graph" if spec.get("per", "graph") == "graph" else "atom",
            "offset": int(spec.get("offset", 0)),
            "use_bias": bool(spec.get("use_bias", True)),
        }
        for name, spec in specs
    ]


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
    if take("use_so3") is not False:
        refusals.append(
            f"use_so3 is {recorded['use_so3']!r}, which v1 has no field for"
        )
    readout_class = take("readout_cls")
    if family == "dielectric":
        _dielectric_settings(take, readout_class, refusals)
        readout_class = "NonLinearReadoutBlock"
    # The frozen tree reads the class off the last readout, and a one-layer
    # model's only readout is the linear first-layer one.
    one_layer = int(recorded.get("num_interactions", 0)) == 1
    if readout_class not in _LAST_READOUTS and not (
        one_layer and readout_class == "LinearReadoutBlock"
    ):
        refusals.append(
            f"the last readout is a {recorded['readout_cls']}, and v1 builds "
            f"{' or '.join(_LAST_READOUTS)} there"
        )
    graph_features = _graph_features(take("embedding_specs") or [])
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
        "observables": list(
            _DIELECTRIC_OBSERVABLES if family == "dielectric" else _OBSERVABLES
        ),
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
            "bias": _LAST_READOUTS.get(readout_class, False),
        },
        "graph_features": graph_features,
    }
    if one_layer:
        # No later layer and no gated readout to build, and the frozen tree's
        # reader fills both from the one layer there is: its interaction is
        # the first layer's and its gate that of a linear readout, None.
        # Neither builds anything, so the defaults stand in for them.
        model["interaction"] = "RealAgnosticResidualInteractionBlock"
        model["readout"]["gate"] = "silu"
    if family == "plain":
        model["scaling"] = "none"
    sections: dict[str, Any] = {}
    if family == "polar":
        model["polar"], sections["electrostatics"] = _polar_settings(take)
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
        head: {"e0s": {"kind": "table", "values": values}} if values else {}
        for head, values in isolated_atom_energies(artifact).items()
    }
    return ResolvedConfig.model_validate(
        {"model": model, "data": {"heads": heads}, **sections}
    )


#: The one field-update and field-readout block the frozen tree has, by the
#: class name it records and the name v1 builds it under.
_FIELD_UPDATES = {"AgnosticEmbeddedOneBodyVariableUpdate": "embedded_one_body"}
_FIELD_READOUTS = {"OneBodyMLPFieldReadout": "one_body_mlp"}
_POTENTIAL_EMBEDDINGS = {"AgnosticChargeBiasedLinearPotentialEmbedding"}


def _polar_settings(take) -> tuple[dict[str, Any], dict[str, Any]]:
    """The charge-aware settings: the model's ``polar`` section, and the
    ``electrostatics`` one the solver is set up from.

    Three settings the frozen tree records change nothing it computes, and are
    checked rather than carried: the potentials are returned or not, the
    field is divided by one, and the update's nonlinearity class is discarded
    by the block that reads it.
    """
    refusals = []
    if not take("keep_last_layer_irreps"):
        refusals.append(
            "keep_last_layer_irreps is False, and the charge-aware model reads "
            "its dipoles off the last layer's degree one features"
        )
    take("return_electrostatic_potentials")
    if float(take("field_norm_factor")) != 1.0:
        refusals.append("field_norm_factor is not one, and v1 divides by one")
    update = take("fixedpoint_update_config") or {}
    readout = take("field_readout_config") or {}
    kind = update.get("type", "AgnosticEmbeddedOneBodyVariableUpdate")
    embedding = update.get(
        "potential_embedding_cls", "AgnosticChargeBiasedLinearPotentialEmbedding"
    )
    reading = readout.get("type", "OneBodyMLPFieldReadout")
    for what, name, known in (
        ("field update", kind, _FIELD_UPDATES),
        ("potential embedding", embedding, _POTENTIAL_EMBEDDINGS),
        ("field readout", reading, _FIELD_READOUTS),
    ):
        if name not in known:
            refusals.append(f"the {what} is {name!r}, and v1 builds {sorted(known)}")
    if refusals:
        raise NeutralImportError(
            "the artifact records a charge-aware model v1 cannot build: "
            + "; ".join(refusals)
            + "."
        )
    polar = {
        "multipole_max_l": int(take("atomic_multipoles_max_l")),
        "multipole_width": float(take("atomic_multipoles_smearing_width")),
        "feature_max_l": int(take("field_feature_max_l")),
        "feature_widths": [float(width) for width in take("field_feature_widths")],
        "feature_norms": [float(norm) for norm in take("field_feature_norms")],
        "num_recursion_steps": int(take("num_recursion_steps")),
        "feature_self_interaction": bool(take("field_si")),
        "energy_self_interaction": bool(take("include_electrostatic_self_interaction")),
        "add_local_electron_energy": bool(take("add_local_electron_energy")),
        "quadrupole_feature_corrections": bool(take("quadrupole_feature_corrections")),
        "field_update": _FIELD_UPDATES[kind],
        "field_readout": _FIELD_READOUTS[reading],
    }
    # The factor that gives the cutoff the source sums to. The frozen tree
    # records a factor and sums to a stored cutoff, and the two disagree on
    # the published models: they hold the cutoff of a factor of one.
    take("kspace_cutoff_factor")
    heuristic = gto_basis_kspace_cutoff(
        [polar["multipole_width"], *polar["feature_widths"]],
        max(polar["multipole_max_l"], polar["feature_max_l"]),
    )
    electrostatics = {
        "enabled": True,
        "kspace_cutoff_factor": float(take("kspace_cutoff")) / heuristic,
        # A molecule summed in real space and every other structure
        # periodically, which is what the frozen tree gives a structure
        # evaluated alone. It decides per batch instead, and boxes a molecule
        # batched with anything periodic.
        "periodicity_profile": "per_structure",
    }
    return polar, electrostatics


def isolated_atom_energies(artifact: NeutralArtifact) -> dict[str, dict[int, float]]:
    """Each head's isolated-atom energies, by atomic number.

    Raises:
        NeutralImportError: If the table's shape does not match the heads and
            the element table.
    """
    heads = tuple(artifact.sidecar.heads)
    numbers = [int(z) for z in artifact.sidecar.config["atomic_numbers"]]
    if "energy.atomic_energies" not in artifact.sidecar.ops:
        # A model that reads out no energy has none.
        return {head: {} for head in heads}
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
    tensor = torch.from_numpy(np.array(value, order="C")).to(like.dtype)
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


#: Each distance transform's tensors: the model's name for one, the
#: artifact's.
_TRANSFORM_TENSORS = {
    "agnesi_transform": {
        "amplitude": "a",
        "exponent_q": "q",
        "exponent_p": "p",
        "covalent_radii": "covalent_radii",
    },
    "soft_transform": {"steepness": "alpha", "covalent_radii": "covalent_radii"},
}

#: The repulsion's tensors: the model's name for one, the artifact's.
_REPULSION_TENSORS = {
    "screening_coefficients": "c",
    "covalent_radii": "covalent_radii",
    "screening_length_exponent": "a_exp",
    "screening_length_prefactor": "a_prefactor",
}

#: The nonlinear interaction's linears, each an op of the same name.
_NONLINEAR_LINEARS = frozenset(
    {"linear_up", "linear_res", "source", "target", "linear_mid", "linear_out"}
)


def _state(
    artifact: NeutralArtifact, engine: DerivativeEngine
) -> dict[str, dict[str, Tensor]]:
    """Every canonical tensor the model holds, filled from the artifact."""
    expected = canonical_state(engine)
    backbone = "backbone.backbone."
    outputs = "backbone.outputs."
    ops = artifact.sidecar.ops
    state: dict[str, dict[str, Tensor]] = {}
    dielectric: dict[str, dict[str, Tensor]] | None = None
    for path, tensors in expected.items():
        if path == f"{backbone}node_embedding":
            state[path] = _linear(artifact, "node_embedding", tensors)
        elif path.startswith(f"{backbone}interactions."):
            index, _, rest = path.removeprefix(f"{backbone}interactions.").partition(
                "."
            )
            source = f"interactions.{index}"
            if rest == "":
                # Scalars, which the tensor file holds with one axis.
                state[path] = {
                    name: _as(artifact.tensor(source, name).reshape(()), value)
                    for name, value in tensors.items()
                }
            elif rest == "body":
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
            elif rest in _NONLINEAR_LINEARS or (rest == "skip" and "bias" in tensors):
                state[path] = _linear(artifact, f"{source}.{rest}", tensors)
            elif rest in {"body.radial", "body.density", "radial", "density"}:
                op = f"{source}.{rest.removeprefix('body.')}"
                state[path] = {
                    name: _as(artifact.tensor(op, name), value)
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
        elif path == f"{backbone}radial":
            state[path] = {
                "frequencies": _as(
                    artifact.tensor("radial_basis", "weights"), tensors["frequencies"]
                ),
                "prefactor": _as(
                    artifact.tensor("radial_basis", "prefactor").reshape(()),
                    tensors["prefactor"],
                ),
            }
        elif path == "backbone.repulsion":
            state[path] = {}
            for name, source in _REPULSION_TENSORS.items():
                value = artifact.tensor("pair_repulsion", source)
                if tensors[name].dim() == 0:
                    value = value.reshape(())
                state[path][name] = _as(value, tensors[name])
        elif path == f"{backbone}distance_transform":
            names = _TRANSFORM_TENSORS[ops["distance_transform"].op_kind]
            state[path] = {}
            for name, source in names.items():
                value = artifact.tensor("distance_transform", source)
                if tensors[name].dim() == 0:
                    value = value.reshape(())
                state[path][name] = _as(value, tensors[name])
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
        elif path == f"{backbone}graph_features":
            state[path] = {
                name: _as(artifact.tensor("graph_features", name), value)
                for name, value in tensors.items()
            }
        elif path == f"{outputs}embedding_readout":
            state[path] = _linear(artifact, "embedding_readout", tensors)
        elif path.removeprefix("backbone.") in ops and path.startswith("backbone."):
            state[path] = _polar_op(artifact, engine, path, tensors)
        elif path.startswith(f"{outputs}heads.") and artifact.sidecar.family == (
            "dielectric"
        ):
            if dielectric is None:
                dielectric = _dielectric_readouts(artifact, engine, expected)
            state[path] = dielectric[path]
        elif path.startswith(f"{outputs}heads.energy.readouts."):
            state[path] = _linear(
                artifact,
                "readouts." + path.removeprefix(f"{outputs}heads.energy.readouts."),
                tensors,
            )
        else:
            raise NeutralImportError(f"the model holds {path}, which no op fills.")
    return state


def _polar_op(
    artifact: NeutralArtifact, engine, path: str, expected: dict[str, Tensor]
) -> dict[str, Tensor]:
    """One op of the charge-aware half, by the kind the artifact records."""
    op = path.removeprefix("backbone.")
    kind = artifact.sidecar.ops[op].op_kind
    if kind == "linear":
        return _linear(artifact, op, expected)
    if kind == "layer_norm_mlp":
        return {
            name: _as(artifact.tensor(op, name), value)
            for name, value in expected.items()
        }
    if kind == "sparse_product":
        return {"weight": _sparse(artifact, op, engine.get_submodule(path), expected)}
    raise NeutralImportError(f"the model holds {path}, which a {kind} op cannot fill.")


def _sparse(artifact: NeutralArtifact, op: str, module, expected) -> Tensor:
    """A channel-pair product's path blocks, in the order the model holds them.

    The invariant products hold one block per pair of input terms, the
    modulation one per term of its first input against the scalars.
    """
    blocks = {
        name: artifact.tensor(op, name) for name in artifact.sidecar.ops[op].tensors
    }
    if hasattr(module, "paths"):
        order = [f"path.{left}.{right}" for _, _, left, right, _ in module.paths]
    else:
        order, offset = [], 0
        for multiplicity, dimension in module.spans:
            order.append(f"path.{offset}.0")
            offset += multiplicity * dimension
    missing = sorted(set(order) - set(blocks))
    extra = sorted(set(blocks) - set(order))
    if missing or extra:
        raise NeutralImportError(
            f"{op} holds the paths {sorted(blocks)} and the model's product "
            f"{order}: the artifact misses {missing} and adds {extra}."
        )
    flat = np.concatenate([blocks[name].reshape(-1) for name in order])
    return _as(flat, expected["weight"])


#: Which output copy of the frozen tree's dielectric readouts each head reads.
#: They map to ``2x0e+1x1o+1x2e``: the charge, the polarizability's scalar, the
#: dipole and the polarizability's ``2e``.
_DIELECTRIC_COPIES = {
    "charges": [0],
    "polarizability_sh": [1, 3],
    "atomic_dipoles": [2],
}


def _copies(irreps: str) -> list[tuple[str, int]]:
    """Each copy as its irrep and its channel within that irrep's terms."""
    seen: dict[str, int] = {}
    copies = []
    for multiplicity, irrep in Irreps.parse(irreps).terms:
        for _ in range(multiplicity):
            name = str(irrep)
            copies.append((name, seen.get(name, 0)))
            seen[name] = seen.get(name, 0) + 1
    return copies


def _entries(artifact: NeutralArtifact, op: str) -> dict[tuple[int, int], float]:
    """A linear op's weight for each (output copy, input copy) it joins."""
    descriptor = artifact.sidecar.ops[op].descriptor
    weight = artifact.tensor(op, "weight").astype(np.float64)
    table = linear_weight_table(descriptor["irreps_in"], descriptor["irreps_out"])
    return {key: float(weight[index]) for key, index in table.items()}


def _copied(
    entries: dict[tuple[int, int], float],
    irreps_in: str,
    irreps_out: str,
    in_map: list[int | None],
    out_map: list[int | None],
    expected: dict[str, Tensor],
) -> dict[str, Tensor]:
    """A linear written copy by copy from another's entries.

    ``in_map`` and ``out_map`` give, for each input and output copy, the
    source's copy it is, or ``None`` where the source has none, whose weight is
    then zero.
    """
    weight = np.zeros(expected["weight"].shape)
    for (out_copy, in_copy), index in linear_weight_table(
        irreps_in, irreps_out
    ).items():
        source = (out_map[out_copy], in_map[in_copy])
        if None not in source:
            weight[index] = entries.get(source, 0.0)
    return {
        "weight": _as(weight, expected["weight"]),
        "bias": expected["bias"].new_zeros(expected["bias"].shape),
    }


def _gate_roles(artifact: NeutralArtifact, layer: int, gate):
    """Where each copy of a head's gated readout sits in the frozen tree's.

    The frozen tree's middle is its scalars, one gate per gated channel and
    the gated channels, all its heads' together; each v1 head has its own,
    built from the same weights, with the sections it reads out.
    """
    sections = artifact.sidecar.ops[f"readouts.{layer}.first"].descriptor[
        "gate_sections"
    ]
    scalars = Irreps.parse(sections["scalars"]).dimension if sections["scalars"] else 0
    gated = Irreps.parse(sections["gated"]).terms if sections["gated"] else ()
    first: dict[tuple, int] = {("scalar", c): c for c in range(scalars)}
    gate_at = scalars
    gated_at = scalars + sum(multiplicity for multiplicity, _ in gated)
    for multiplicity, irrep in gated:
        for channel in range(multiplicity):
            first[("gate", str(irrep), channel)] = gate_at + channel
            first[("gated", str(irrep), channel)] = gated_at + channel
        gate_at += multiplicity
        gated_at += multiplicity
    second_in = artifact.sidecar.ops[f"readouts.{layer}.second"].descriptor["irreps_in"]
    second = {
        (("scalar", channel) if irrep == "0e" else ("gated", irrep, channel)): position
        for position, (irrep, channel) in enumerate(_copies(second_in))
    }
    terms = _copies(gate.gated_declaration) if gate.gated_declaration else []
    scalar_roles = [("scalar", channel) for _, channel in _copies(str(gate.scalars))]
    first_roles = (
        scalar_roles
        + [("gate", irrep, channel) for irrep, channel in terms]
        + [("gated", irrep, channel) for irrep, channel in terms]
    )
    second_roles = scalar_roles + [
        ("gated", irrep, channel) for irrep, channel in terms
    ]
    return [first.get(role) for role in first_roles], [
        second.get(role) for role in second_roles
    ]


def _dielectric_readouts(
    artifact: NeutralArtifact, engine, expected: dict[str, dict[str, Tensor]]
) -> dict[str, dict[str, Tensor]]:
    """Every head of the dielectric model, from the frozen tree's readouts.

    One of its readouts reads out all three quantities and slices them; v1
    gives each its own head. A linear map joins each output copy to input
    copies independently, so each head takes the rows of the copies it reads.
    """
    state: dict[str, dict[str, Tensor]] = {}
    for name, sources in _DIELECTRIC_COPIES.items():
        path = f"backbone.outputs.heads.{name}"
        head = engine.get_submodule(path)
        for position, layer in enumerate(head.reachable):
            grouped = head._grouped[position]
            identity: list[int | None] = list(range(len(_copies(grouped))))
            if f"readouts.{layer}" in artifact.sidecar.ops:
                key = f"{path}.readouts.{position}"
                state[key] = _copied(
                    _entries(artifact, f"readouts.{layer}"),
                    grouped,
                    head.spec.irreps,
                    identity,
                    list(sources),
                    expected[key],
                )
                continue
            readout = head.readouts[position]
            first_in, second_in = _gate_roles(artifact, layer, readout.gate)
            key = f"{path}.readouts.{position}"
            state[f"{key}.first"] = _copied(
                _entries(artifact, f"readouts.{layer}.first"),
                grouped,
                readout.gate.irreps_in,
                identity,
                first_in,
                expected[f"{key}.first"],
            )
            state[f"{key}.second"] = _copied(
                _entries(artifact, f"readouts.{layer}.second"),
                readout.gate.irreps_out,
                head.spec.irreps,
                second_in,
                list(sources),
                expected[f"{key}.second"],
            )
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


def _check_constants(artifact: NeutralArtifact, config: ResolvedConfig) -> None:
    """The settings v1 builds from the configuration, compared with the ones
    the source used."""
    ops = artifact.sidecar.ops
    refusals: list[str] = []

    order = int(ops["cutoff"].descriptor["p"])
    if order != config.model.num_cutoff_basis:
        refusals.append(
            f"the cutoff envelope has order {order} and the configuration "
            f"{config.model.num_cutoff_basis}"
        )
    if "pair_repulsion" in ops and int(ops["pair_repulsion"].descriptor["p"]) != order:
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
    _check_constants(artifact, config)
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
        elements=[
            chemical_symbols[int(z)]
            for z in sorted(artifact.sidecar.config["atomic_numbers"])
        ],
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
