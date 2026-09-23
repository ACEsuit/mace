"""The ASE calculator for v1 models of the energy family.

It wraps :func:`mace_torch.deploy.load_deployed` rather than loading models
itself, builds each structure's graph with the same builder training uses, and
reads every result off the typed output. It holds no energy algebra: the
isolated-atom energies are inside the model, which reports its per-atom
energies both with them and without them.

**Results, as the frozen tree names them.** ``energy`` and ``free_energy``,
``forces``, ``stress`` in Voigt order, ``energies`` (each atom's energy with its
isolated-atom energy) and ``node_energy`` (without it), and with
``compute_atomic_stresses`` the per-atom ``stresses`` in Voigt order and
``virials``. A committee of models puts the mean in those keys and, for
``energy``, ``forces`` and ``stress``, the per-model values under ``_comm`` and
their population variance under ``_var``. Every key it writes is in
``implemented_properties``.

**Units.** The model computes in eV and Angstrom. ``energy_units_to_eV`` and
``length_units_to_A`` convert each result by its dimension: energies by E,
forces by E/L, stresses by E/L^3, virials by E and the Hessian by E/L^2. A
variance takes the square of its quantity's factor.
"""

from __future__ import annotations

import glob
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from mace_core.data.configuration import Configuration
from mace_core.outputs import MACEOutput
from torch import Tensor

from mace_torch.calculators.padding import (
    PaddingInfo,
    PaddingPolicy,
    output_rows,
    pad_batch,
    resolve_budget,
    unpad_outputs,
)
from mace_torch.data import collate_training
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.deploy.loader import DeployedModel, load_deployed
from mace_torch.models.outputs import ENERGY_OBSERVABLE
from mace_torch.nn import MACEBackbone
from mace_torch.serialization import FORMAT as CHECKPOINT_FORMAT

__all__ = ["ENSEMBLE_KEYS", "MACECalculator", "declared_properties"]

logger = logging.getLogger(__name__)

#: The results a committee also reports per model and as a spread. The frozen
#: tree adds ``dipole``, which belongs to the dipole families.
ENSEMBLE_KEYS = ("energy", "forces", "stress")

#: What ``atoms.info`` entries reach the graph by default, as the frozen tree
#: passes them: graph field, then info key.
DEFAULT_INFO_KEYS = {
    "total_spin": "spin",
    "total_charge": "charge",
    "external_field": "external_field",
}


def declared_properties(*, committee: bool, atomic_stresses: bool) -> list[str]:
    """Every result key a calculator so configured writes, and no other."""
    properties = ["energy", "free_energy", "energies", "node_energy", "forces"]
    properties.append("stress")
    if atomic_stresses:
        properties += ["stresses", "virials"]
    if committee:
        properties += [
            f"{key}{suffix}" for key in ENSEMBLE_KEYS for suffix in ("_comm", "_var")
        ]
    return properties


class MACECalculator(Calculator):
    """A v1 model, or a committee of them, as an ASE calculator.

    Args:
        model_paths: A checkpoint, a list of them, or a pattern such as
            ``"models/mace_*.json"`` that names a committee.
        models: Models already loaded, instead of paths.
        device: Where to evaluate.
        energy_units_to_eV: What one model energy unit is in eV.
        length_units_to_A: What one model length unit is in Angstrom.
        charges_key: The ``atoms.arrays`` entry passed as ``charges``.
        info_keys: Graph field to ``atoms.info`` key, for the entries that
            reach the model.
        arrays_keys: Graph field to ``atoms.arrays`` key, likewise.
        head: The head to evaluate, by name. Required when the model has
            several and none is called ``default``.
        pad_num_atoms: A fixed atom budget. Read from
            ``MACE_ASE_PAD_NUM_ATOMS`` when zero.
        pad_num_edges: A fixed edge budget. Read from
            ``MACE_ASE_PAD_NUM_EDGES`` when zero.
        padding: A padding policy, instead of the two budgets.
        compile_mode: Compile each model with static shapes in this
            ``torch.compile`` mode, which pads automatically unless a budget
            was given.
        compute_atomic_stresses: Also report per-atom ``stresses`` and
            ``virials``.

    Raises:
        ValueError: If no model is named, a pattern matches nothing, the
            committee's cutoffs differ, or the head is not one of the model's.
    """

    def __init__(
        self,
        model_paths: str | Path | Sequence[str | Path] | None = None,
        models: DeployedModel | Sequence[DeployedModel] | None = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        charges_key: str = "Qs",
        info_keys: Mapping[str, str] | None = None,
        arrays_keys: Mapping[str, str] | None = None,
        head: str | None = None,
        pad_num_atoms: int = 0,
        pad_num_edges: int = 0,
        padding: PaddingPolicy | None = None,
        compile_mode: str | None = None,
        compute_atomic_stresses: bool = False,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.models = _committee(model_paths, models, device)
        cutoffs = [model.r_max for model in self.models]
        if len(set(cutoffs)) != 1:
            raise ValueError(
                f"the committee's cutoffs are {cutoffs}, and one structure's "
                f"graph has one cutoff. Every member has to share it."
            )
        first = self.models[0]
        for model in self.models:
            if model.config.model.observables and (
                ENERGY_OBSERVABLE not in model.config.model.observables
            ):
                raise ValueError(
                    f"{model.path} does not declare {ENERGY_OBSERVABLE!r}. This "
                    f"calculator is the energy family's; the others have their "
                    f"own."
                )
        self.r_max = cutoffs[0]
        self.z_table = first.z_table
        self.heads = first.heads
        self.head = _choose_head(self.heads, head)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.info_keys = dict(DEFAULT_INFO_KEYS if info_keys is None else info_keys)
        self.arrays_keys = {**dict(arrays_keys or {}), "charges": charges_key}
        self.compute_atomic_stresses = compute_atomic_stresses
        self.padding = padding or PaddingPolicy.requested(
            pad_num_atoms, pad_num_edges, compiled=compile_mode is not None
        )
        self._rows = output_rows(first.outputs)
        self._engines = [model.engine for model in self.models]
        if compile_mode is not None:
            self._engines = [
                torch.compile(engine, mode=compile_mode, dynamic=False)
                for engine in self._engines
            ]
        for model in self.models:
            for parameter in model.engine.parameters():
                parameter.requires_grad_(False)

        if len(self.models) > 1:
            logger.info("Running a committee of %d models", len(self.models))
        # The instance's own list: the class attribute on the ASE base is
        # shared, and extending it grows every calculator built after.
        self.implemented_properties = declared_properties(
            committee=len(self.models) > 1,
            atomic_stresses=compute_atomic_stresses,
        )

    # -----------------------------------------------------------------------
    # The ASE interface
    # -----------------------------------------------------------------------

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list[str]:
        """What changed since the last calculation, the passed-through
        ``atoms.info`` entries included.

        Those entries are inputs to the model, so a change in one has to
        invalidate the cached results. Array values are left to the arrays
        comparison ASE already makes.
        """
        state = super().check_state(atoms, tol=tol)
        if (
            not state
            and self.atoms is not None
            and not _infos_equal(self.atoms.info, atoms.info)
        ):
            state.append("info")
        return state

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms)
        assert self.atoms is not None
        compute = ["forces", "stress"]
        if self.compute_atomic_stresses:
            compute += ["atomic_stresses", "atomic_virials"]
        graph, info = self._graph(self.atoms, padded=True)
        per_model = [
            unpad_outputs(
                engine(dict(graph), compute=tuple(compute), training=False),
                info,
                self._rows,
            )
            for engine in self._engines
        ]
        self.results = self._results(per_model)

    # -----------------------------------------------------------------------
    # Beyond the ASE interface
    # -----------------------------------------------------------------------

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray | list[np.ndarray]:
        """``d2E / dr dr``, ``[3 * n_atoms, n_atoms, 3]``, per model.

        Returns:
            The array for one model, a list of them for a committee.
        """
        atoms = self._atoms(atoms)
        graph, _ = self._graph(atoms, padded=False)
        scale = self.energy_units_to_eV / self.length_units_to_A**2
        hessians = [
            engine(dict(graph), compute=("hessian",), training=False)
            .extras["hessian"]
            .detach()
            .cpu()
            .numpy()
            * scale
            for engine in self._engines
        ]
        return hessians[0] if len(hessians) == 1 else hessians

    def get_descriptors(
        self,
        atoms: Atoms | None = None,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> np.ndarray | list[np.ndarray]:
        """Each atom's node features, the layers side by side.

        Args:
            atoms: The structure, or the last one calculated.
            invariants_only: Keep only the scalar channels of each layer.
            num_layers: How many layers, from the first. ``-1`` is all.

        Returns:
            ``[n_atoms, features]`` for one model, a list for a committee.
        """
        atoms = self._atoms(atoms)
        graph, _ = self._graph(atoms, padded=False)
        descriptors = []
        with torch.no_grad():
            for model in self.models:
                backbone = model.model.get_submodule("backbone")
                assert isinstance(backbone, MACEBackbone)
                value = backbone.descriptors(
                    graph,
                    num_layers=None if num_layers == -1 else num_layers,
                    invariants_only=invariants_only,
                )
                descriptors.append(value.detach().cpu().numpy())
        return descriptors[0] if len(descriptors) == 1 else descriptors

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _atoms(self, atoms: Atoms | None) -> Atoms:
        if atoms is not None:
            return atoms
        if self.atoms is None:
            raise ValueError("no structure given and none calculated before.")
        return self.atoms

    def _graph(
        self, atoms: Atoms, *, padded: bool
    ) -> tuple[dict[str, Any], PaddingInfo]:
        """One structure's batch, padded to the budget when asked, on device."""
        periodic = atoms.get_pbc()
        configuration = Configuration(
            atomic_numbers=np.asarray(atoms.get_atomic_numbers()),
            positions=np.asarray(atoms.get_positions(), dtype=np.float64),
            cell=np.asarray(atoms.get_cell().array, dtype=np.float64),
            pbc=(bool(periodic[0]), bool(periodic[1]), bool(periodic[2])),
        )
        structure = graph_from_configuration(
            configuration,
            cutoff=self.r_max,
            z_table=self.z_table,
            head=self.heads.index(self.head),
        )
        if padded:
            nodes = len(atoms)
            edges = int(structure["edge_index"].shape[1])
            policy, _ = resolve_budget(self.padding, nodes, edges)
            before = (self.padding.nodes_budget, self.padding.edges_budget)
            after = (policy.nodes_budget, policy.edges_budget)
            if any(
                then and now != then for then, now in zip(before, after, strict=True)
            ):
                logger.warning(
                    "The structure has %d atoms and %d edges, over the padding "
                    "budget of %d and %d; the budget is now %d and %d, which "
                    "changes the batch shape once",
                    nodes,
                    edges,
                    self.padding.nodes_budget,
                    self.padding.edges_budget,
                    policy.nodes_budget,
                    policy.edges_budget,
                )
            self.padding = policy
            structures, info = pad_batch(structure, policy, self.r_max)
        else:
            structures = [structure]
            info = PaddingInfo(
                nodes=len(atoms), edges=int(structure["edge_index"].shape[1])
            )
        dtype = next(self.models[0].engine.parameters()).dtype
        batch = collate_training(
            [(item, {}, {}) for item in structures],
            z_table=self.z_table,
            float_dtype="float64" if dtype == torch.float64 else "float32",
        )
        graph: dict[str, Any] = {
            name: value.to(self.device) if isinstance(value, Tensor) else value
            for name, value in batch.graph.items()
        }
        graph.update(self._passed_through(atoms, graph, info, dtype))
        return graph, info

    def _passed_through(
        self,
        atoms: Atoms,
        graph: Mapping[str, Any],
        info: PaddingInfo,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        """The ``atoms.info`` and ``atoms.arrays`` entries the model reads,
        with zero rows for the padding."""
        nodes = int(graph["positions"].shape[0])
        graphs = int(graph["num_graphs"])
        values: dict[str, Tensor] = {}
        for field, key in self.info_keys.items():
            if key in atoms.info:
                row = torch.as_tensor(np.asarray(atoms.info[key]), dtype=dtype)
                padded = torch.zeros((graphs, *row.shape), dtype=dtype)
                padded[0] = row
                values[field] = padded.to(self.device)
        for field, key in self.arrays_keys.items():
            if key in atoms.arrays:
                rows = torch.as_tensor(np.asarray(atoms.arrays[key]), dtype=dtype)
                padded = torch.zeros((nodes, *rows.shape[1:]), dtype=dtype)
                padded[: info.nodes] = rows
                values[field] = padded.to(self.device)
        return values

    def _results(self, per_model: Sequence[MACEOutput[Tensor]]) -> dict[str, Any]:
        energy = self.energy_units_to_eV
        length = self.length_units_to_A
        quantities: dict[str, tuple[str, float]] = {
            "energy": ("total_energy", energy),
            "energies": ("node_energies", energy),
            "node_energy": ("node_interaction_energy", energy),
            "forces": ("forces", energy / length),
            "stress": ("stress", energy / length**3),
        }
        if self.compute_atomic_stresses:
            quantities["stresses"] = ("atomic_stresses", energy / length**3)
            quantities["virials"] = ("atomic_virials", energy)
        results: dict[str, Any] = {}
        for key, (name, factor) in quantities.items():
            values = [output.get(name) for output in per_model]
            if any(value is None for value in values):
                continue
            stacked = torch.stack([v.detach() for v in values if v is not None])
            if key in ("energy", "stress"):
                stacked = stacked[:, 0]
            stack = stacked.to(torch.float64).cpu().numpy()
            results[key] = stack.mean(axis=0) * factor
            if len(per_model) > 1 and key in ENSEMBLE_KEYS:
                results[f"{key}_comm"] = stack * factor
                results[f"{key}_var"] = stack.var(axis=0) * factor**2
        results["energy"] = float(results["energy"])
        if len(per_model) > 1:
            results["energy_var"] = float(results["energy_var"])
        results["free_energy"] = results["energy"]
        for key in ("stress", "stress_comm", "stress_var", "stresses"):
            if key in results:
                results[key] = full_3x3_to_voigt_6_stress(results[key])
        return results


def _committee(
    model_paths: str | Path | Sequence[str | Path] | None,
    models: DeployedModel | Sequence[DeployedModel] | None,
    device: str,
) -> list[DeployedModel]:
    if models is not None:
        members = [models] if isinstance(models, DeployedModel) else list(models)
        if not members:
            raise ValueError("an empty list of models was given.")
        return members
    if model_paths is None:
        raise ValueError("give model_paths or models.")
    if isinstance(model_paths, str):
        paths: list[str | Path] = _models_matching(model_paths)
        if not paths:
            raise ValueError(f"no model file matches {model_paths!r}.")
    elif isinstance(model_paths, Path):
        paths = [model_paths]
    else:
        paths = list(model_paths)
    if not paths:
        raise ValueError("an empty list of model paths was given.")
    return [load_deployed(path, device=device) for path in paths]


def _models_matching(pattern: str) -> list[str | Path]:
    """The model checkpoints a pattern names, one per model.

    A pattern can match both files of a checkpoint, and the records of the
    run checkpoints a training run leaves beside its model. Each model is
    taken once, by its record, and a record of another format is not a model.
    """
    records: set[Path] = set()
    for match in glob.glob(pattern):
        record = Path(match).with_suffix(".json")
        if not record.is_file():
            continue
        try:
            written = json.loads(record.read_text()).get("format")
        except (OSError, ValueError):
            continue
        if written == CHECKPOINT_FORMAT:
            records.add(record)
    return sorted(records)


def _choose_head(heads: Sequence[str], head: str | None) -> str:
    """The head a structure is evaluated with.

    Raises:
        ValueError: For a head the model does not have, naming the ones it
            does. The frozen tree warns and takes the last one, which
            evaluates a level of theory nobody asked for.
    """
    if head is not None:
        if head not in heads:
            raise ValueError(
                f"the model has no head {head!r}; its heads are {list(heads)}."
            )
        return head
    if len(heads) == 1:
        return heads[0]
    defaults = [name for name in heads if name.lower() == "default"]
    if not defaults:
        raise ValueError(
            f"the model has the heads {list(heads)} and none is called "
            f"'default', so which to evaluate has to be said: pass head=."
        )
    return defaults[0]


def _infos_equal(saved: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if saved.keys() != current.keys():
        return False
    for key, value in saved.items():
        other = current[key]
        if isinstance(value, np.ndarray) or isinstance(other, np.ndarray):
            continue
        if value != other:
            return False
    return True
