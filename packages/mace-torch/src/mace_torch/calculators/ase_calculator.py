"""The ASE calculator for every v1 model: energy, charge-aware and response.

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

**The charge-aware model adds its own results**, under the frozen tree's
names: ``dipole``, ``charges`` and ``spins`` per atom, the three energy parts
``interaction_energy``, ``electrostatic_energy`` and ``electron_energy``, the
density as ``density_coefficients`` and per spin as ``spin_charge_density``,
and ``fukui_functions``. A committee reports ``dipole`` per model and as a
spread too. Its charge, multiplicity and applied field are read from
``atoms.info`` (``charge``, ``spin``, ``external_field`` by default), and a
structure that gives none is neutral, a singlet and in no field.

**Units.** The model computes in eV and Angstrom. ``energy_units_to_eV`` and
``length_units_to_A`` convert each result by its dimension: energies by E,
forces by E/L, stresses by E/L^3, virials by E, the Hessian by E/L^2 and the
dipole by L. A variance takes the square of its quantity's factor. Charges,
spins and the density coefficients are left as the model computes them, in
units of the elementary charge with the dipole components in e Angstrom.

**A response model adds its own results**: ``dipole``, and for the
dielectric model ``charges``, ``polarizability`` as a matrix and
``polarizability_sh`` as its six spherical components. It has no energy, so it
writes no energy, forces or stress. A committee reports ``dipole`` per model and
as a spread, whichever model it comes from.

**The magnetic model adds** ``magforces``, ``-dE/dm`` per atom, and reads its
moments from ``atoms.arrays`` under ``magmom_key`` (``REF_magmom`` by default,
the frozen tree's key). A structure that carries none is refused rather than
evaluated at zero moments, since ase's initial magnetic moments live elsewhere
and are not what the model reads. A change to the moments invalidates the
cached results like a change to the positions. With ``fixed_point`` the moments
are relaxed first, and the relaxed ones are reported as ``MACE_magmoms`` and
written back into ``atoms.arrays`` under that name, as the frozen tree does.

**What is written is read off the model, not off its kind.** There is no
``model_type``. Every result is one row of :data:`RESULTS`, naming the model
output it is read from, and a calculator writes the rows whose output its
models produce: the observables they declare and what each model says it adds.
A capability the model does not declare is refused by the observable's name:
the Hessian needs ``energy``, and :meth:`MACECalculator.get_dielectric_derivatives`
needs ``dipole`` and its declared derivative ``dmu_dr``.
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
from mace_core.config import FixedPointSpec
from mace_core.data.configuration import Configuration
from mace_core.outputs import FIELD_BY_OBSERVABLE, MACEOutput
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
from mace_torch.physics import DerivativeEngine
from mace_torch.physics.fixed_point import FixedPointDriver
from mace_torch.serialization import FORMAT as CHECKPOINT_FORMAT
from mace_torch.train.data_stage import graph_inputs_of

__all__ = [
    "ENSEMBLE_KEYS",
    "POLAR_RESULTS",
    "RESULTS",
    "MACECalculator",
    "declared_properties",
    "produced_outputs",
]

logger = logging.getLogger(__name__)

#: The results a committee also reports per model and as a spread, when the
#: model produces them.
ENSEMBLE_KEYS = ("energy", "forces", "stress", "dipole")

#: What ``atoms.info`` entries reach the graph by default, as the frozen tree
#: passes them: graph field, then info key.
DEFAULT_INFO_KEYS = {
    "total_spin": "spin",
    "total_charge": "charge",
    "external_field": "external_field",
}


#: The charge-aware model's results, with the model output each is read from
#: and its dimension: energies convert by E, the dipole by L, the rest not.
POLAR_RESULTS: dict[str, tuple[str, str]] = {
    "dipole": ("dipole", "length"),
    "charges": ("charges", "none"),
    "spins": ("spins", "none"),
    "interaction_energy": ("interaction_energy", "energy"),
    "electrostatic_energy": ("electrostatic_energy", "energy"),
    "electron_energy": ("electron_energy", "energy"),
    "density_coefficients": ("density_coefficients", "none"),
    "spin_charge_density": ("spin_charge_density", "none"),
    "fukui_functions": ("fukui_functions", "none"),
}


#: Every result the calculator can write, in the order it lists them: the model
#: output each is read from, and its dimension. Energies convert by E, forces
#: by E/L, stresses by E/L^3, the dipole by L, and the rest are left as the
#: model computes them.
RESULTS: dict[str, tuple[str, str]] = {
    "energy": ("total_energy", "energy"),
    "energies": ("node_energies", "energy"),
    "node_energy": ("node_interaction_energy", "energy"),
    "forces": ("forces", "force"),
    "stress": ("stress", "stress"),
    "stresses": ("atomic_stresses", "stress"),
    "virials": ("atomic_virials", "energy"),
    **POLAR_RESULTS,
    "polarizability": ("polarizability", "none"),
    "polarizability_sh": ("polarizability_sh", "none"),
    # An energy per moment: the moments are not converted, so the energy is.
    "magforces": ("magforces", "energy"),
    "MACE_magmoms": ("converged_magmom", "none"),
}

#: The derivative a model that reads moments is always asked for, beside the
#: forces and the stress.
MAGNETIC_DERIVATIVE = "magforces"

#: Where the moments are read from, and where the relaxed ones are written.
MAGMOM_KEY = "REF_magmom"
RELAXED_MAGMOM_KEY = "MACE_magmoms"


def produced_outputs(
    observables: Sequence[str],
    extra_rows: Mapping[str, str],
    *,
    atomic_stresses: bool = False,
    produced: Sequence[str] = (),
    derivatives: Sequence[str] = (),
) -> set[str]:
    """The model outputs a calculator reads from a model so declared.

    Args:
        observables: The declared observables, by name.
        extra_rows: What the model says it adds to its output.
        atomic_stresses: Whether per-atom stresses are asked for.
        produced: What the model computes itself rather than reads out.
        derivatives: The energy derivatives asked of it beyond the forces and
            the stress, ``magforces`` for a model that reads moments.
    """
    names = {FIELD_BY_OBSERVABLE.get(name, name) for name in observables}
    names |= set(extra_rows) | set(produced) | set(derivatives)
    if ENERGY_OBSERVABLE in observables:
        # Forces and stress are always taken for an energy model, as the frozen
        # tree takes them, whatever derivatives it was trained against.
        names |= {"total_energy", "node_energies", "forces", "stress"}
        if atomic_stresses:
            names |= {"atomic_stresses", "atomic_virials"}
    return names


def declared_properties(outputs: set[str], *, committee: bool) -> list[str]:
    """Every result key a calculator reading these outputs writes, and no other."""
    properties = [key for key, (name, _) in RESULTS.items() if name in outputs]
    if "energy" in properties:
        properties.insert(properties.index("energy") + 1, "free_energy")
    if committee:
        properties += [
            f"{key}{suffix}"
            for key in ENSEMBLE_KEYS
            if key in properties
            for suffix in ("_comm", "_var")
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
        external_field: An applied field ``[Ex, Ey, Ez]`` in V/Angstrom for
            every structure, in place of each one's ``atoms.info`` entry. Read
            by a charge-aware model only.
        magmom_key: The ``atoms.arrays`` entry the moments are read from, in
            muB. Read by a magnetic model only.
        fixed_point: Relax the moments to where the energy's derivative
            against them vanishes before reporting anything. For a magnetic
            model, and one model at a time.

    Raises:
        ValueError: If no model is named, a pattern matches nothing, the
            committee's cutoffs differ, the head is not one of the model's, or
            a fixed point is asked of a committee or of a model that does not
            read what it relaxes.
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
        external_field: Sequence[float] | None = None,
        magmom_key: str = MAGMOM_KEY,
        fixed_point: FixedPointSpec | None = None,
        **kwargs: Any,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        if external_field is not None and np.asarray(external_field).size != 3:
            raise ValueError(
                f"external_field is {list(np.asarray(external_field).ravel())}; "
                f"it is one vector, [Ex, Ey, Ez]."
            )
        self.external_field = (
            None
            if external_field is None
            else np.asarray(external_field, dtype=float).reshape(3)
        )
        self.device = device
        self.models = _committee(model_paths, models, device)
        cutoffs = [model.r_max for model in self.models]
        if len(set(cutoffs)) != 1:
            raise ValueError(
                f"the committee's cutoffs are {cutoffs}, and one structure's "
                f"graph has one cutoff. Every member has to share it."
            )
        first = self.models[0]
        declared = {tuple(_declared(model)) for model in self.models}
        if len(declared) != 1:
            raise ValueError(
                f"the committee's members declare different observables, "
                f"{sorted(declared)}, and one calculator reports one set of "
                f"results for all of them."
            )
        self.observables = declared.pop()
        inputs = {graph_inputs_of(model.config.model.model) for model in self.models}
        if len(inputs) != 1:
            raise ValueError(
                "the committee mixes models that read different inputs, and one "
                "structure's graph carries one set."
            )
        self.graph_inputs = inputs.pop()
        self.has_energy = ENERGY_OBSERVABLE in self.observables
        self.magmom_key = magmom_key
        self.reads_moments = "magmom" in self.graph_inputs
        self.fixed_point = fixed_point
        if fixed_point is not None:
            if fixed_point.variable not in self.graph_inputs:
                raise ValueError(
                    f"the fixed point relaxes {fixed_point.variable!r}, and the "
                    f"model reads {list(self.graph_inputs)}, so the energy does "
                    f"not depend on it and every structure is already relaxed."
                )
            if len(self.models) > 1:
                raise ValueError(
                    f"a fixed point was asked of a committee of "
                    f"{len(self.models)} models. Each would relax the moments "
                    f"to its own, and there is no one relaxed state to report."
                )
            if compile_mode is not None:
                raise ValueError(
                    "a fixed point runs a variable number of evaluations at "
                    "unpadded shapes, which a compiled static-shape model "
                    "cannot. Leave compile_mode unset."
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
        extra_rows = _extra_rows(first)
        self._rows = output_rows(first.outputs, extra_rows)
        self._outputs = produced_outputs(
            self.observables,
            extra_rows,
            atomic_stresses=compute_atomic_stresses,
            produced=[
                *sorted(getattr(first.model, "PRODUCED", ())),
                *([f"converged_{fixed_point.variable}"] if fixed_point else []),
            ],
            derivatives=[MAGNETIC_DERIVATIVE] if self.reads_moments else [],
        )
        self._engines = [model.engine for model in self.models]
        if compile_mode is not None:
            self._engines = [
                torch.compile(engine, mode=compile_mode, dynamic=False)
                for engine in self._engines
            ]
        for model in self.models:
            for parameter in model.engine.parameters():
                parameter.requires_grad_(False)

        if fixed_point is not None:
            # Around the engines themselves: a fixed point is never compiled.
            self._engines = [
                FixedPointDriver(_derivative_engine(model), fixed_point)
                for model in self.models
            ]
        if len(self.models) > 1:
            logger.info("Running a committee of %d models", len(self.models))
        # The instance's own list: the class attribute on the ASE base is
        # shared, and extending it grows every calculator built after.
        self.implemented_properties = declared_properties(
            self._outputs, committee=len(self.models) > 1
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
        # The moments are an input as much as the positions are, and ase's own
        # comparison does not look at an entry it does not know.
        if (
            not state
            and self.reads_moments
            and self.atoms is not None
            and not _arrays_equal(
                self.atoms.arrays.get(self.magmom_key),
                atoms.arrays.get(self.magmom_key),
                tol,
            )
        ):
            state.append(self.magmom_key)
        return state

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] | None = None,
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms)
        assert self.atoms is not None
        compute = ["forces", "stress"] if self.has_energy else []
        if self.compute_atomic_stresses and self.has_energy:
            compute += ["atomic_stresses", "atomic_virials"]
        if self.reads_moments:
            compute.append(MAGNETIC_DERIVATIVE)
        if self.fixed_point is not None:
            # Unpadded: padding atoms would be relaxed with the real ones and
            # move the solver's path.
            graph, _ = self._graph(self.atoms, padded=False)
            per_model = [
                engine(dict(graph), compute=tuple(compute), training=False)
                for engine in self._engines
            ]
        else:
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
        if RELAXED_MAGMOM_KEY in self.results:
            # Onto the structure the caller holds, not only the copy ase keeps.
            for target in {
                id(item): item for item in (atoms, self.atoms) if item is not None
            }.values():
                target.arrays[RELAXED_MAGMOM_KEY] = self.results[
                    RELAXED_MAGMOM_KEY
                ].copy()

    # -----------------------------------------------------------------------
    # Beyond the ASE interface
    # -----------------------------------------------------------------------

    def get_hessian(self, atoms: Atoms | None = None) -> np.ndarray | list[np.ndarray]:
        """``d2E / dr dr``, ``[3 * n_atoms, n_atoms, 3]``, per model.

        Returns:
            The array for one model, a list of them for a committee.

        Raises:
            NotImplementedError: If the model declares no ``energy``, which is
                what the Hessian is the second derivative of.
        """
        if not self.has_energy:
            raise NotImplementedError(
                f"the Hessian is the second derivative of the {ENERGY_OBSERVABLE!r} "
                f"observable, and the model declares {list(self.observables)}."
            )
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

    def get_dielectric_derivatives(self, atoms: Atoms | None = None):
        """The position derivatives of the dipole, and of the polarizability.

        ``dmu_dr`` is ``[3, n_atoms, 3]``, ``d mu_i / d r_aj`` at ``[i, a, j]``,
        and ``dalpha_dr`` is ``[9, n_atoms, 3]`` with the polarizability's nine
        components row major. Both are left in the model's units.

        Returns:
            For one model, ``dmu_dr``, or ``(dmu_dr, dalpha_dr)`` when the model
            declares a polarizability; for a committee the same with a list per
            quantity, one entry per model. That is the frozen tree's shape.

        Raises:
            NotImplementedError: If the model declares no ``dipole`` with its
                position derivative ``dmu_dr``, naming what it does declare.
        """
        wanted = [
            name
            for name in ("dmu_dr", "dalpha_dr")
            if all(name in _responses(model) for model in self.models)
        ]
        if "dmu_dr" not in wanted:
            raise NotImplementedError(
                f"dielectric derivatives are the position derivatives of the "
                f"'dipole' observable, declared as 'dmu_dr', and the model "
                f"declares {list(self.observables)}."
            )
        atoms = self._atoms(atoms)
        graph, _ = self._graph(atoms, padded=False)
        per_model = [
            engine(dict(graph), compute=tuple(wanted), training=False)
            for engine in self._engines
        ]
        values = [
            [output.extras[name].detach().cpu().numpy() for output in per_model]
            for name in wanted
        ]
        if len(self.models) == 1:
            values = [value[0] for value in values]
        return values[0] if len(values) == 1 else tuple(values)

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
        properties = {
            name: atoms.info[self.info_keys[name]]
            for name in self.graph_inputs
            if name in self.info_keys and self.info_keys[name] in atoms.info
        }
        if self.external_field is not None and "external_field" in self.graph_inputs:
            properties["external_field"] = self.external_field
        if self.reads_moments:
            if self.magmom_key not in atoms.arrays:
                raise ValueError(
                    f"the model reads a moment on every atom from "
                    f"atoms.arrays[{self.magmom_key!r}], and the structure has "
                    f"no such entry; its arrays are {sorted(atoms.arrays)}. "
                    f"ase's initial magnetic moments are not read: set the "
                    f"moments under that key, in muB, or pass magmom_key."
                )
            properties["magmom"] = np.asarray(
                atoms.arrays[self.magmom_key], dtype=np.float64
            )
        configuration = Configuration(
            atomic_numbers=np.asarray(atoms.get_atomic_numbers()),
            positions=np.asarray(atoms.get_positions(), dtype=np.float64),
            cell=np.asarray(atoms.get_cell().array, dtype=np.float64),
            pbc=(bool(periodic[0]), bool(periodic[1]), bool(periodic[2])),
            properties=properties,
        )
        structure = graph_from_configuration(
            configuration,
            cutoff=self.r_max,
            z_table=self.z_table,
            head=self.heads.index(self.head),
            graph_inputs=self.graph_inputs,
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
            # A model's own per-structure inputs are in the graph already, from
            # the builder, with their defaults and their padding.
            if field in self.graph_inputs:
                continue
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
        factors = {
            "energy": self.energy_units_to_eV,
            "force": self.energy_units_to_eV / self.length_units_to_A,
            "stress": self.energy_units_to_eV / self.length_units_to_A**3,
            "length": self.length_units_to_A,
            "none": 1.0,
        }
        results: dict[str, Any] = {}
        for key, (name, dimension) in RESULTS.items():
            if name not in self._outputs:
                continue
            values = [output.get(name) for output in per_model]
            if any(value is None for value in values):
                continue
            factor = factors[dimension]
            stacked = torch.stack([v.detach() for v in values if v is not None])
            if self._rows.get(name) == "graph":
                # One structure: its row, not a batch of one.
                stacked = stacked[:, 0]
            stack = stacked.to(torch.float64).cpu().numpy()
            results[key] = stack.mean(axis=0) * factor
            if len(per_model) > 1 and key in ENSEMBLE_KEYS:
                results[f"{key}_comm"] = stack * factor
                results[f"{key}_var"] = stack.var(axis=0) * factor**2
        for key in ("energy", "energy_var", *_PER_STRUCTURE):
            if key in results:
                results[key] = float(results[key])
        if "energy" in results:
            results["free_energy"] = results["energy"]
        for key in ("stress", "stress_comm", "stress_var", "stresses"):
            if key in results:
                results[key] = full_3x3_to_voigt_6_stress(results[key])
        return results


#: The charge-aware results with one value per structure, reported as numbers.
_PER_STRUCTURE = ("interaction_energy", "electrostatic_energy", "electron_energy")


def _declared(model: DeployedModel) -> list[str]:
    """The observables a model was built to read out, by name."""
    return [spec.name for spec in model.outputs.observables]


def _extra_rows(model: DeployedModel) -> dict[str, str]:
    """What the model says it adds to its output, and the row of each."""
    return dict(getattr(model.model, "extra_rows", {}))


def _derivative_engine(model: DeployedModel) -> DerivativeEngine:
    """The model's derivative engine, which a fixed point relaxes around."""
    engine = model.engine
    assert isinstance(engine, DerivativeEngine)
    return engine


def _responses(model: DeployedModel) -> set[str]:
    """The response derivatives the model's engine takes, by name."""
    return set(getattr(model.engine, "responses", {}))


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


def _arrays_equal(saved: Any, current: Any, tol: float) -> bool:
    if saved is None or current is None:
        return saved is None and current is None
    saved, current = np.asarray(saved), np.asarray(current)
    return saved.shape == current.shape and bool(
        np.allclose(saved, current, atol=tol, rtol=0.0)
    )


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
