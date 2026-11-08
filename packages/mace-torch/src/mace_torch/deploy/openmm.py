"""OpenMM's direct PyTorch callback over the shared deployment loader.

The callback follows OpenMM-Torch's PythonTorchForce interface:
https://github.com/openmm/openmm-torch/blob/b6532900ad825cb3d12193817a1e34a7f46ae6a3/README.md
OpenMM supplies nm and expects kJ/mol and kJ/mol/nm. MACE uses Angstrom and eV.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import units
from mace_core.data.configuration import Configuration
from mace_core.neighbors import NeighborListConfig

from mace_torch.backends.precision import interface_dtype
from mace_torch.data.inputs import graph_inputs_of
from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.graph import build_batch
from mace_torch.neighbors import resolve_neighbor_provider

EV_TO_KJ_MOL = units.mol / units.kJ
NM_TO_ANGSTROM = 10.0


class OpenMMEvaluator:
    """Evaluate a fixed species sequence, with neighbours rebuilt each call.

    ``inputs`` contains declared model inputs in MACE units, such as charge or
    an external field. :meth:`evaluate` can supply updated values each step.
    ``periodic=True`` reads the current triclinic box from OpenMM's State.
    The neighbour provider and model backend are resolved once at construction.
    """

    def __init__(
        self,
        bundle: str | Path,
        atomic_numbers: Sequence[int],
        *,
        device: str = "cpu",
        head: str | None = None,
        backend: str | None = None,
        periodic: bool = False,
        inputs: Mapping[str, Any] | None = None,
        neighbor_list: NeighborListConfig | None = None,
    ) -> None:
        self.deployed = load_deployed(bundle, device=device, backend=backend)
        if "energy" not in {spec.name for spec in self.deployed.outputs.observables}:
            raise DeployError("OpenMM requires a model with an energy observable")
        numbers = np.asarray(atomic_numbers)
        if numbers.ndim != 1 or not numbers.size or numbers.dtype.kind not in "iu":
            raise ValueError("atomic_numbers must be a nonempty integer sequence")
        if not np.isin(numbers, self.deployed.z_table.zs).all():
            raise ValueError("atomic_numbers contains species absent from the model")
        self.atomic_numbers = numbers.copy()
        self.atomic_numbers.setflags(write=False)
        self.device = torch.device(device)
        self.dtype = interface_dtype(self.deployed.engine)
        self.head = self.deployed.select_head(head)
        self.periodic = bool(periodic)
        self.inputs = dict(inputs or {})
        self.input_names = set(graph_inputs_of(self.deployed.config.model))
        self.neighbors = resolve_neighbor_provider(neighbor_list, device=self.device)
        self._check_inputs(self.inputs)

    def _check_inputs(self, values: Mapping[str, Any]) -> None:
        unknown = values.keys() - self.input_names
        if unknown:
            raise ValueError(f"inputs not declared by this model: {sorted(unknown)}")

    @torch.inference_mode(False)
    def evaluate(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | np.ndarray | None = None,
        *,
        inputs: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions/cell in nm; energy and forces in kJ/mol and kJ/mol/nm."""
        if positions.shape != (len(self.atomic_numbers), 3):
            raise ValueError("positions must have one three-vector per declared atom")
        if self.periodic and cell is None:
            raise ValueError("periodic evaluation requires the current cell in nm")
        properties = {**self.inputs, **(inputs or {})}
        self._check_inputs(properties)

        def host(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else value

        configuration = Configuration(
            atomic_numbers=self.atomic_numbers,
            positions=host(positions.to(torch.float64)) * NM_TO_ANGSTROM,
            cell=(
                np.zeros((3, 3))
                if cell is None
                else np.asarray(host(cell), dtype=np.float64) * NM_TO_ANGSTROM
            ),
            pbc=(self.periodic, self.periodic, self.periodic),
            properties={key: host(value) for key, value in properties.items()},
            head=self.head,
        )
        graph = build_batch(
            [configuration],
            atomic_number_table=self.deployed.z_table,
            cutoff=self.deployed.r_max,
            neighbor_backend=self.neighbors,
            heads=self.deployed.heads,
            features=self.deployed.config.model.feature_storages,
            dtype=self.dtype,
            device=self.device,
        )
        # External engines may call with grad mode disabled. MACE's forces
        # still require the energy graph, whose lifetime ends at this callback.
        with torch.enable_grad():
            result = self.deployed.compute(graph, compute=("forces",))
        if result.total_energy is None or result.forces is None:
            raise DeployError("the model did not produce energy and forces")
        return (
            result.total_energy[0].detach() * EV_TO_KJ_MOL,
            result.forces.detach() * (EV_TO_KJ_MOL * NM_TO_ANGSTROM),
        )

    def __call__(self, state, positions):
        cell = None
        if self.periodic:
            from openmm.unit import nanometer  # ty: ignore[unresolved-import]

            cell = state.getPeriodicBoxVectors().value_in_unit(nanometer)
        return self.evaluate(positions, cell)

    def as_force(self):
        """Construct a PythonTorchForce, importing the optional plugin only here."""
        try:
            from openmmtorch import PythonTorchForce  # ty: ignore[unresolved-import]
        except ImportError as error:
            raise ImportError(
                "OpenMM requires openmm-torch with PythonTorchForce support; "
                "install the revision recorded in the integration support document"
            ) from error
        force = PythonTorchForce(self)
        force.setUsesPeriodicBoundaryConditions(self.periodic)
        return force

    def __reduce_ex__(self, protocol):
        """Keep OpenMM's optional callback serialization out of model storage."""
        raise TypeError("rebuild OpenMMEvaluator from its V1 bundle; do not pickle it")
