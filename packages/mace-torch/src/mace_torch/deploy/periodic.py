"""Native Polar evaluation of a complete periodic system at fixed cell.

The discrete reciprocal-space selection is frozen, not the cell derivatives.
Positions and the cell remain inputs so the exported virial includes both
real- and reciprocal-space terms. Runtimes check the cell and atom count.
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from mace_core.data.configuration import Configuration
from torch import Tensor, nn

from mace_torch.deploy.loader import DeployedModel, DeployError
from mace_torch.electrostatics.solver import LONG_RANGE_PLAN
from mace_torch.graph import build_batch
from mace_torch.neighbors import resolve_neighbor_provider


class PeriodicEnergy(nn.Module):
    """Total energy, local site energies, full forces and full virial.

    Polar's site energies describe its short-range readout; they do not sum
    to its total energy. No arbitrary per-atom electrostatic partition is added.
    Charge, spin and field are fixed by the export structure.
    """

    attributes: Tensor
    numbers: Tensor
    total_charge: Tensor
    total_spin: Tensor
    external_field: Tensor

    def __init__(self, deployed, graph):
        super().__init__()
        self.engine = deployed.derivative_engine
        self.head = deployed.heads.index(deployed.select_head())
        self.model.prepare_model_inputs(graph)
        self.plan = graph[LONG_RANGE_PLAN]
        self.register_buffer("numbers", torch.tensor(deployed.z_table.zs))
        self.register_buffer(
            "attributes", torch.eye(len(deployed.z_table.zs), dtype=deployed.dtype)
        )
        for name in ("total_charge", "total_spin", "external_field"):
            self.register_buffer(name, graph[name])

    @property
    def model(self):
        return self.engine.model

    def forward(self, unit_shifts, edge_index, elements, positions, cell):
        graph = {
            "positions": positions,
            "shifts": unit_shifts @ cell,
            "unit_shifts": unit_shifts,
            "edge_index": edge_index,
            "node_attrs": self.attributes.index_select(0, elements),
            "node_atomic_numbers": self.numbers.index_select(0, elements),
            "batch": torch.zeros_like(elements),
            "ptr": elements.new_tensor([0, len(elements)]),
            "head": elements.new_tensor([self.head]),
            "cell": cell.unsqueeze(0),
            "pbc": torch.ones((1, 3), dtype=torch.bool, device=cell.device),
            "total_charge": self.total_charge,
            "total_spin": self.total_spin,
            "external_field": self.external_field,
            LONG_RANGE_PLAN: self.plan,
        }
        output = self.engine(graph, compute=("forces", "virials"))
        return tuple(
            x.detach()
            for x in (
                output.total_energy.sum(),
                output.node_energies,
                output.forces,
                output.virials[0],
            )
        )


def periodic_export_inputs(
    deployed: DeployedModel, atoms: Atoms, *, nodes: int, edges: int, device: str
) -> tuple[PeriodicEnergy, tuple[tuple[Tensor, ...], ...]]:
    """Construct representative inputs without introducing fictitious charges."""
    if nodes != len(atoms):
        raise DeployError(
            "Polar native node slots must equal the structure's atom count"
        )
    if not all(atoms.pbc) or np.linalg.det(atoms.cell) <= 0:
        raise DeployError(
            "Polar native export requires a right-handed, fully periodic cell"
        )
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise DeployError("Polar export geometry must be finite")
    if deployed.config.electrostatics.solver != "reference":
        raise DeployError(
            "native Polar export currently requires the reference electrostatics solver"
        )
    if deployed.config.electrostatics.periodicity_profile in {"molecular", "z_slab"}:
        raise DeployError(
            "native Polar requires a periodic bulk electrostatics profile"
        )
    configuration = Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=tuple(atoms.pbc),
        properties={
            name: atoms.info[name]
            for name in ("total_charge", "total_spin", "external_field")
            if name in atoms.info
        },
        head=deployed.select_head(),
    )
    graph = build_batch(
        [configuration],
        atomic_number_table=deployed.z_table,
        cutoff=deployed.r_max,
        heads=deployed.heads,
        neighbor_backend=resolve_neighbor_provider(None, device=torch.device(device)),
        features=deployed.config.model.feature_storages,
        dtype=deployed.dtype,
        device=device,
    )
    for name in ("total_charge", "total_spin", "external_field"):
        if not torch.isfinite(graph[name]).all():
            raise DeployError(f"Polar export input {name!r} must be finite")
    live = graph["edge_index"].shape[1]
    if live > edges:
        raise DeployError(f"export structure needs {live} edges, capacity is {edges}")
    # Zero-cutoff self edges pad only the local network. There is no padding
    # atom in the charge constraint or in the reciprocal-space solve.
    index = torch.zeros((2, edges), dtype=torch.long, device=device)
    index[:, :live] = graph["edge_index"]
    positions = graph["positions"].detach()
    cell = graph["cell"].view(3, 3).detach()
    unit_shifts = positions.new_zeros((edges, 3))
    unit_shifts[:, 0] = (
        int(np.ceil(2 * deployed.r_max / np.linalg.norm(atoms.cell[0]))) + 1
    )
    unit_shifts[:live] = graph["unit_shifts"]
    elements = graph["node_attrs"].argmax(-1)
    model = (
        PeriodicEnergy(deployed, graph).to(device=device).eval().requires_grad_(False)
    )
    args = (unit_shifts, index, elements, positions, cell)
    moved = positions.clone()
    moved[0] += positions.new_tensor([0.013, -0.007, 0.009])
    variants = (args, (unit_shifts, index, elements, moved, cell))
    return model, variants
