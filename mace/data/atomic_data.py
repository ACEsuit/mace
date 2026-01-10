###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from collections.abc import Sequence
from copy import deepcopy

import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    polarizability: torch.Tensor
    total_charge: torch.Tensor
    total_spin: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    dipole_weight: torch.Tensor
    charges_weight: torch.Tensor
    polarizability_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: torch.Tensor | None,  # [3,3]
        weight: torch.Tensor | None,  # [,]
        head: torch.Tensor | None,  # [,]
        energy_weight: torch.Tensor | None,  # [,]
        forces_weight: torch.Tensor | None,  # [,]
        stress_weight: torch.Tensor | None,  # [,]
        virials_weight: torch.Tensor | None,  # [,]
        dipole_weight: torch.Tensor | None,  # [,]
        charges_weight: torch.Tensor | None,  # [,]
        polarizability_weight: torch.Tensor | None,  # [,]
        forces: torch.Tensor | None,  # [n_nodes, 3]
        energy: torch.Tensor | None,  # [, ]
        stress: torch.Tensor | None,  # [1,3,3]
        virials: torch.Tensor | None,  # [1,3,3]
        dipole: torch.Tensor | None,  # [, 3]
        charges: torch.Tensor | None,  # [n_nodes, ]
        polarizability: torch.Tensor | None,  # [1, 3, 3]
        elec_temp: torch.Tensor | None,  # [,]
        total_charge: torch.Tensor | None = None,  # [,]
        total_spin: torch.Tensor | None = None,  # [,]
        pbc: torch.Tensor | None = None,  # [, 3]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2
        assert len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert head is None or len(head.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert dipole_weight is None or dipole_weight.shape == (1, 3), dipole_weight
        assert charges_weight is None or len(charges_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        assert elec_temp is None or len(elec_temp.shape) == 0
        assert total_charge is None or len(total_charge.shape) == 0
        assert total_spin is None or len(total_spin.shape) == 0
        assert polarizability is None or polarizability.shape == (1, 3, 3)
        assert pbc is None or (pbc.shape[-1] == 3 and pbc.dtype == torch.bool)
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "weight": weight,
            "head": head,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "dipole_weight": dipole_weight,
            "charges_weight": charges_weight,
            "polarizability_weight": polarizability_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
            "polarizability": polarizability,
            "elec_temp": elec_temp,
            "total_charge": total_charge,
            "total_spin": total_spin,
            "pbc": pbc,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        heads: list | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "AtomicData":
        dtype = dtype or torch.get_default_dtype()
        if heads is None:
            heads = ["Default"]
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=config.positions,
            cutoff=cutoff,
            pbc=deepcopy(config.pbc),
            cell=deepcopy(config.cell),
        )
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
            dtype=dtype,
        )
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)

        cell = (
            torch.tensor(cell, dtype=dtype)
            if cell is not None
            else torch.tensor(3 * [0.0, 0.0, 0.0], dtype=dtype).view(3, 3)
        )

        num_atoms = len(config.atomic_numbers)

        weight = (
            torch.tensor(config.weight, dtype=dtype)
            if config.weight is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        energy_weight = (
            torch.tensor(config.property_weights.get("energy"), dtype=dtype)
            if config.property_weights.get("energy") is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        forces_weight = (
            torch.tensor(config.property_weights.get("forces"), dtype=dtype)
            if config.property_weights.get("forces") is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        stress_weight = (
            torch.tensor(config.property_weights.get("stress"), dtype=dtype)
            if config.property_weights.get("stress") is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        virials_weight = (
            torch.tensor(config.property_weights.get("virials"), dtype=dtype)
            if config.property_weights.get("virials") is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        dipole_weight = (
            torch.tensor(config.property_weights.get("dipole"), dtype=dtype)
            if config.property_weights.get("dipole") is not None
            else torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
        )
        if len(dipole_weight.shape) == 0:
            dipole_weight = dipole_weight * torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
        elif len(dipole_weight.shape) == 1:
            dipole_weight = dipole_weight.unsqueeze(0)

        charges_weight = (
            torch.tensor(config.property_weights.get("charges"), dtype=dtype)
            if config.property_weights.get("charges") is not None
            else torch.tensor(1.0, dtype=dtype)
        )
        polarizability_weight = (
            torch.tensor(
                config.property_weights.get("polarizability"),
                dtype=dtype,
            )
            if config.property_weights.get("polarizability") is not None
            else torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                dtype=dtype,
            )
        )
        if len(polarizability_weight.shape) == 0:
            polarizability_weight = polarizability_weight * torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                dtype=dtype,
            )
        elif len(polarizability_weight.shape) == 2:
            polarizability_weight = polarizability_weight.unsqueeze(0)
        forces = (
            torch.tensor(config.properties.get("forces"), dtype=dtype)
            if config.properties.get("forces") is not None
            else torch.zeros(num_atoms, 3, dtype=dtype)
        )
        energy = (
            torch.tensor(config.properties.get("energy"), dtype=dtype)
            if config.properties.get("energy") is not None
            else torch.tensor(0.0, dtype=dtype)
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(config.properties.get("stress"), dtype=dtype)
            ).unsqueeze(0)
            if config.properties.get("stress") is not None
            else torch.zeros(1, 3, 3, dtype=dtype)
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(config.properties.get("virials"), dtype=dtype)
            ).unsqueeze(0)
            if config.properties.get("virials") is not None
            else torch.zeros(1, 3, 3, dtype=dtype)
        )
        dipole = (
            torch.tensor(config.properties.get("dipole"), dtype=dtype).unsqueeze(0)
            if config.properties.get("dipole") is not None
            else torch.zeros(1, 3, dtype=dtype)
        )
        charges = (
            torch.tensor(config.properties.get("charges"), dtype=dtype)
            if config.properties.get("charges") is not None
            else torch.zeros(num_atoms, dtype=dtype)
        )
        elec_temp = (
            torch.tensor(
                config.properties.get("elec_temp"),
                dtype=dtype,
            )
            if config.properties.get("elec_temp") is not None
            else torch.tensor(0.0, dtype=dtype)
        )

        total_charge = (
            torch.tensor(config.properties.get("total_charge"), dtype=dtype)
            if config.properties.get("total_charge") is not None
            else torch.tensor(0.0, dtype=dtype)
        )

        polarizability = (
            torch.tensor(config.properties.get("polarizability"), dtype=dtype).view(
                1, 3, 3
            )
            if config.properties.get("polarizability") is not None
            else torch.zeros(1, 3, 3, dtype=dtype)
        )

        total_spin = (
            torch.tensor(config.properties.get("total_spin"), dtype=dtype)
            if config.properties.get("total_spin") is not None
            else torch.tensor(1.0, dtype=dtype)
        )

        pbc = (
            torch.tensor([config.pbc], dtype=torch.bool)
            if config.pbc is not None
            else torch.tensor([[False, False, False]], dtype=torch.bool)
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=dtype),
            shifts=torch.tensor(shifts, dtype=dtype),
            unit_shifts=torch.tensor(unit_shifts, dtype=dtype),
            cell=cell,
            node_attrs=one_hot,
            weight=weight,
            head=head,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            dipole_weight=dipole_weight,
            charges_weight=charges_weight,
            polarizability_weight=polarizability_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
            elec_temp=elec_temp,
            total_charge=total_charge,
            polarizability=polarizability,
            total_spin=total_spin,
            pbc=pbc,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
