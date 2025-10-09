###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from copy import deepcopy
from typing import Optional, Sequence

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
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        head: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        dipole_weight: Optional[torch.Tensor],  # [,]
        charges_weight: Optional[torch.Tensor],  # [,]
        polarizability_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
        polarizability: Optional[torch.Tensor],  # [1, 3, 3]
        elec_temp: Optional[torch.Tensor],  # [,]
        total_charge: Optional[torch.Tensor] = None,  # [,]
        total_spin: Optional[torch.Tensor] = None,  # [,]
        pbc: Optional[torch.Tensor] = None,  # [, 3]
        **extra_data # additional properties that aren't hard-coded and therefore not
                     # for correct shape, etc
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
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
        super().__init__(**data, **extra_data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        heads: Optional[list] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "AtomicData":
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
        )
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        num_atoms = len(config.atomic_numbers)

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        energy_weight = (
            torch.tensor(
                config.property_weights.get("energy"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("energy") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces_weight = (
            torch.tensor(
                config.property_weights.get("forces"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("forces") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        stress_weight = (
            torch.tensor(
                config.property_weights.get("stress"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("stress") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        virials_weight = (
            torch.tensor(
                config.property_weights.get("virials"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("virials") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        dipole_weight = (
            torch.tensor(
                config.property_weights.get("dipole"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("dipole") is not None
            else torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype())
        )
        if len(dipole_weight.shape) == 0:
            dipole_weight = dipole_weight * torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype()
            )
        elif len(dipole_weight.shape) == 1:
            dipole_weight = dipole_weight.unsqueeze(0)

        charges_weight = (
            torch.tensor(
                config.property_weights.get("charges"), dtype=torch.get_default_dtype()
            )
            if config.property_weights.get("charges") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        polarizability_weight = (
            torch.tensor(
                config.property_weights.get("polarizability"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("polarizability") is not None
            else torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                dtype=torch.get_default_dtype(),
            )
        )
        if len(polarizability_weight.shape) == 0:
            polarizability_weight = polarizability_weight * torch.tensor(
                [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
                dtype=torch.get_default_dtype(),
            )
        elif len(polarizability_weight.shape) == 2:
            polarizability_weight = polarizability_weight.unsqueeze(0)
        forces = (
            torch.tensor(
                config.properties.get("forces"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("forces") is not None
            else torch.zeros(num_atoms, 3, dtype=torch.get_default_dtype())
        )
        energy = (
            torch.tensor(
                config.properties.get("energy"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("energy") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("stress"), dtype=torch.get_default_dtype()
                )
            ).unsqueeze(0)
            if config.properties.get("stress") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("virials"), dtype=torch.get_default_dtype()
                )
            ).unsqueeze(0)
            if config.properties.get("virials") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        dipole = (
            torch.tensor(
                config.properties.get("dipole"), dtype=torch.get_default_dtype()
            ).unsqueeze(0)
            if config.properties.get("dipole") is not None
            else torch.zeros(1, 3, dtype=torch.get_default_dtype())
        )
        charges = (
            torch.tensor(
                config.properties.get("charges"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("charges") is not None
            else torch.zeros(num_atoms, dtype=torch.get_default_dtype())
        )
        elec_temp = (
            torch.tensor(
                config.properties.get("elec_temp"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("elec_temp") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        total_charge = (
            torch.tensor(
                config.properties.get("total_charge"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("total_charge") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        polarizability = (
            torch.tensor(
                config.properties.get("polarizability"), dtype=torch.get_default_dtype()
            ).view(1, 3, 3)
            if config.properties.get("polarizability") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )

        total_spin = (
            torch.tensor(
                config.properties.get("total_spin"), dtype=torch.get_default_dtype()
            )
            if config.properties.get("total_spin") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )
        if config.pbc is not None:
            pbc = list(bool(pbc_) for pbc_ in config.pbc)
        else:
            pbc = None
        pbc = (
            torch.tensor([pbc], dtype=torch.bool)
            if pbc is not None
            else torch.tensor([[False, False, False]], dtype=torch.bool)
        )

        cls_kwargs = dict(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
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

        # Add other properties that aren't checked. WARNING: shape dependence
        # and unsqueeze(-1) call may implicitly be assuming that these are all
        # per-atom, not per-config.
        #
        # NOTE: might be able to simpligy a lot by adding most properties this
        # way, except a few that need to be special cased, e.g. are mandatory,
        # really need a default value (e.g. weight), or need shape conversion
        # (e.g. stress/virial). Getting the right shape might require some knowledge
        # of whether the property is per-config or per-atom
        for k, v in config.properties.items():
            if k not in cls_kwargs and v is not None:
                if len(v.shape) == 1:
                    # promote to n_atoms x 1
                    cls_kwargs[k] = torch.tensor(v, dtype=torch.get_default_dtype()).unsqueeze(-1)
                elif len(v.shape) == 2:
                    cls_kwargs[k] = torch.tensor(v, dtype=torch.get_default_dtype())

        return cls(**cls_kwargs)


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
