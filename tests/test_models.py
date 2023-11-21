from typing import List
import ase
import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from ase.build import bulk, make_supercell
import numpy as np
from mace.data import get_neighborhood

torch.set_default_dtype(torch.float64)
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
    charges=np.array([-2.0, 1.0, 1.0]),
    dipole=np.array([-1.5, 1.5, 2.0]),
)
# Created the rotated environment
rot = R.from_euler("z", 60, degrees=True).as_matrix()
positions_rotated = np.array(rot @ config.positions.T).T
config_rotated = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=positions_rotated,
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
    charges=np.array([-2.0, 1.0, 1.0]),
    dipole=np.array([-1.5, 1.5, 2.0]),
)
table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)


def test_mace():
    # Create MACE model
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=5,
        num_elements=2,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )
    model = modules.MACE(**model_config)
    model_compiled = jit.compile(model)

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output1 = model(batch.to_dict(), training=True)
    output2 = model_compiled(batch.to_dict(), training=True)
    assert torch.allclose(output1["energy"][0], output2["energy"][0])
    assert torch.allclose(output2["energy"][0], output2["energy"][1])


def test_dipole_mace():
    # create dipole MACE model
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o + 16x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=None,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="gaussian",
    )
    model = modules.AtomicDipolesMACE(**model_config)

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output = model(
        batch,
        training=True,
    )
    # sanity check of dipoles being the right shape
    assert output["dipole"][0].unsqueeze(0).shape == atomic_data.dipole.shape
    # test equivariance of output dipoles
    assert np.allclose(
        np.array(rot @ output["dipole"][0].detach().numpy()),
        output["dipole"][1].detach().numpy(),
    )


def test_energy_dipole_mace():
    # create dipole MACE model
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o + 16x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
    )
    model = modules.EnergyDipolesMACE(**model_config)

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output = model(
        batch,
        training=True,
    )
    # sanity check of dipoles being the right shape
    assert output["dipole"][0].unsqueeze(0).shape == atomic_data.dipole.shape
    # test energy is invariant
    assert torch.allclose(output["energy"][0], output["energy"][1])
    # test equivariance of output dipoles
    assert np.allclose(
        np.array(rot @ output["dipole"][0].detach().numpy()),
        output["dipole"][1].detach().numpy(),
    )


def create_example_graphs(atoms: ase.Atoms, local_list: List[int], r_cut: int):
    ghost_list_1 = []
    ghost_list_2 = []
    # Get ghost_list_1 and edge_index_1 (for within 1*r_cut)
    edge_index_1, _, _ = get_neighborhood(
        positions=atoms.positions, cutoff=r_cut, pbc=None
    )
    for i, j in edge_index_1.T:
        if i in local_list and j not in local_list and j not in ghost_list_1:
            ghost_list_1.append(j)
    # Get ghost_list_2 and edge_index_2 (for 2*r_cut)
    edge_index_2, _, _ = get_neighborhood(
        positions=atoms.positions, cutoff=2 * r_cut, pbc=None
    )
    for i, j in edge_index_2.T:
        if (
            i in local_list
            and j not in local_list
            and j not in ghost_list_1
            and j not in ghost_list_2
        ):
            ghost_list_2.append(j)
    # Reorder atoms.positions
    new_order = local_list + ghost_list_1 + ghost_list_2
    reordered_positions = atoms.positions[new_order]
    # Update edge_index_1 and edge_index_2 to reflect the new ordering
    mapping = {old: new for new, old in enumerate(new_order)}
    edge_index = np.array([[mapping[i], mapping[j]] for i, j in edge_index_2.T]).T
    # Edge masks
    ghost_list_1 = [i + len(local_list) for i in range(len(ghost_list_1))]
    edge_mask_1 = np.isin(edge_index[0], local_list + ghost_list_1)
    edge_mask_2 = np.isin(edge_index[0], local_list + ghost_list_1 + ghost_list_2)
    atoms.positions = reordered_positions
    atoms.symbols = atoms.symbols[new_order]
    node_mask = [local_list, ghost_list_1]
    edge_mask = [edge_mask_2, edge_mask_1]
    return (
        atoms,
        edge_mask,
        node_mask,
    )


def test_mace_ghost():
    # Create MACE model
    table = tools.AtomicNumberTable([26])
    atomic_energies = np.array([1.0], dtype=float)
    model_config = dict(
        r_max=8.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=3,
        num_elements=1,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )
    model = modules.MACE(**model_config)
    model_compiled = jit.compile(model)
    atoms_example = bulk("Fe", a=3.0, cubic=True)
    atoms_example = make_supercell(atoms_example, 4 * np.identity(3))
    atoms_example.wrap()
    local_list = [i for i in range(4)]
    r_cut = 8.0
    atoms_example, edge_mask, node_mask = create_example_graphs(
        atoms_example, local_list, r_cut
    )
    config = config_from_atoms(atoms_example)
    table = tools.AtomicNumberTable([26])
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=r_cut)
    atomic_data2 = data.AtomicData.from_config(
        config,
        z_table=table,
        cutoff=3.0,
        edge_mask_index=edge_mask,
        node_mask_index=node_mask,
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output1 = model(batch.to_dict(), training=True)
    output2 = model_compiled(batch.to_dict(), training=True)
    assert torch.allclose(output1["energy"][0], output2["energy"][0])
