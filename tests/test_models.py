import numpy as np
import torch
import torch.nn.functional
from ase import build
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric

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
    properties={
        "forces": np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        "energy": -1.5,
        "charges": np.array([-2.0, 1.0, 1.0]),
        "dipole": np.array([-1.5, 1.5, 2.0]),
        "polarizability": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    },
    property_weights={
        "forces": 1.0,
        "energy": 1.0,
        "charges": 1.0,
        "dipole": 1.0,
        "polarizability": 1.0,
    },
)
# Created the rotated environment
rot = R.from_euler("z", 60, degrees=True).as_matrix()
positions_rotated = np.array(rot @ config.positions.T).T
config_rotated = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=positions_rotated,
    properties={
        "forces": np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        "energy": -1.5,
        "charges": np.array([-2.0, 1.0, 1.0]),
        "dipole": np.array([-1.5, 1.5, 2.0]),
        "polarizability": np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    },
    property_weights={
        "forces": 1.0,
        "energy": 1.0,
        "charges": 1.0,
        "dipole": 1.0,
        "polarizability": 1.0,
    },
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


def test_dipole_polar_mace():
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
        MLP_irreps=o3.Irreps("16x0e + 16x1o + 16x2e"),
        gate=torch.nn.functional.silu,
        atomic_energies=None,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="gaussian",
    )
    model = modules.AtomicDielectricMACE(**model_config)

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
    # sanity check of polarizability being the right shape
    assert (
        output["polarizability"][0].unsqueeze(0).shape
        == atomic_data.polarizability.shape
    )
    # test equivariance of output polarizability
    assert np.allclose(
        np.array(rot @ output["polarizability"][0].detach().numpy() @ rot.T),
        output["polarizability"][1].detach().numpy(),
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


def test_mace_multi_reference():
    atomic_energies_multi = np.array([[1.0, 3.0], [0.0, 0.0]], dtype=float)
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("96x0e + 96x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies_multi,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        distance_transform=True,
        pair_repulsion=True,
        correlation=3,
        heads=["Default", "dft"],
        # radial_type="chebyshev",
        atomic_inter_scale=[1.0, 1.0],
        atomic_inter_shift=[0.0, 0.1],
    )
    model = modules.ScaleShiftMACE(**model_config)
    model_compiled = jit.compile(model)
    config.head = "Default"
    config_rotated.head = "dft"
    atomic_data = data.AtomicData.from_config(
        config, z_table=table, cutoff=3.0, heads=["Default", "dft"]
    )
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0, heads=["Default", "dft"]
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
    assert output2["energy"].shape[0] == 2


def test_atomic_virials_stresses():
    """
    Test that atomic virials and stresses sum to the total virials and stress.
    """
    # Set default dtype for reproducibility
    torch.set_default_dtype(torch.float64)

    # Create a periodic cell with ASE
    atoms = build.bulk("Si", "diamond", a=5.43)
    # Apply strain to ensure non-zero stress
    strain_tensor = np.eye(3) * 1.02  # 2% strain
    atoms.set_cell(np.dot(atoms.get_cell(), strain_tensor), scale_atoms=True)

    # Add forces and energy for completeness
    atoms.arrays["REF_forces"] = np.random.normal(0, 0.1, size=atoms.positions.shape)
    atoms.info["REF_energy"] = np.random.normal(0, 1)
    atoms.info["REF_stress"] = np.random.normal(0, 0.1, size=6)

    # Setup MACE model configuration
    stress_z_table = tools.AtomicNumberTable([14])  # Silicon
    stress_atomic_energies = np.array([0.0])

    model_config = dict(
        r_max=5.0,
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
        atomic_energies=stress_atomic_energies,
        avg_num_neighbors=4.0,
        atomic_numbers=table.zs,
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )

    # Create the model
    model = modules.ScaleShiftMACE(**model_config)

    # Create atomic data
    atomic_data = data.AtomicData.from_config(
        data.config_from_atoms(
            atoms, key_specification=data.KeySpecification.from_defaults()
        ),
        z_table=stress_z_table,
        cutoff=5.0,
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch_dict = batch.to_dict()

    # Run the model with compute_atomic_stresses=True
    output = model(
        batch_dict,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
        compute_atomic_stresses=True,
    )

    # Get total virials/stress and atomic virials/stresses
    total_virials = output["virials"]
    atomic_virials = output["atomic_virials"]
    total_stress = output["stress"]
    atomic_stresses = output["atomic_stresses"]

    # Test that atomic values are not None
    assert atomic_virials is not None, "Atomic virials were not computed"
    assert atomic_stresses is not None, "Atomic stresses were not computed"

    # Test shape of atomic values
    assert atomic_virials.shape[0] == len(atoms), "Wrong shape for atomic virials"
    assert atomic_virials.shape[1:] == (3, 3), "Atomic virials should be 3x3 matrices"
    assert atomic_stresses.shape[0] == len(atoms), "Wrong shape for atomic stresses"
    assert atomic_stresses.shape[1:] == (3, 3), "Atomic stresses should be 3x3 matrices"

    # Compute sum of atomic values
    summed_atomic_virials = torch.sum(atomic_virials, dim=0)
    summed_atomic_stresses = torch.sum(atomic_stresses, dim=0)

    # Test that sums match total values
    assert torch.allclose(
        summed_atomic_virials, total_virials.squeeze(0), atol=1e-6
    ), f"Sum of atomic virials {summed_atomic_virials} does not match total virials {total_virials.squeeze(0)}"

    assert torch.allclose(
        summed_atomic_stresses, total_stress.squeeze(0), atol=1e-6
    ), f"Sum of atomic stresses (normalized by volume) {summed_atomic_stresses} does not match total stress {total_stress.squeeze(0)}"
