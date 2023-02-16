import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R
import dataclasses

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)

Z_TABLE = tools.AtomicNumberTable([1, 8])


@pytest.fixture
def default_params():
    atomic_energies = np.array([1.0, 3.0], dtype=float)
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
        atomic_numbers=Z_TABLE.zs,
        correlation=3,
    )
    return model_config


@pytest.fixture
def water_configuration():
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
    return config


@pytest.fixture
def rotation():
    rot = R.from_euler("z", 60, degrees=True).as_matrix()
    return rot


@pytest.fixture
def rotated_water(water_configuration, rotation):
    config_rotated = dataclasses.replace(water_configuration)
    positions_rotated = np.array(rotation @ config_rotated.positions.T).T
    config_rotated.positions = positions_rotated
    return config_rotated


def test_mace(default_params, water_configuration, rotated_water):
    # Create MACE model
    model = modules.MACE(**default_params)
    model_compiled = jit.compile(model)

    config = water_configuration
    config_rotated = rotated_water

    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=Z_TABLE, cutoff=3.0
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


def test_dipole_mace(default_params, water_configuration, rotated_water, rotation):
    # create dipole MACE model
    model_config = default_params.copy()
    model_config.update(
        {
            "hidden_irreps": o3.Irreps("16x0e + 16x1o + 16x2e"),
            "atomic_energies": None,
            "avg_num_neighbors": 3,
        }
    )
    model = modules.AtomicDipolesMACE(**model_config)

    config = water_configuration
    config_rotated = rotated_water

    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=Z_TABLE, cutoff=3.0
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
        np.array(rotation @ output["dipole"][0].detach().numpy()),
        output["dipole"][1].detach().numpy(),
    )


def test_energy_dipole_mace(
    default_params, water_configuration, rotated_water, rotation
):
    # create dipole MACE model
    model_config = default_params.copy()
    model_config.update(
        {
            "hidden_irreps": o3.Irreps("16x0e + 16x1o + 16x2e"),
            "avg_num_neighbors": 3,
            "num_interactions": 2,
        }
    )
    model = modules.EnergyDipolesMACE(**model_config)

    config = water_configuration
    config_rotated = rotated_water

    atomic_data = data.AtomicData.from_config(config, z_table=Z_TABLE, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=Z_TABLE, cutoff=3.0
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
        np.array(rotation @ output["dipole"][0].detach().numpy()),
        output["dipole"][1].detach().numpy(),
    )


@pytest.mark.parametrize(
    "param_changes, expected_shape",
    [
        (
            {},
            (
                3,
                5,
            ),
        ),
        (
            {"num_interactions": 1, "hidden_irreps": o3.Irreps("96x0e + 96x1o")},
            (
                3,
                1,
            ),
        ),
    ],
)
def test_get_local_embeddings(
    default_params, water_configuration, param_changes, expected_shape
):
    default_params = default_params.copy()
    default_params.update(param_changes)
    model = modules.MACE(**default_params)
    atomic_data = data.AtomicData.from_config(
        water_configuration, z_table=Z_TABLE, cutoff=3.0
    )
    embeddings = model.get_node_invariant_descriptors(atomic_data)
    assert embeddings.shape == (*expected_shape, model._num_invariant_features)
