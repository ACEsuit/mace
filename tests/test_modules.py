import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3

from mace.data import AtomicData, Configuration
from mace.modules import (
    AtomicEnergiesBlock,
    BesselBasis,
    PolynomialCutoff,
    SymmetricContraction,
    WeightedEnergyForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
    compute_mean_rms_energy_forces,
    compute_statistics,
)
from mace.tools import AtomicNumberTable, scatter, to_numpy, torch_geometric
from mace.tools.scripts_utils import dict_to_array


@pytest.fixture(name="config")
def _config():
    return Configuration(
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
            "stress": np.array([1.0, 0.0, 0.5, 0.0, -1.0, 0.0]),
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
            "stress": 1.0,
        },
    )


@pytest.fixture(name="table")
def _table():
    return AtomicNumberTable([1, 8])


@pytest.fixture(name="config1")
def _config1():
    return Configuration(
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
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
        },
        head="DFT",
    )


@pytest.fixture(name="config2")
def _config2():
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.1, -1.9, 0.1],
                [1.1, 0.1, 0.1],
                [0.1, 1.1, 0.1],
            ]
        ),
        properties={
            "forces": np.array(
                [
                    [0.1, -1.2, 0.1],
                    [1.1, 0.3, 0.1],
                    [0.1, 1.2, 0.4],
                ]
            ),
            "energy": -1.4,
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
        },
        head="MP2",
    )


@pytest.fixture(name="atomic_data")
def _atomic_data(config1, config2, table):
    atomic_data1 = AtomicData.from_config(
        config1, z_table=table, cutoff=3.0, heads=["DFT", "MP2"]
    )
    atomic_data2 = AtomicData.from_config(
        config2, z_table=table, cutoff=3.0, heads=["DFT", "MP2"]
    )
    return [atomic_data1, atomic_data2]


@pytest.fixture(name="data_loader")
def _data_loader(atomic_data):
    return torch_geometric.dataloader.DataLoader(
        dataset=atomic_data,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )


@pytest.fixture(name="atomic_energies")
def _atomic_energies():
    atomic_energies_dict = {
        "DFT": np.array([0.0, 0.0]),
        "MP2": np.array([0.1, 0.1]),
    }
    return dict_to_array(atomic_energies_dict, ["DFT", "MP2"])


@pytest.fixture(autouse=True)
def _set_torch_default_dtype():
    torch.set_default_dtype(torch.float64)


def test_weighted_loss(config, table):
    loss1 = WeightedEnergyForcesLoss(energy_weight=1, forces_weight=10)
    loss2 = WeightedHuberEnergyForcesStressLoss(energy_weight=1, forces_weight=10)
    data = AtomicData.from_config(config, z_table=table, cutoff=3.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data, data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    pred = {
        "energy": batch.energy,
        "forces": batch.forces,
        "stress": batch.stress,
    }
    out1 = loss1(batch, pred)
    assert out1 == 0.0
    out2 = loss2(batch, pred)
    assert out2 == 0.0


def test_symmetric_contraction():
    operation = SymmetricContraction(
        irreps_in=o3.Irreps("16x0e + 16x1o + 16x2e"),
        irreps_out=o3.Irreps("16x0e + 16x1o"),
        correlation=3,
        num_elements=2,
    )
    torch.manual_seed(123)
    features = torch.randn(30, 16, 9)
    one_hots = torch.nn.functional.one_hot(torch.arange(0, 30) % 2).to(
        torch.get_default_dtype()
    )
    out = operation(features, one_hots)
    assert out.shape == (30, 64)
    assert operation.contractions[0].weights_max.shape == (2, 11, 16)


def test_bessel_basis():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    bessel_basis = BesselBasis(r_max=6.0, num_basis=5)
    output = bessel_basis(d.unsqueeze(-1))
    assert output.shape == (10, 5)


def test_polynomial_cutoff():
    d = torch.linspace(start=0.5, end=5.5, steps=10)
    cutoff_fn = PolynomialCutoff(r_max=5.0)
    output = cutoff_fn(d)
    assert output.shape == (10,)


def test_atomic_energies(config, table):
    energies_block = AtomicEnergiesBlock(atomic_energies=np.array([1.0, 3.0]))
    data = AtomicData.from_config(config, z_table=table, cutoff=3.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data, data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    energies = energies_block(batch.node_attrs).squeeze(-1)
    out = scatter.scatter_sum(src=energies, index=batch.batch, dim=-1, reduce="sum")
    out = to_numpy(out)
    assert np.allclose(out, np.array([5.0, 5.0]))


def test_atomic_energies_multireference(config, table):
    energies_block = AtomicEnergiesBlock(
        atomic_energies=np.array([[1.0, 3.0], [2.0, 4.0]])
    )
    config.head = "MP2"
    data = AtomicData.from_config(
        config, z_table=table, cutoff=3.0, heads=["DFT", "MP2"]
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[data, data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    num_atoms_arange = torch.arange(batch["positions"].shape[0])
    node_heads = (
        batch["head"][batch["batch"]]
        if "head" in batch
        else torch.zeros_like(batch["batch"])
    )
    energies = energies_block(batch.node_attrs).squeeze(-1)
    energies = energies[num_atoms_arange, node_heads]
    out = scatter.scatter_sum(src=energies, index=batch.batch, dim=-1, reduce="sum")
    out = to_numpy(out)
    assert np.allclose(out, np.array([8.0, 8.0]))


def test_compute_mean_rms_energy_forces_multi_head(data_loader, atomic_energies):
    mean, rms = compute_mean_rms_energy_forces(data_loader, atomic_energies)
    assert isinstance(mean, np.ndarray)
    assert isinstance(rms, np.ndarray)
    assert mean.shape == (2,)
    assert rms.shape == (2,)
    assert np.all(rms >= 0)
    assert rms[0] != rms[1]


def test_compute_statistics(data_loader, atomic_energies):
    avg_num_neighbors, mean, std = compute_statistics(data_loader, atomic_energies)
    assert isinstance(avg_num_neighbors, float)
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert avg_num_neighbors > 0
    assert np.all(mean != 0)
    assert np.all(std > 0)
    assert mean[0] != mean[1]
    assert std[0] != std[1]
