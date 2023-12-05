import numpy as np
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
)
from mace.tools import AtomicNumberTable, scatter, to_numpy, torch_geometric

config = Configuration(
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
    # stress if voigt 6 notation
    stress=np.array([1.0, 0.0, 0.5, 0.0, -1.0, 0.0]),
)

table = AtomicNumberTable([1, 8])

torch.set_default_dtype(torch.float64)


class TestLoss:
    def test_weighted_loss(self):
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


class TestSymmetricContract:
    def test_symmetric_contraction(self):
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


class TestBlocks:
    def test_bessel_basis(self):
        d = torch.linspace(start=0.5, end=5.5, steps=10)
        bessel_basis = BesselBasis(r_max=6.0, num_basis=5)
        output = bessel_basis(d.unsqueeze(-1))
        assert output.shape == (10, 5)

    def test_polynomial_cutoff(self):
        d = torch.linspace(start=0.5, end=5.5, steps=10)
        cutoff_fn = PolynomialCutoff(r_max=5.0)
        output = cutoff_fn(d)
        assert output.shape == (10,)

    def test_atomic_energies(self):
        energies_block = AtomicEnergiesBlock(atomic_energies=np.array([1.0, 3.0]))

        data = AtomicData.from_config(config, z_table=table, cutoff=3.0)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data, data],
            batch_size=2,
            shuffle=True,
            drop_last=False,
        )
        batch = next(iter(data_loader))

        energies = energies_block(batch.node_attrs)
        out = scatter.scatter_sum(src=energies, index=batch.batch, dim=-1, reduce="sum")
        out = to_numpy(out)
        assert np.allclose(out, np.array([5.0, 5.0]))
