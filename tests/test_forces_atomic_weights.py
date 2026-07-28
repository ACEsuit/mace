"""Tests for per-atom (e.g. per-element) force loss weighting."""

import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from mace import data
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.modules import (
    UniversalLoss,
    WeightedEnergyForcesLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
)
from mace.tools import AtomicNumberTable, torch_geometric

torch.set_default_dtype(torch.float64)

TABLE = AtomicNumberTable([1, 8])
CUTOFF = 3.0


def make_config(atomic_weights=None):
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
            "energy": -1.5,
            "forces": np.array(
                [
                    [0.0, -1.3, 0.0],
                    [1.0, 0.2, 0.0],
                    [0.0, 1.1, 0.3],
                ]
            ),
            "stress": np.zeros(6),
        },
        property_weights={"energy": 1.0, "forces": 1.0, "stress": 1.0},
    )
    if atomic_weights is not None:
        config.properties["forces_atomic_weights"] = np.asarray(atomic_weights)
    return config


def make_batch(configs):
    dataset = [AtomicData.from_config(c, z_table=TABLE, cutoff=CUTOFF) for c in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    return next(iter(loader))


def make_pred(batch, forces_offset):
    return {
        "energy": batch.energy.clone(),
        "forces": batch.forces + forces_offset,
        "stress": batch.stress.clone(),
    }


@pytest.mark.parametrize(
    "loss_cls",
    [
        WeightedEnergyForcesLoss,
        WeightedForcesLoss,
        WeightedHuberEnergyForcesStressLoss,
        UniversalLoss,
    ],
)
def test_uniform_weights_match_default(loss_cls):
    """weights == 1 must reproduce the unweighted loss exactly."""
    loss_fn = loss_cls(forces_weight=1.0)
    batch_default = make_batch([make_config(), make_config()])
    batch_ones = make_batch(
        [make_config([1.0, 1.0, 1.0]), make_config([1.0, 1.0, 1.0])]
    )
    offset = 0.1 * torch.ones_like(batch_default.forces)
    out_default = loss_fn(batch_default, make_pred(batch_default, offset))
    out_ones = loss_fn(batch_ones, make_pred(batch_ones, offset))
    assert torch.allclose(out_default, out_ones)


def test_weights_scale_mse_loss():
    """Uniform weight w scales the (linear-in-weight) MSE force loss by w."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    batch_1 = make_batch([make_config([1.0, 1.0, 1.0])])
    batch_2 = make_batch([make_config([2.0, 2.0, 2.0])])
    offset = 0.1 * torch.ones_like(batch_1.forces)
    out_1 = loss_fn(batch_1, make_pred(batch_1, offset))
    out_2 = loss_fn(batch_2, make_pred(batch_2, offset))
    assert torch.allclose(out_2, 2.0 * out_1)


def test_zero_weight_masks_atom():
    """An atom with weight 0 must not contribute to the force loss."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    batch = make_batch([make_config([0.0, 1.0, 1.0])])
    # error only on the zero-weighted atom
    offset = torch.zeros_like(batch.forces)
    offset[0] = 100.0
    out = loss_fn(batch, make_pred(batch, offset))
    assert out.item() == pytest.approx(0.0)
    # sanity: same error on a weighted atom is nonzero
    offset2 = torch.zeros_like(batch.forces)
    offset2[1] = 100.0
    out2 = loss_fn(batch, make_pred(batch, offset2))
    assert out2.item() > 0.0


def test_per_element_weighting_ratio():
    """Element-keyed weights: minority-atom error is up-weighted as expected."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    element_weights = {8: 5.0, 1: 1.0}
    config = make_config()
    config.properties["forces_atomic_weights"] = np.array(
        [element_weights[z] for z in config.atomic_numbers]
    )
    batch = make_batch([config])
    err_on_O = torch.zeros_like(batch.forces)
    err_on_O[0] = 1.0
    err_on_H = torch.zeros_like(batch.forces)
    err_on_H[1] = 1.0
    out_O = loss_fn(batch, make_pred(batch, err_on_O))
    out_H = loss_fn(batch, make_pred(batch, err_on_H))
    assert torch.allclose(out_O, 5.0 * out_H)


def test_mixed_batch_with_and_without_weights():
    """Configs lacking the column default to 1 and collate with weighted ones."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    batch = make_batch([make_config([3.0, 3.0, 3.0]), make_config()])
    assert batch.forces_atomic_weights.shape == (6, 1)
    assert torch.allclose(
        batch.forces_atomic_weights.view(-1),
        torch.tensor([3.0, 3.0, 3.0, 1.0, 1.0, 1.0]),
    )
    offset = 0.1 * torch.ones_like(batch.forces)
    out = loss_fn(batch, make_pred(batch, offset))
    batch_ref = make_batch([make_config(), make_config()])
    out_ref = loss_fn(batch_ref, make_pred(batch_ref, offset))
    assert torch.allclose(out, 2.0 * out_ref)  # mean weight = 2


def test_legacy_batch_without_attribute():
    """Batches built without the field (legacy caches) must still work."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    batch = make_batch([make_config()])
    del batch.forces_atomic_weights
    offset = 0.1 * torch.ones_like(batch.forces)
    out = loss_fn(batch, make_pred(batch, offset))
    assert torch.isfinite(out)


def test_xyz_column_roundtrip():
    """The default arrays column is picked up via KeySpecification."""
    atoms = Atoms("OH2", positions=[[0, -2, 0], [1, 0, 0], [0, 1, 0]])
    atoms.info["REF_energy"] = -1.5
    atoms.arrays["REF_forces"] = np.zeros((3, 3))
    atoms.arrays["REF_forces_atomic_weights"] = np.array([5.0, 1.0, 1.0])
    keyspec = KeySpecification.from_defaults()
    config = config_from_atoms(atoms, key_specification=keyspec)
    assert np.allclose(
        config.properties["forces_atomic_weights"], np.array([5.0, 1.0, 1.0])
    )
    atomic_data = AtomicData.from_config(config, z_table=TABLE, cutoff=CUTOFF)
    assert atomic_data.forces_atomic_weights.shape == (3, 1)
    assert torch.allclose(
        atomic_data.forces_atomic_weights.view(-1), torch.tensor([5.0, 1.0, 1.0])
    )


def test_ddp_reduction_consistency():
    """reduce path: ddp=False and explicit mean agree for weighted loss."""
    loss_fn = WeightedForcesLoss(forces_weight=1.0)
    batch = make_batch([make_config([2.0, 0.5, 1.0])])
    offset = 0.3 * torch.ones_like(batch.forces)
    pred = make_pred(batch, offset)
    out = loss_fn(batch, pred, ddp=False)
    expected = (
        batch.forces_atomic_weights * torch.square(batch.forces - pred["forces"])
    ).mean()
    assert torch.allclose(out, expected)