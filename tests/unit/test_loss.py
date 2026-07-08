"""Unit tests for mace/modules/loss.py.

Every expected value is computed BY HAND in the test (concrete numbers for
tiny batches), never by re-applying the module's formula. The canonical
batch is a single config with 2 atoms; multi-config cases use 2 configs.

The loss functions only touch a handful of Batch fields (via attribute or
item access): energy, forces, stress, virials, dipole, polarizability, ptr,
weight, energy_weight, forces_weight, stress_weight, virials_weight. A
minimal stand-in object providing both access styles is enough.
"""

import pytest
import torch

from mace.modules.loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
    conditional_huber_forces,
    conditional_mse_forces,
    mean_squared_error_energy,
    mean_squared_error_forces,
    reduce_loss,
    weighted_mean_squared_error_dipole,
    weighted_mean_squared_error_energy,
    weighted_mean_squared_stress,
    weighted_mean_squared_virials,
)


class FakeBatch:
    """Minimal stand-in for torch_geometric.Batch: the loss functions use
    both attribute access (ref.ptr, ref.weight) and item access
    (ref["energy"])."""

    def __init__(self, **fields):
        self._fields = dict(fields)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key):
        return self._fields[key]


def make_ref(num_atoms_per_graph=(2,), **overrides):
    """Build a reference batch of len(num_atoms_per_graph) configs with all
    fields zeroed / weights 1, then apply overrides."""
    n_graphs = len(num_atoms_per_graph)
    ptr = [0]
    for n in num_atoms_per_graph:
        ptr.append(ptr[-1] + n)
    total_atoms = ptr[-1]
    fields = {
        "ptr": torch.tensor(ptr, dtype=torch.long),
        "weight": torch.ones(n_graphs),
        "energy_weight": torch.ones(n_graphs),
        "forces_weight": torch.ones(n_graphs),
        "stress_weight": torch.ones(n_graphs),
        "virials_weight": torch.ones(n_graphs),
        "energy": torch.zeros(n_graphs),
        "forces": torch.zeros(total_atoms, 3),
        "stress": torch.zeros(n_graphs, 3, 3),
        "virials": torch.zeros(n_graphs, 3, 3),
        "dipole": torch.zeros(n_graphs, 3),
        "polarizability": torch.zeros(n_graphs, 3, 3),
    }
    fields.update(overrides)
    return FakeBatch(**fields)


def clone_pred(ref):
    """Prediction dict identical to the reference (loss must be 0)."""
    keys = ("energy", "forces", "stress", "virials", "dipole", "polarizability")
    return {k: ref[k].clone() for k in keys}


# ---------------------------------------------------------------------------
# reduce_loss
# ---------------------------------------------------------------------------


def test_reduce_loss_is_plain_mean_without_ddp():
    raw = torch.tensor([1.0, 2.0, 3.0])
    assert reduce_loss(raw, ddp=False).item() == pytest.approx(2.0)
    # ddp=None with torch.distributed not initialized also means plain mean
    assert reduce_loss(raw, ddp=None).item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# elementary loss functions
# ---------------------------------------------------------------------------


def test_mean_squared_error_energy():
    ref = make_ref(energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    assert mean_squared_error_energy(ref, pred).item() == pytest.approx(0.0)

    # 1 config: (10 - 13)^2 = 9 (no per-atom normalization in this variant)
    pred["energy"] = torch.tensor([13.0])
    assert mean_squared_error_energy(ref, pred).item() == pytest.approx(9.0)


def test_weighted_mean_squared_error_energy_per_atom_normalization():
    # 1 config, 2 atoms, energy deviation 2.0 -> ((2)/2)^2 = 1.0
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([12.0])
    assert weighted_mean_squared_error_energy(ref, pred).item() == pytest.approx(1.0)


def test_weighted_mean_squared_error_energy_config_weights():
    # 2 configs of 2 atoms, both with per-atom deviation 1.0 (raw = 1 each),
    # config weights [1, 3] -> raw = [1, 3] -> mean = 2
    ref = make_ref(
        num_atoms_per_graph=(2, 2),
        energy=torch.tensor([0.0, 0.0]),
        weight=torch.tensor([1.0, 3.0]),
    )
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([2.0, 2.0])
    assert weighted_mean_squared_error_energy(ref, pred).item() == pytest.approx(2.0)

    # per-config energy_weight scales the same way: [2, 2] doubles everything
    ref2 = make_ref(
        num_atoms_per_graph=(2, 2),
        energy=torch.tensor([0.0, 0.0]),
        weight=torch.tensor([1.0, 3.0]),
        energy_weight=torch.tensor([2.0, 2.0]),
    )
    assert weighted_mean_squared_error_energy(ref2, pred).item() == pytest.approx(4.0)


def test_mean_squared_error_forces():
    # 1 config, 2 atoms; deviation 1.0 in a single component:
    # squared errors = [1, 0, 0, 0, 0, 0] -> mean over 6 elements = 1/6
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    assert mean_squared_error_forces(ref, pred).item() == pytest.approx(0.0)

    pred["forces"] = torch.zeros(2, 3)
    pred["forces"][0, 0] = 1.0
    assert mean_squared_error_forces(ref, pred).item() == pytest.approx(1.0 / 6.0)

    # config weight 3 multiplies every per-atom contribution -> 3/6
    ref_w = make_ref(num_atoms_per_graph=(2,), weight=torch.tensor([3.0]))
    assert mean_squared_error_forces(ref_w, pred).item() == pytest.approx(3.0 / 6.0)

    # per-config forces_weight acts identically -> 3/6
    ref_fw = make_ref(num_atoms_per_graph=(2,), forces_weight=torch.tensor([3.0]))
    assert mean_squared_error_forces(ref_fw, pred).item() == pytest.approx(3.0 / 6.0)


def test_weighted_mean_squared_stress():
    # deviation 3.0 in one of 9 components: 9 / 9 = 1.0
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    pred["stress"] = torch.zeros(1, 3, 3)
    pred["stress"][0, 0, 0] = 3.0
    assert weighted_mean_squared_stress(ref, pred).item() == pytest.approx(1.0)

    # per-config stress_weight of 2 doubles the value
    ref_w = make_ref(num_atoms_per_graph=(2,), stress_weight=torch.tensor([2.0]))
    assert weighted_mean_squared_stress(ref_w, pred).item() == pytest.approx(2.0)


def test_weighted_mean_squared_virials_per_atom_normalization():
    # 2 atoms, deviation 4.0 in one component: (4/2)^2 = 4, mean over 9 = 4/9
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    pred["virials"] = torch.zeros(1, 3, 3)
    pred["virials"][0, 1, 1] = 4.0
    assert weighted_mean_squared_virials(ref, pred).item() == pytest.approx(4.0 / 9.0)


def test_weighted_mean_squared_error_dipole():
    # 2 atoms, deviation 2.0 in x: (2/2)^2 = 1, mean over 3 components = 1/3
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    pred["dipole"] = torch.tensor([[2.0, 0.0, 0.0]])
    assert weighted_mean_squared_error_dipole(ref, pred).item() == pytest.approx(
        1.0 / 3.0
    )


def test_conditional_mse_forces_regimes():
    # atom 0: |F_ref| = 150 -> regime 2, factor 0.7; error 1 -> se = 0.7
    # atom 1: |F_ref| = 0   -> regime 1, factor 1.0; error 2 -> se = 4.0
    # mean over 6 elements = 4.7 / 6
    ref = make_ref(
        num_atoms_per_graph=(2,),
        forces=torch.tensor([[150.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    pred = clone_pred(ref)
    pred["forces"][0, 0] += 1.0
    pred["forces"][1, 1] += 2.0
    assert conditional_mse_forces(ref, pred).item() == pytest.approx(4.7 / 6.0)


def test_conditional_huber_forces():
    # single atom with |F_ref| = 150 -> regime 2 -> delta = 0.7 * 1.0 = 0.7
    # error 2.0 in x is in the linear regime: 0.7 * (2 - 0.7/2) = 1.155
    # mean over 3 components = 1.155 / 3 = 0.385
    ref_forces = torch.tensor([[150.0, 0.0, 0.0]])
    pred_forces = torch.tensor([[152.0, 0.0, 0.0]])
    out = conditional_huber_forces(ref_forces, pred_forces, huber_delta=1.0, ddp=False)
    assert out.item() == pytest.approx(0.385)

    # identical forces -> 0
    out0 = conditional_huber_forces(ref_forces, ref_forces.clone(), huber_delta=1.0)
    assert out0.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# WeightedEnergyForcesLoss
# ---------------------------------------------------------------------------


def test_weighted_energy_forces_loss_zero():
    ref = make_ref(energy=torch.tensor([-7.5]))
    pred = clone_pred(ref)
    loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)


def test_weighted_energy_forces_loss_hand_value():
    # energy: dev 2.0 over 2 atoms -> 1.0; forces: dev 1.0 in one of 6 -> 1/6
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([12.0])
    pred["forces"][0, 0] = 1.0
    loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(1.0 + 1.0 / 6.0)


def test_weighted_energy_forces_loss_global_weights_scale():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([12.0])
    pred["forces"][0, 0] = 1.0
    loss = WeightedEnergyForcesLoss(energy_weight=2.0, forces_weight=12.0)
    # 2 * 1.0 + 12 * (1/6) = 4.0
    assert loss(ref, pred).item() == pytest.approx(4.0)


def test_weighted_energy_forces_loss_config_weight_scales_both_terms():
    ref = make_ref(
        num_atoms_per_graph=(2,),
        energy=torch.tensor([10.0]),
        weight=torch.tensor([3.0]),
    )
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([12.0])
    pred["forces"][0, 0] = 1.0
    loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    # 3 * 1.0 + 3 * (1/6) = 3.5
    assert loss(ref, pred).item() == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# WeightedForcesLoss
# ---------------------------------------------------------------------------


def test_weighted_forces_loss():
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    loss = WeightedForcesLoss(forces_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)

    pred["forces"][0, 0] = 1.0
    assert loss(ref, pred).item() == pytest.approx(1.0 / 6.0)
    # global forces_weight scaling
    loss6 = WeightedForcesLoss(forces_weight=6.0)
    assert loss6(ref, pred).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# WeightedEnergyForcesStressLoss
# ---------------------------------------------------------------------------


def test_weighted_energy_forces_stress_loss_zero_and_hand_value():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([1.0]))
    pred = clone_pred(ref)
    loss = WeightedEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=1.0, stress_weight=1.0
    )
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # energy: dev 2 over 2 atoms -> 1.0
    # forces: dev 1 in one of 6 -> 1/6
    # stress: dev 3 in one of 9 -> 9/9 = 1.0
    pred["energy"] = torch.tensor([3.0])
    pred["forces"][1, 2] = 1.0
    pred["stress"][0, 0, 0] = 3.0
    assert loss(ref, pred).item() == pytest.approx(1.0 + 1.0 / 6.0 + 1.0)

    loss_w = WeightedEnergyForcesStressLoss(
        energy_weight=10.0, forces_weight=6.0, stress_weight=0.5
    )
    assert loss_w(ref, pred).item() == pytest.approx(10.0 + 1.0 + 0.5)


# ---------------------------------------------------------------------------
# WeightedEnergyForcesVirialsLoss
# ---------------------------------------------------------------------------


def test_weighted_energy_forces_virials_loss():
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    loss = WeightedEnergyForcesVirialsLoss(
        energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    )
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # virials only: dev 4 in one comp, 2 atoms -> (4/2)^2 / 9 = 4/9
    pred["virials"][0, 2, 2] = 4.0
    assert loss(ref, pred).item() == pytest.approx(4.0 / 9.0)
    # virials_weight = 9 -> 4.0
    loss9 = WeightedEnergyForcesVirialsLoss(virials_weight=9.0)
    assert loss9(ref, pred).item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# WeightedHuberEnergyForcesStressLoss
# ---------------------------------------------------------------------------


def test_weighted_huber_energy_forces_stress_loss():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    loss = WeightedHuberEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=1.0
    )
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # energy: per-atom diff (11-10)/2 = 0.5 <= delta -> 0.5 * 0.5^2 = 0.125
    # forces: diff 2.0 in one of 6 elements, |x| > delta:
    #         1 * (2 - 0.5) = 1.5 -> mean = 1.5/6 = 0.25
    # stress: unchanged -> 0
    pred["energy"] = torch.tensor([11.0])
    pred["forces"][0, 1] = 2.0
    assert loss(ref, pred).item() == pytest.approx(0.125 + 0.25)

    # global weights scale each term independently
    loss_w = WeightedHuberEnergyForcesStressLoss(
        energy_weight=8.0, forces_weight=4.0, stress_weight=1.0, huber_delta=1.0
    )
    assert loss_w(ref, pred).item() == pytest.approx(8.0 * 0.125 + 4.0 * 0.25)


# ---------------------------------------------------------------------------
# UniversalLoss
# ---------------------------------------------------------------------------


def test_universal_loss():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([5.0]))
    pred = clone_pred(ref)
    loss = UniversalLoss(
        energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=1.0
    )
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # only forces deviate: |F_ref| = 0 < 100 -> factor 1.0 -> delta 1.0
    # diff 0.5 (quadratic regime): 0.5 * 0.25 = 0.125, mean over 6 = 0.125/6
    pred["forces"][0, 0] = 0.5
    assert loss(ref, pred).item() == pytest.approx(0.125 / 6.0)

    loss_fw = UniversalLoss(forces_weight=6.0, huber_delta=1.0)
    assert loss_fw(ref, pred).item() == pytest.approx(6.0 * 0.125 / 6.0)


# ---------------------------------------------------------------------------
# Dipole / polarizability losses
# ---------------------------------------------------------------------------


def test_dipole_single_loss():
    ref = make_ref(num_atoms_per_graph=(2,), dipole=torch.tensor([[1.0, 0.0, 0.0]]))
    pred = clone_pred(ref)
    loss = DipoleSingleLoss(dipole_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # dev 2 in x over 2 atoms: (2/2)^2 = 1, mean over 3 = 1/3, x100 scale
    pred["dipole"] = torch.tensor([[3.0, 0.0, 0.0]])
    assert loss(ref, pred).item() == pytest.approx(100.0 / 3.0)
    loss3 = DipoleSingleLoss(dipole_weight=3.0)
    assert loss3(ref, pred).item() == pytest.approx(100.0)


def test_dipole_polar_loss():
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    loss = DipolePolarLoss(dipole_weight=1.0, polarizability_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # dipole: dev 2 in x -> (2/2)^2 / 3 = 1/3 (note: NO x100 here)
    # polarizability: dev 6 in one of 9 comps -> (6/2)^2 / 9 = 1.0
    pred["dipole"] = torch.tensor([[2.0, 0.0, 0.0]])
    pred["polarizability"][0, 0, 0] = 6.0
    assert loss(ref, pred).item() == pytest.approx(1.0 / 3.0 + 1.0)

    loss_w = DipolePolarLoss(dipole_weight=3.0, polarizability_weight=0.5)
    assert loss_w(ref, pred).item() == pytest.approx(1.0 + 0.5)


def test_weighted_energy_forces_dipole_loss():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([0.0]))
    pred = clone_pred(ref)
    loss = WeightedEnergyForcesDipoleLoss(
        energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0
    )
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # energy: dev 2 over 2 atoms -> 1.0
    # forces: dev 1 in one of 6 -> 1/6
    # dipole: dev 2 in x -> 1/3, x100 -> 100/3
    pred["energy"] = torch.tensor([2.0])
    pred["forces"][0, 0] = 1.0
    pred["dipole"] = torch.tensor([[2.0, 0.0, 0.0]])
    assert loss(ref, pred).item() == pytest.approx(1.0 + 1.0 / 6.0 + 100.0 / 3.0)

    loss_w = WeightedEnergyForcesDipoleLoss(
        energy_weight=2.0, forces_weight=6.0, dipole_weight=0.03
    )
    assert loss_w(ref, pred).item() == pytest.approx(2.0 + 1.0 + 1.0)


# ---------------------------------------------------------------------------
# WeightedEnergyForcesL1L2Loss
# ---------------------------------------------------------------------------


def test_weighted_energy_forces_l1l2_loss():
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([1.0]))
    pred = clone_pred(ref)
    loss = WeightedEnergyForcesL1L2Loss(energy_weight=1.0, forces_weight=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)

    # energy: |dev 3| / 2 atoms = 1.5 (L1)
    # forces: atom 0 error vector (3, 4, 0) -> norm 5; atom 1 -> 0; mean 2.5
    pred["energy"] = torch.tensor([4.0])
    pred["forces"][0] = torch.tensor([3.0, 4.0, 0.0])
    assert loss(ref, pred).item() == pytest.approx(1.5 + 2.5)

    loss_w = WeightedEnergyForcesL1L2Loss(energy_weight=2.0, forces_weight=0.4)
    assert loss_w(ref, pred).item() == pytest.approx(3.0 + 1.0)


# ---------------------------------------------------------------------------
# repr smoke: weights render in __repr__ for logging
# ---------------------------------------------------------------------------


def test_loss_repr_contains_weights():
    rep = repr(WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=100.0))
    assert "energy_weight=1.000" in rep
    assert "forces_weight=100.000" in rep
