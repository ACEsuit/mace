"""Unit tests for mace/modules/loss.py.

Every expected value is computed BY HAND in the test (concrete numbers for
tiny batches), never by re-applying the module's formula. The canonical
batch is a single config with 2 atoms; multi-config cases use 2 configs.

The loss functions only touch a handful of Batch fields (via attribute or
item access): energy, forces, stress, virials, dipole, polarizability, ptr,
weight, energy_weight, forces_weight, stress_weight, virials_weight. A
minimal stand-in object providing both access styles is enough.
"""

import argparse

import pytest
import torch
import torch.distributed as dist

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
    is_ddp_enabled,
    mean_normed_error_forces,
    mean_squared_error_energy,
    mean_squared_error_forces,
    reduce_loss,
    weighted_mean_absolute_error_energy,
    weighted_mean_squared_error_dipole,
    weighted_mean_squared_error_energy,
    weighted_mean_squared_error_polarizability,
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
        "magforces_weight": torch.ones(n_graphs),
        "stress_weight": torch.ones(n_graphs),
        "virials_weight": torch.ones(n_graphs),
        "energy": torch.zeros(n_graphs),
        "forces": torch.zeros(total_atoms, 3),
        "stress": torch.zeros(n_graphs, 3, 3),
        "virials": torch.zeros(n_graphs, 3, 3),
        "dipole": torch.zeros(n_graphs, 3),
        "polarizability": torch.zeros(n_graphs, 3, 3),
        "magforces": torch.zeros(total_atoms, 3),
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


# ---------------------------------------------------------------------------
# UniversalLoss: the magforces term
#
# This is the branch that carries most of the file's uncovered arithmetic. It
# is guarded three ways -- the key has to be in `pred`, and neither
# `pred["magforces"]` nor `ref["magforces"]` may be None -- so "the model does
# not predict magnetic forces" is expressed as a silently dropped term rather
# than as an error. All three guards are pinned below, because a port that
# turns one of them into a KeyError changes what a non-magnetic training run
# does.
# ---------------------------------------------------------------------------


def test_universal_loss_magforces_hand_value():
    # 1 config, 2 atoms, huber_delta = 1.0, every weight 1.
    # magforces: one component off by 0.5, |0.5| <= delta -> 0.5 * 0.5^2 =
    # 0.125, meaned over the 6 elements -> 0.125 / 6.
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    pred["magforces"] = torch.zeros(2, 3)
    pred["magforces"][1, 2] = 0.5
    loss = UniversalLoss(huber_delta=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.125 / 6.0)

    # the global magforces_weight is a plain linear factor on that term
    loss_w = UniversalLoss(huber_delta=1.0, magforces_weight=6.0)
    assert loss_w(ref, pred).item() == pytest.approx(0.125)


def test_universal_loss_magforces_per_config_weight_multiplies_the_arguments():
    """Not the term: the *inputs* of the huber, which is not the same thing.

    `configs_magforces_weight` multiplies ref and pred before the huber
    (mace/modules/loss.py:486-491), so the error it sees is scaled and the
    regime it lands in can change. With weight 2 the 0.5 deviation above
    becomes 1.0, which is exactly at delta: 0.5 * 1.0^2 = 0.5, meaned over 6
    -> 0.5 / 6, i.e. four times the unweighted value, not twice.
    """
    ref = make_ref(num_atoms_per_graph=(2,), magforces_weight=torch.tensor([2.0]))
    pred = clone_pred(ref)
    pred["magforces"] = torch.zeros(2, 3)
    pred["magforces"][1, 2] = 0.5
    loss = UniversalLoss(huber_delta=1.0)
    assert loss(ref, pred).item() == pytest.approx(0.5 / 6.0)


@pytest.mark.parametrize(
    "make_pred_entry",
    [
        pytest.param(lambda: None, id="pred_is_none"),
        pytest.param(lambda: "absent", id="key_absent"),
    ],
)
def test_universal_loss_drops_the_magforces_term_when_it_is_not_predicted(
    make_pred_entry,
):
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([1.0]))
    pred = clone_pred(ref)
    entry = make_pred_entry()
    if entry != "absent":
        pred["magforces"] = entry
    loss = UniversalLoss(huber_delta=1.0, magforces_weight=1000.0)
    # nothing deviates and the magforces term is skipped: exactly zero, and
    # in particular not a TypeError from huber_loss(None, ...).
    assert loss(ref, pred).item() == pytest.approx(0.0)


def test_universal_loss_drops_the_magforces_term_when_the_reference_is_none():
    ref = make_ref(num_atoms_per_graph=(2,), magforces=None)
    pred = clone_pred(ref)
    pred["magforces"] = torch.ones(2, 3)
    loss = UniversalLoss(huber_delta=1.0, magforces_weight=1000.0)
    assert loss(ref, pred).item() == pytest.approx(0.0)


def test_universal_loss_full_hand_value_over_all_four_terms():
    # 1 config, 2 atoms, huber_delta = 1.0, all per-config weights 1.
    #   energy:     (12 - 10) / 2 = 1.0 = delta -> 0.5 * 1^2       = 0.5
    #   forces:     |F_ref| = 0 -> factor 1.0 -> delta 1.0; error 2.0 is in
    #               the linear regime: 1 * (2 - 0.5) = 1.5, / 6     = 0.25
    #   stress:     error 3.0, linear: 1 * (3 - 0.5) = 2.5, / 9     = 2.5/9
    #   magforces:  error 0.5, quadratic: 0.5 * 0.25 = 0.125, / 6   = 0.125/6
    ref = make_ref(num_atoms_per_graph=(2,), energy=torch.tensor([10.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([12.0])
    pred["forces"][1, 2] = 2.0
    pred["stress"][0, 0, 0] = 3.0
    pred["magforces"] = torch.zeros(2, 3)
    pred["magforces"][0, 1] = 0.5
    expected = 0.5 + 0.25 + 2.5 / 9.0 + 0.125 / 6.0
    loss = UniversalLoss(huber_delta=1.0)
    assert loss(ref, pred).item() == pytest.approx(expected)

    # the four global weights are a plain linear combination of those terms
    loss_w = UniversalLoss(
        energy_weight=2.0,
        forces_weight=4.0,
        stress_weight=9.0,
        magforces_weight=6.0,
        huber_delta=1.0,
    )
    assert loss_w(ref, pred).item() == pytest.approx(
        2.0 * 0.5 + 4.0 * 0.25 + 9.0 * (2.5 / 9.0) + 6.0 * (0.125 / 6.0)
    )


def test_universal_loss_per_config_energy_weight_is_not_a_linear_factor():
    """Doubling it tripled this loss. Measured, and pinned so a port keeps it.

    `configs_energy_weight` multiplies both sides *inside* the huber
    (mace/modules/loss.py:464-469), so it rescales the error and can push it
    from the quadratic branch into the linear one. Deviation 2.0 over 2
    atoms: at weight 1 the argument is 1.0 = delta -> 0.5; at weight 2 it is
    2.0 -> 1 * (2 - 0.5) = 1.5.
    """
    pred_energy = torch.tensor([2.0])
    plain = make_ref(num_atoms_per_graph=(2,))
    doubled = make_ref(
        num_atoms_per_graph=(2,), energy_weight=torch.tensor([2.0])
    )
    loss = UniversalLoss(huber_delta=1.0)

    pred = clone_pred(plain)
    pred["energy"] = pred_energy
    assert loss(plain, pred).item() == pytest.approx(0.5)
    assert loss(doubled, pred).item() == pytest.approx(1.5)


def test_universal_loss_per_config_forces_weight_can_change_the_huber_regime():
    """The same rescaling, but on the *regime selector* as well.

    `conditional_huber_forces` picks its delta from `torch.norm(ref_forces)`
    -- and what it is handed is the already-weighted reference. A 60 eV/Ang
    reference force sits in regime 1 (delta = 1.0 * huber_delta); at
    forces_weight 2 it is 120 and lands in regime 2 (delta = 0.7 *
    huber_delta), so the loss changes by 2.31x rather than by 2x or 4x.
    """
    base_forces = torch.tensor([[60.0, 0.0, 0.0]])
    plain = make_ref(num_atoms_per_graph=(1,), forces=base_forces)
    doubled = make_ref(
        num_atoms_per_graph=(1,),
        forces=base_forces,
        forces_weight=torch.tensor([2.0]),
    )
    pred = clone_pred(plain)
    pred["forces"] = torch.tensor([[61.0, 0.0, 0.0]])
    loss = UniversalLoss(huber_delta=1.0)

    # weight 1: |F| = 60 -> delta 1.0, error 1.0 -> 0.5 * 1^2 = 0.5, / 3
    assert loss(plain, pred).item() == pytest.approx(1.0 / 6.0)
    # weight 2: |F| = 120 -> delta 0.7, error 2.0 -> 0.7 * (2 - 0.35) = 1.155
    assert loss(doubled, pred).item() == pytest.approx(1.155 / 3.0)


# ---------------------------------------------------------------------------
# The ddp=True branches
#
# Two loss modules keep a whole second copy of their arithmetic under
# `if ddp:` -- reduction="none" followed by reduce_loss instead of
# reduction="mean". With no process group initialised, reduce_loss falls
# through to a plain mean, so the two branches must agree exactly. That
# equality is the only thing that makes the single-process test suite say
# anything at all about the distributed path.
# ---------------------------------------------------------------------------


def _deviating_pair():
    ref = make_ref(num_atoms_per_graph=(2, 3), energy=torch.tensor([10.0, -4.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([11.0, -6.5])
    pred["forces"][0, 0] = 2.0
    pred["forces"][3, 1] = -0.25
    pred["stress"][0, 0, 0] = 3.0
    pred["stress"][1, 1, 2] = -0.5
    pred["magforces"] = torch.zeros(5, 3)
    pred["magforces"][2, 1] = 0.75
    return ref, pred


@pytest.mark.parametrize(
    "loss_fn",
    [
        WeightedHuberEnergyForcesStressLoss(huber_delta=1.0),
        UniversalLoss(huber_delta=1.0),
    ],
    ids=["huber", "universal"],
)
def test_the_ddp_branch_is_the_same_arithmetic_without_a_process_group(loss_fn):
    ref, pred = _deviating_pair()
    assert loss_fn(ref, pred, ddp=True).item() == loss_fn(ref, pred, ddp=False).item()


def test_is_ddp_enabled_is_false_without_a_process_group():
    assert is_ddp_enabled() is False


def test_reduce_loss_ddp_formula_in_a_world_of_one(tmp_path):
    """The one path a single-process test can still reach exactly.

    `reduce_loss` under ddp does not take a mean: it computes
    `local_sum * world_size / global_num_elements`, which is the mean only
    because every rank contributes its own element count to the all_reduce.
    With a world of one that reduces to the plain mean, which is what makes
    it checkable here -- and the formula, not the mean, is what a port has to
    reproduce, because the two disagree the moment ranks hold different
    numbers of atoms.
    """
    store = tmp_path / "gloo_store"
    dist.init_process_group(
        backend="gloo", init_method=f"file://{store}", rank=0, world_size=1
    )
    try:
        raw = torch.tensor([1.0, 2.0, 6.0])
        assert dist.get_world_size() == 1
        # world_size 1 means is_ddp_enabled() stays False even here: the
        # helper asks for > 1, so a one-rank run takes the plain-mean path
        # unless a caller passes ddp=True explicitly.
        assert is_ddp_enabled() is False
        assert reduce_loss(raw, ddp=True).item() == pytest.approx(3.0)
        assert reduce_loss(raw, ddp=None).item() == pytest.approx(3.0)
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# The remaining elementary functions
# ---------------------------------------------------------------------------


def test_weighted_mean_absolute_error_energy():
    # 2 configs of 2 atoms; deviations 3.0 and -1.0 -> |3/2| and |-1/2|
    # weights [1, 3] -> raw = [1.5, 1.5] -> mean = 1.5
    ref = make_ref(num_atoms_per_graph=(2, 2), weight=torch.tensor([1.0, 3.0]))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([3.0, -1.0])
    assert weighted_mean_absolute_error_energy(ref, pred).item() == pytest.approx(1.5)


def test_mean_normed_error_forces_is_unweighted():
    """The L1L2 forces term ignores every weight, unlike every other one.

    `mean_normed_error_forces` takes the per-atom error norm and means it,
    with no `ref.weight` and no `ref.forces_weight` anywhere
    (mace/modules/loss.py:138-142). Pinned because it is the single
    exception, and a port that "regularises" it changes what
    `--loss l1l2energyforces` fits.
    """
    ref = make_ref(num_atoms_per_graph=(2,), weight=torch.tensor([7.0]))
    pred = clone_pred(ref)
    pred["forces"][0] = torch.tensor([3.0, 4.0, 0.0])  # norm 5
    pred["forces"][1] = torch.tensor([0.0, 0.0, 1.0])  # norm 1
    assert mean_normed_error_forces(ref, pred).item() == pytest.approx(3.0)


def test_weighted_mean_squared_error_polarizability_reshapes_only_the_reference():
    """An asymmetry worth pinning: `.view(-1, 3, 3)` is applied to `ref` only.

    A reference stored flat as [n_graphs, 9] is accepted and reshaped
    (mace/modules/loss.py:174); a prediction stored flat is not, and would
    broadcast into nonsense instead of failing. Both sides are pinned.
    """
    ref = make_ref(
        num_atoms_per_graph=(2,), polarizability=torch.zeros(1, 9)
    )
    pred = clone_pred(ref)
    pred["polarizability"] = torch.zeros(1, 3, 3)
    pred["polarizability"][0, 2, 0] = 6.0
    # (6 / 2)^2 = 9 in one of nine components -> 1.0
    assert weighted_mean_squared_error_polarizability(
        ref, pred
    ).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Every class renders its weights, and every CLI name reaches a class
# ---------------------------------------------------------------------------


ALL_LOSS_CLASSES = (
    (WeightedEnergyForcesLoss, ("energy_weight", "forces_weight")),
    (WeightedForcesLoss, ("forces_weight",)),
    (WeightedEnergyForcesStressLoss, ("energy_weight", "forces_weight", "stress_weight")),
    (
        WeightedHuberEnergyForcesStressLoss,
        ("energy_weight", "forces_weight", "stress_weight"),
    ),
    (
        UniversalLoss,
        ("energy_weight", "forces_weight", "stress_weight", "magforces_weight"),
    ),
    (
        WeightedEnergyForcesVirialsLoss,
        ("energy_weight", "forces_weight", "virials_weight"),
    ),
    (DipoleSingleLoss, ("dipole_weight",)),
    (DipolePolarLoss, ("dipole_weight", "polarizability_weight")),
    (
        WeightedEnergyForcesDipoleLoss,
        ("energy_weight", "forces_weight", "dipole_weight"),
    ),
    (WeightedEnergyForcesL1L2Loss, ("energy_weight", "forces_weight")),
)


@pytest.mark.parametrize(
    "cls,weight_names", ALL_LOSS_CLASSES, ids=lambda v: getattr(v, "__name__", "")
)
def test_every_loss_class_renders_its_weights(cls, weight_names):
    """`__repr__` is what the training log records the loss as.

    It is the only record of which weights a finished run used, so every
    class has to name all of its own, at three decimals.
    """
    loss = cls(**{name: 2.5 for name in weight_names})
    rep = repr(loss)
    assert rep.startswith(cls.__name__ + "(")
    for name in weight_names:
        assert f"{name}=2.500" in rep, rep


LOSS_CLI_NAMES = {
    "weighted": WeightedEnergyForcesLoss,
    "forces_only": WeightedForcesLoss,
    "virials": WeightedEnergyForcesVirialsLoss,
    "stress": WeightedEnergyForcesStressLoss,
    "huber": WeightedHuberEnergyForcesStressLoss,
    "universal": UniversalLoss,
    "l1l2energyforces": WeightedEnergyForcesL1L2Loss,
    "dipole": DipoleSingleLoss,
    "dipole_polar": DipolePolarLoss,
    "energy_forces_dipole": WeightedEnergyForcesDipoleLoss,
}


def _loss_args(name):
    return argparse.Namespace(
        loss=name,
        energy_weight=2.0,
        forces_weight=3.0,
        stress_weight=4.0,
        virials_weight=5.0,
        dipole_weight=6.0,
        polarizability_weight=7.0,
        magforces_weight=8.0,
        huber_delta=0.5,
    )


@pytest.mark.parametrize("name,cls", sorted(LOSS_CLI_NAMES.items()))
def test_every_cli_loss_name_reaches_its_class_with_its_weights(name, cls):
    """`--loss <name>` is the only way a user selects any of this.

    The mapping lives in `get_loss_fn` and is otherwise untested; a rename
    there is silent, because the `else` branch hands back a default
    WeightedEnergyForcesLoss rather than refusing.
    """
    from mace.tools.scripts_utils import get_loss_fn  # noqa: PLC0415

    args = _loss_args(name)
    loss = get_loss_fn(
        args, dipole_only=(name == "dipole"), compute_dipole=("dipole" in name)
    )
    assert isinstance(loss, cls)
    for attr, value in (
        ("energy_weight", 2.0),
        ("forces_weight", 3.0),
        ("stress_weight", 4.0),
        ("virials_weight", 5.0),
        ("dipole_weight", 6.0),
        ("polarizability_weight", 7.0),
        ("magforces_weight", 8.0),
    ):
        if hasattr(loss, attr):
            assert float(getattr(loss, attr)) == value
    if hasattr(loss, "huber_delta"):
        assert loss.huber_delta == 0.5


def test_an_unknown_cli_loss_name_falls_back_instead_of_failing():
    """Characterization, not endorsement: a typo silently trains `weighted`."""
    from mace.tools.scripts_utils import get_loss_fn  # noqa: PLC0415

    loss = get_loss_fn(
        _loss_args("universl"), dipole_only=False, compute_dipole=False
    )
    assert isinstance(loss, WeightedEnergyForcesLoss)
    # and with the *default* weights, not the ones on the command line
    assert float(loss.energy_weight) == 1.0
    assert float(loss.forces_weight) == 1.0


def test_a_zero_config_weight_dilutes_rather_than_renormalizing():
    """Masking semantics: weight 0 removes a config's contribution, but the
    mean keeps its elements in the denominator.

    So the remaining loss is *halved*, not preserved. Stated as its own test
    because the plausible-looking alternative -- renormalising over the
    non-zero weights -- differs by exactly the factor a rewrite would not
    notice: every individual term is still right, and only the total moves.
    """
    ref = make_ref(num_atoms_per_graph=(1, 1), energy=torch.zeros(2))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([1.0, 1.0])
    both = weighted_mean_squared_error_energy(ref, pred).item()

    ref.weight = torch.tensor([1.0, 0.0])
    masked = weighted_mean_squared_error_energy(ref, pred).item()
    assert masked == pytest.approx(both / 2)


def test_the_weighted_huber_loss_ignores_config_and_property_weights():
    """Unlike every other class here, this one drops the per-config weights.

    Pinned because it is surprising: the constructor takes global weights and
    honours them, so the natural assumption is that the per-config ones apply
    too. They do not, and a test that only ever passes weights of 1.0 -- as
    the hand-value test above does -- cannot tell the difference.
    """
    ref = make_ref(num_atoms_per_graph=(2,))
    pred = clone_pred(ref)
    pred["energy"] = torch.tensor([0.004])
    pred["forces"][0, 0] = 0.002
    pred["forces"][1, 0] = 0.002
    pred["stress"][0, 0, 0] = 0.001

    loss = WeightedHuberEnergyForcesStressLoss(
        energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    )
    unweighted = loss(ref, pred).item()

    for field in ("weight", "energy_weight", "forces_weight", "stress_weight"):
        setattr(ref, field, torch.tensor([9.0]))
    assert loss(ref, pred).item() == pytest.approx(unweighted)
