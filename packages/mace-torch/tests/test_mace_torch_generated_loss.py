"""The generated loss, against the values the frozen tree's classes are pinned to.

Every number here is one of the hand-computed values in
``tests/unit/test_loss.py``, which pins the ten legacy loss classes by
arithmetic rather than by re-applying their formulas. Reaching the same numbers
from a declaration instead of from a class is the whole claim of this layer, so
they are quoted rather than recomputed.

The canonical case is one structure of two atoms: an energy off by 2.0 gives
``(2/2)^2 = 1.0`` because an energy is extensive and compared per atom, and one
force component off by 1.0 gives ``1/6`` because the mean is over all six.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.config.loss import LossConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import load_default_catalogue, resolve_requested
from mace_core.outputs import MACEOutput
from mace_torch.data import GraphDataset, collate_training, target_specs
from mace_torch.train import GeneratedLoss, build_loss, register_loss, terms_for

CATALOGUE = load_default_catalogue()
REQUESTED = resolve_requested(["energy", "forces"], CATALOGUE)
SPECS = target_specs(REQUESTED)
Z_TABLE = AtomicNumberTable([1])

#: Two atoms, far enough apart that no edge exists. No model runs here: the
#: predictions are handed in directly, so the loss is the only thing measured.
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def structure(energy: float = 10.0, weight: float = 1.0, **properties):
    values = {"energy": energy, "forces": np.zeros((2, 3)), **properties}
    return Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=POSITIONS,
        properties=values,
        weight=weight,
    )


def batch_of(*configurations):
    dataset = GraphDataset(
        list(configurations), cutoff=0.5, z_table=Z_TABLE, targets=SPECS
    )
    return collate_training(
        [dataset[i] for i in range(len(configurations))], z_table=Z_TABLE
    )


def prediction(batch, *, energy_off: float = 0.0, force_off: float = 0.0):
    """The reference values, perturbed by the given amounts."""
    energies = batch.targets["energy"].clone()
    energies[0] = energies[0] + energy_off
    forces = batch.targets["forces"].clone()
    forces[0, 0] = forces[0, 0] + force_off
    return MACEOutput(total_energy=energies, forces=forces)


def loss_of(batch, output, **weights):
    config = LossConfig(weights=weights)
    return float(build_loss(REQUESTED, config)(output, batch))


# ---------------------------------------------------------------------------
# The declared shape of a term
# ---------------------------------------------------------------------------


def test_a_term_reads_its_shape_off_the_declaration():
    """Nothing here is a table of which names are which."""
    terms = {term.name: term for term in terms_for(REQUESTED, LossConfig())}
    assert terms["energy"].extensive and not terms["energy"].per_atom
    assert terms["forces"].per_atom and not terms["forces"].extensive


def test_a_loss_with_no_terms_is_refused():
    with pytest.raises(ValueError, match="scores nothing"):
        GeneratedLoss([])


# ---------------------------------------------------------------------------
# The hand-computed values
# ---------------------------------------------------------------------------


@fp64_only
def test_a_perfect_prediction_scores_zero():
    batch = batch_of(structure())
    assert loss_of(batch, prediction(batch), energy=1.0, forces=1.0) == 0.0


@fp64_only
def test_the_canonical_energy_and_force_value():
    """`1.0 + 1/6`, from `test_weighted_energy_forces_loss_hand_value`."""
    batch = batch_of(structure())
    output = prediction(batch, energy_off=2.0, force_off=1.0)
    assert loss_of(batch, output, energy=1.0, forces=1.0) == pytest.approx(
        1.0 + 1.0 / 6.0
    )


@fp64_only
def test_the_weights_scale_their_own_term():
    """`2 * 1.0 + 12 * (1/6) = 4.0`, from the global-weights test."""
    batch = batch_of(structure())
    output = prediction(batch, energy_off=2.0, force_off=1.0)
    assert loss_of(batch, output, energy=2.0, forces=12.0) == pytest.approx(4.0)


@fp64_only
def test_a_structure_weight_scales_every_term_of_that_structure():
    """`3 * 1.0 + 3 * (1/6) = 3.5`, from the config-weight test."""
    batch = batch_of(structure(weight=3.0))
    output = prediction(batch, energy_off=2.0, force_off=1.0)
    assert loss_of(batch, output, energy=1.0, forces=1.0) == pytest.approx(3.5)


@fp64_only
def test_the_energy_is_compared_per_atom():
    """The same absolute error over twice the atoms is a quarter of the term.

    This is what `extensive` buys, and it is the difference between a fit that
    treats a large structure as one example and one that treats it as many.
    """
    small = batch_of(structure())
    large = batch_of(
        Configuration(
            atomic_numbers=np.array([1, 1, 1, 1]),
            positions=np.repeat(POSITIONS, 2, axis=0),
            properties={"energy": 10.0, "forces": np.zeros((4, 3))},
        )
    )
    on_small = loss_of(small, prediction(small, energy_off=2.0), energy=1.0, forces=0.0)
    on_large = loss_of(large, prediction(large, energy_off=2.0), energy=1.0, forces=0.0)
    assert on_small == pytest.approx(1.0)
    assert on_large == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@fp64_only
def test_a_structure_without_a_property_contributes_nothing_to_that_term():
    """Not a NaN, and not a zero that a loss had to remember to insert."""
    labelled = structure()
    unlabelled = Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=POSITIONS,
        properties={"energy": 10.0},
    )
    batch = batch_of(labelled, unlabelled)
    assert batch.property_weights["forces"].tolist() == [1.0, 0.0]
    output = MACEOutput(
        total_energy=batch.targets["energy"].clone(),
        forces=batch.targets["forces"] + 1.0,
    )
    # Six of the twelve force rows are masked, and the mean is over all twelve.
    assert loss_of(batch, output, energy=1.0, forces=1.0) == pytest.approx(0.5)


@fp64_only
def test_a_property_weight_scales_one_structure_s_term():
    batch = batch_of(
        Configuration(
            atomic_numbers=np.array([1, 1]),
            positions=POSITIONS,
            properties={"energy": 10.0, "forces": np.zeros((2, 3))},
            property_weights={"energy": 0.5},
        )
    )
    output = prediction(batch, energy_off=2.0)
    assert loss_of(batch, output, energy=1.0, forces=0.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@pytest.fixture(name="clean_registry")
def fixture_clean_registry():
    """Registration is a global side effect, so a test that does it undoes it.

    Without this the second run of a parametrized test meets its own first
    registration and reads as a duplicate.
    """
    from mace_torch.train.loss import LOSS_REGISTRY

    before = dict(LOSS_REGISTRY)
    yield LOSS_REGISTRY
    LOSS_REGISTRY.clear()
    LOSS_REGISTRY.update(before)


def test_a_registered_loss_is_selectable_without_editing_the_registry(clean_registry):
    @register_loss("probe_loss_for_the_test")
    class Probe(torch.nn.Module):
        def forward(self, output, batch):
            return torch.zeros(())

    assert clean_registry["probe_loss_for_the_test"] is Probe


def test_registering_one_name_twice_is_refused(clean_registry):
    """Two under one name means the run scores against whichever module was
    imported last, which depends on import order and nothing else."""

    @register_loss("probe_duplicate")
    class First(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="already a registered loss"):

        @register_loss("probe_duplicate")
        class Second(torch.nn.Module):
            pass
