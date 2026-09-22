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
from mace_core.config.loss import HuberLoss as HuberLossConfig
from mace_core.config.loss import L1L2Loss as L1L2LossConfig
from mace_core.config.loss import LossConfig, RegisteredLoss
from mace_core.config.loss import UniversalLoss as UniversalLossConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import (
    ObservableCatalogue,
    load_default_catalogue,
    resolve_requested,
)
from mace_core.outputs import MACEOutput
from mace_torch.data import GraphDataset, collate_training, target_specs
from mace_torch.train import (
    GeneratedLoss,
    UnknownLossError,
    build_loss,
    register_loss,
    terms_for,
)

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


# ---------------------------------------------------------------------------
# Quantities the catalogue does not ship, declared here and nowhere else
# ---------------------------------------------------------------------------

#: An energy whose strain derivative is called `virials` rather than `stress`,
#: plus a dipole. Neither needs a line of loss code: the declaration says the
#: shape and the extensivity, and the term follows. This is the claim the ten
#: legacy loss classes cannot make, and it is checked against their numbers.
EXTENDED = ObservableCatalogue(
    inputs=[
        {"name": "pos", "irreps": "1o", "per_atom": True, "units": "A"},
        {"name": "strain", "irreps": "0e+2e", "per_atom": False, "units": "1"},
    ],
    observables=[
        {
            "name": "energy",
            "irreps": "0e",
            "per_atom": False,
            "units": "eV",
            "extensive": True,
            "derivatives": [
                {"wrt": "pos", "name": "forces", "sign": -1, "units": "eV/A"},
                # A virial is extensive where a stress is not, which is the
                # whole reason extensivity is asked of the derivative.
                {
                    "wrt": "strain",
                    "name": "virials",
                    "sign": 1,
                    "units": "eV",
                    "extensive": True,
                },
            ],
        },
        {
            "name": "dipole",
            "irreps": "1o",
            "per_atom": False,
            "units": "eA",
            "extensive": True,
        },
    ],
)


def extended_batch(**properties):
    """One structure of two atoms carrying whatever the test declares."""
    values = {"energy": 10.0, "forces": np.zeros((2, 3)), **properties}
    configuration = Configuration(
        atomic_numbers=np.array([1, 1]), positions=POSITIONS, properties=values
    )
    requested = resolve_requested(list(values), EXTENDED)
    dataset = GraphDataset(
        [configuration],
        cutoff=0.5,
        z_table=Z_TABLE,
        targets=target_specs(requested),
    )
    return requested, collate_training([dataset[0]], z_table=Z_TABLE)


@fp64_only
def test_a_virial_is_compared_per_atom_and_a_stress_is_not():
    """`(4/2)^2 / 9 = 4/9`, from `test_weighted_mean_squared_virials`.

    The same derivative of the same energy against the same input, and the two
    differ only in a declaration.
    """
    requested, batch = extended_batch(virials=np.zeros((3, 3)))
    predicted = batch.targets["virials"].clone()
    predicted[0, 1, 1] = 4.0
    output = MACEOutput(
        total_energy=batch.targets["energy"].clone(),
        forces=batch.targets["forces"].clone(),
        virials=predicted,
    )
    loss = build_loss(requested, LossConfig(weights={"virials": 1.0}))
    assert float(loss(output, batch)) == pytest.approx(4.0 / 9.0)


@fp64_only
def test_the_virial_weight_scales_it():
    """`virials_weight = 9` gives `4.0`, from the same file."""
    requested, batch = extended_batch(virials=np.zeros((3, 3)))
    predicted = batch.targets["virials"].clone()
    predicted[0, 2, 2] = 4.0
    output = MACEOutput(
        total_energy=batch.targets["energy"].clone(),
        forces=batch.targets["forces"].clone(),
        virials=predicted,
    )
    loss = build_loss(requested, LossConfig(weights={"virials": 9.0}))
    assert float(loss(output, batch)) == pytest.approx(4.0)


@fp64_only
def test_a_dipole_is_compared_per_atom():
    """`(2/2)^2 / 3 = 1/3`, from `test_weighted_mean_squared_error_dipole`.

    Where this deliberately differs from the frozen tree is documented in the
    test below: there the dipole term ignores the structure's own weight and
    carries a hardcoded factor of one hundred.
    """
    requested, batch = extended_batch(dipole=np.zeros(3))
    predicted = batch.targets["dipole"].clone()
    predicted[0, 0] = 2.0
    output = MACEOutput(
        total_energy=batch.targets["energy"].clone(),
        forces=batch.targets["forces"].clone(),
        dipole=predicted,
    )
    loss = build_loss(requested, LossConfig(weights={"dipole": 1.0}))
    assert float(loss(output, batch)) == pytest.approx(1.0 / 3.0)


@fp64_only
def test_the_hundred_the_frozen_tree_hides_in_its_dipole_loss_is_a_weight():
    """`DipoleSingleLoss` multiplies by 100.0, commented `scale adjustment`.

    It is a weight written where a weight cannot be configured, so asking for
    one hundred reaches the same number and says so. The same term also ignores
    the structure's weight there, which is not reproduced: a weight that
    applies to every term except one is a rule nobody can state.
    """
    requested, batch = extended_batch(dipole=np.zeros(3))
    predicted = batch.targets["dipole"].clone()
    predicted[0, 0] = 2.0
    output = MACEOutput(
        total_energy=batch.targets["energy"].clone(),
        forces=batch.targets["forces"].clone(),
        dipole=predicted,
    )
    loss = build_loss(requested, LossConfig(weights={"dipole": 100.0}))
    assert float(loss(output, batch)) == pytest.approx(100.0 / 3.0)


@fp64_only
def test_a_declared_observable_needs_no_loss_code_at_all():
    """The acceptance criterion, as one assertion: a quantity this package
    ships no term for gets one from its declaration."""
    requested, _ = extended_batch(dipole=np.zeros(3), virials=np.zeros((3, 3)))
    names = {term.name for term in terms_for(requested, LossConfig())}
    assert {"energy", "forces", "dipole", "virials"} <= names


# ---------------------------------------------------------------------------
# The three legacy classes that are genuinely different reductions
# ---------------------------------------------------------------------------


def loss_with(kind, batch, output, **weights):
    config = LossConfig(kind=kind, weights=weights)
    return float(build_loss(REQUESTED, config)(output, batch))


@fp64_only
def test_the_huber_loss_is_quadratic_below_the_crossover_and_linear_above():
    """`0.125 + 0.25`, from `test_weighted_huber_energy_forces_stress_loss`.

    The energy is off by one over two atoms, so its normalised residual of 0.5
    is inside the crossover and costs half its square. A force is off by two,
    outside it, and costs the linear tail.
    """
    batch = batch_of(structure())
    output = prediction(batch, energy_off=1.0, force_off=2.0)
    assert loss_with(
        HuberLossConfig(delta=1.0), batch, output, energy=1.0, forces=1.0
    ) == pytest.approx(0.125 + 0.25)


@fp64_only
def test_the_huber_weights_scale_each_term_independently():
    """`8 * 0.125 + 4 * 0.25`, from the same test."""
    batch = batch_of(structure())
    output = prediction(batch, energy_off=1.0, force_off=2.0)
    assert loss_with(
        HuberLossConfig(delta=1.0), batch, output, energy=8.0, forces=4.0
    ) == pytest.approx(8.0 * 0.125 + 4.0 * 0.25)


@fp64_only
def test_the_universal_loss_bands_its_crossover_by_the_reference_force():
    """`0.125 / 6`, from `test_universal_loss`.

    The reference force is zero, which is the first band, so the crossover is
    the delta itself and a residual of 0.5 stays quadratic.
    """
    batch = batch_of(structure())
    output = prediction(batch, force_off=0.5)
    assert loss_with(
        UniversalLossConfig(delta=1.0), batch, output, energy=1.0, forces=1.0
    ) == pytest.approx(0.125 / 6.0)


@fp64_only
def test_the_band_is_read_off_the_reference_and_not_the_prediction():
    """Otherwise what a structure costs moves while the model learns, and the
    same structure is scored differently at two points of one run."""
    large = Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=POSITIONS,
        properties={"energy": 10.0, "forces": np.full((2, 3), 250.0)},
    )
    batch = batch_of(large)
    output = prediction(batch, force_off=1.0)
    # |F_ref| is 433, the fourth band, so the crossover is a tenth of delta.
    scored = loss_with(
        UniversalLossConfig(delta=1.0), batch, output, energy=0.0, forces=1.0
    )
    assert scored == pytest.approx(0.1 * (1.0 - 0.05) / 6.0)


@fp64_only
def test_the_l1l2_loss_costs_a_length_rather_than_a_square():
    """`1.5 + 2.5`, from `test_weighted_energy_forces_l1l2_loss`.

    The energy is off by three over two atoms and costs 1.5 rather than its
    square. One atom's force error is `(3, 4, 0)`, which costs five rather
    than twenty-five, and the mean over the two atoms is 2.5.
    """
    batch = batch_of(structure(energy=1.0))
    energies = batch.targets["energy"].clone()
    energies[0] = energies[0] + 3.0
    forces = batch.targets["forces"].clone()
    forces[0] = torch.tensor([3.0, 4.0, 0.0], dtype=forces.dtype)
    output = MACEOutput(total_energy=energies, forces=forces)
    assert loss_with(
        L1L2LossConfig(), batch, output, energy=1.0, forces=1.0
    ) == pytest.approx(1.5 + 2.5)


@fp64_only
def test_the_l1l2_weights_scale_their_terms():
    """`2 * 1.5 + 0.4 * 2.5`, from the same test."""
    batch = batch_of(structure(energy=1.0))
    energies = batch.targets["energy"].clone()
    energies[0] = energies[0] + 3.0
    forces = batch.targets["forces"].clone()
    forces[0] = torch.tensor([3.0, 4.0, 0.0], dtype=forces.dtype)
    output = MACEOutput(total_energy=energies, forces=forces)
    assert loss_with(
        L1L2LossConfig(), batch, output, energy=2.0, forces=0.4
    ) == pytest.approx(3.0 + 1.0)


@fp64_only
def test_a_loss_from_another_package_is_selectable_without_touching_the_schema(
    clean_registry,
):
    """The acceptance criterion. The kinds the schema knows are the ones whose
    settings it can validate; a loss from elsewhere cannot be, so it is named
    with its settings and the loss validates them itself."""

    @register_loss("probe_outside")
    class Outside(torch.nn.Module):
        def __init__(self, scale: float = 1.0) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, output, batch):
            return torch.as_tensor(self.scale)

    batch = batch_of(structure())
    config = LossConfig(
        kind=RegisteredLoss(name="probe_outside", settings={"scale": 7.0})
    )
    loss = build_loss(REQUESTED, config)
    assert float(loss(prediction(batch), batch)) == pytest.approx(7.0)


@fp64_only
def test_a_registered_name_nobody_registered_lists_what_there_is():
    config = LossConfig(kind=RegisteredLoss(name="no_such_loss"))
    with pytest.raises(UnknownLossError, match="huber"):
        build_loss(REQUESTED, config)
