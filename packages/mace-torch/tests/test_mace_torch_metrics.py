"""Errors over a loader, and what a missing label does to them.

The fixture is the one the frozen tree was measured on: two structures of two
atoms each, where only the first carries labels. That is the whole subject.
With every structure labelled the two implementations agree, so a test that
used a fully labelled set would pass against either.

No model runs. The predictions are handed in directly, so what is measured is
the accumulator and nothing else.
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
from mace_torch.train import (
    RunningMetrics,
    build_loss,
    log_validation,
    metric_specs,
    selection_loss,
)

CATALOGUE = load_default_catalogue()
REQUESTED = resolve_requested(["energy", "forces"], CATALOGUE)
SPECS = target_specs(REQUESTED)
Z_TABLE = AtomicNumberTable([1])

#: Two atoms far enough apart that no edge exists.
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

#: What the frozen tree reports on the fixture below, measured by running its
#: own `MACELoss` over it. Quoted rather than recomputed: the point of the
#: numbers is that they are the other implementation's, and reproducing them
#: from a formula here would be reproducing the defect.
LEGACY = {
    "mae_e": 1.0,
    "mae_e_per_atom": 1.5,
    "rmse_e_per_atom": 1.8027756377319946,
    "rel_mae_f": 100.0,
}


def structure(*, labelled: bool) -> Configuration:
    """One structure, with or without values.

    An unlabelled structure carries ``None`` for every property, which is what
    a parsed file gives for a key it did not hold, and the collate function
    turns into a zero row with weight zero.
    """
    if labelled:
        properties = {"energy": -10.0, "forces": np.zeros((2, 3))}
        properties["forces"][0, 0] = 1.0
        properties["forces"][1, 0] = -1.0
    else:
        properties = {"energy": None, "forces": None}
    return Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=POSITIONS,
        properties=properties,
    )


def batch_of(*configurations):
    dataset = GraphDataset(
        list(configurations), cutoff=0.5, z_table=Z_TABLE, targets=SPECS
    )
    return collate_training(
        [dataset[index] for index in range(len(configurations))], z_table=Z_TABLE
    )


def measured(batch, energies, forces) -> dict[str, float]:
    """The metrics of one batch against the given predictions."""
    loss = build_loss(REQUESTED, LossConfig())
    metrics = RunningMetrics(metric_specs(REQUESTED), loss)
    metrics.update(
        MACEOutput(
            total_energy=torch.tensor(energies, dtype=torch.float64),
            forces=torch.tensor(forces, dtype=torch.float64),
        ),
        batch,
    )
    return metrics.compute()


#: The prediction the legacy numbers were measured against: the labelled
#: structure's energy off by 1.0 and one force component off by 0.5, and
#: wildly wrong values on the unlabelled one.
ENERGIES = [-9.0, 5.0]
FORCES = [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [7.0, 0.0, 0.0], [7.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# What a spec reads off a declaration
# ---------------------------------------------------------------------------


def test_a_spec_reads_its_shape_off_the_declaration():
    specs = {spec.name: spec for spec in metric_specs(REQUESTED)}
    assert specs["energy"].per_atom_variant and not specs["energy"].per_atom
    assert specs["forces"].per_atom and not specs["forces"].per_atom_variant


def test_a_per_atom_quantity_gets_no_per_atom_variant():
    """A force is already per atom, so dividing it by the count means nothing."""
    results = measured(batch_of(structure(labelled=True)), ENERGIES[:1], FORCES[:2])
    assert "rmse_forces_per_atom" not in results


# ---------------------------------------------------------------------------
# The fully labelled case, where the two implementations agree
# ---------------------------------------------------------------------------


@fp64_only
def test_a_perfect_prediction_scores_zero_everywhere():
    batch = batch_of(structure(labelled=True))
    reference_forces = batch.targets["forces"].tolist()
    results = measured(batch, [-10.0], reference_forces)
    assert results["rmse_energy"] == 0.0
    assert results["mae_forces"] == 0.0
    assert results["loss"] == 0.0


@fp64_only
def test_the_errors_of_one_labelled_structure_are_the_plain_arithmetic():
    batch = batch_of(structure(labelled=True))
    results = measured(batch, ENERGIES[:1], FORCES[:2])
    assert results["mae_energy"] == pytest.approx(1.0)
    # Two atoms, so the residual halves.
    assert results["mae_energy_per_atom"] == pytest.approx(0.5)
    # One component off by 0.5 in each of two atoms, over six components.
    assert results["mae_forces"] == pytest.approx(1.0 / 6)
    # The reference has two components of magnitude 1.0, over the same six.
    assert results["rel_mae_forces"] == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The unlabelled structure, which is the whole point
# ---------------------------------------------------------------------------


@fp64_only
def test_an_unlabelled_structure_changes_nothing():
    """Adding a structure with no values leaves every error where it was."""
    alone = measured(batch_of(structure(labelled=True)), ENERGIES[:1], FORCES[:2])
    with_unlabelled = measured(
        batch_of(structure(labelled=True), structure(labelled=False)),
        ENERGIES,
        FORCES,
    )
    for name, value in alone.items():
        if name == "loss":
            # The loss is per structure, and there are now two of them.
            continue
        assert with_unlabelled[name] == pytest.approx(value), name


@fp64_only
def test_the_per_atom_energy_error_is_not_the_frozen_trees():
    """The frozen tree filters the total and not the per-atom delta.

    `mace/tools/train.py:659-663` appends both and passes only `delta_es` to
    `filter_nonzero_weight`, so the per-atom row of a structure with no energy
    is averaged in. It is the default error table's energy column.
    """
    results = measured(
        batch_of(structure(labelled=True), structure(labelled=False)),
        ENERGIES,
        FORCES,
    )
    assert results["mae_energy"] == pytest.approx(LEGACY["mae_e"])
    assert results["mae_energy_per_atom"] == pytest.approx(0.5)
    assert results["mae_energy_per_atom"] != pytest.approx(LEGACY["mae_e_per_atom"])
    assert results["rmse_energy_per_atom"] == pytest.approx(0.5)
    assert results["rmse_energy_per_atom"] != pytest.approx(LEGACY["rmse_e_per_atom"])


@fp64_only
def test_the_relative_force_error_divides_by_the_labelled_rows_only():
    """The frozen tree's denominator is every row, including the unlabelled.

    `mace/tools/train.py:666-667` appends the targets before the filter runs
    and never masks them, so the target norm is diluted by every row the model
    was not asked to fit, and the reported relative error is inflated by
    exactly that factor. Here it is 2.0.
    """
    results = measured(
        batch_of(structure(labelled=True), structure(labelled=False)),
        ENERGIES,
        FORCES,
    )
    assert results["rel_mae_forces"] == pytest.approx(50.0, abs=1e-6)
    assert results["rel_mae_forces"] == pytest.approx(LEGACY["rel_mae_f"] / 2, abs=1e-6)


@fp64_only
def test_a_quantity_no_structure_carries_is_left_out_rather_than_zeroed():
    """An error over nothing is not a small error."""
    results = measured(
        batch_of(structure(labelled=False)),
        ENERGIES[1:],
        FORCES[2:],
    )
    assert "rmse_energy" not in results
    assert "rmse_forces" not in results
    assert "loss" in results


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


@fp64_only
def test_the_loss_is_per_structure_and_not_per_batch():
    """The frozen tree's definition, which is not the training loop's.

    The loop reports a mean over batches. This is the sum over batches divided
    by the structures, so two identical structures in one batch score half of
    what one of them scores on its own. Both definitions are stated where they
    are produced and neither is converted into the other.
    """
    one = measured(batch_of(structure(labelled=True)), ENERGIES[:1], FORCES[:2])
    two = measured(
        batch_of(structure(labelled=True), structure(labelled=True)),
        ENERGIES[:1] * 2,
        FORCES[:2] * 2,
    )
    assert two["loss"] == pytest.approx(one["loss"] / 2)


@fp64_only
def test_the_reported_loss_moves_with_the_batch_size():
    """Carried from the frozen tree, and worth knowing before reading one.

    The loss is already a mean over the batch's elements, so a batch twice the
    size halves it, and the per-structure division above halves it again. The
    number is therefore comparable within a run and not across two runs whose
    validation batch size differs.
    """
    one = measured(batch_of(structure(labelled=True)), ENERGIES[:1], FORCES[:2])
    padded = measured(
        batch_of(structure(labelled=True), structure(labelled=False)),
        ENERGIES,
        FORCES,
    )
    assert padded["loss"] == pytest.approx(one["loss"] / 4)


def test_scoring_nothing_is_refused():
    loss = build_loss(REQUESTED, LossConfig())
    with pytest.raises(ValueError, match="no structure was scored"):
        RunningMetrics(metric_specs(REQUESTED), loss).compute()


# ---------------------------------------------------------------------------
# The number a checkpoint is chosen by
# ---------------------------------------------------------------------------


PER_HEAD = {"small": {"loss": 1.0}, "large": {"loss": 3.0}}


def test_the_default_rule_averages_the_heads():
    assert selection_loss(PER_HEAD, "mean_over_heads") == pytest.approx(2.0)


def test_the_average_is_unweighted():
    """Weighting by structure count would hand the checkpoint to the largest
    head, which is the thing the balancing exists to stop."""
    assert selection_loss(PER_HEAD, "mean_over_heads") == selection_loss(
        {name: dict(value) for name, value in reversed(list(PER_HEAD.items()))},
        "mean_over_heads",
    )


def test_the_frozen_trees_rule_is_reachable_and_is_not_the_default():
    """It selects on the last head alone, so it depends on the order the
    configuration lists the heads in."""
    assert selection_loss(PER_HEAD, "last_head") == 3.0
    assert (
        selection_loss({"large": {"loss": 3.0}, "small": {"loss": 1.0}}, "last_head")
        == 1.0
    )


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="selection rule"):
        selection_loss(PER_HEAD, "best_head")


# ---------------------------------------------------------------------------
# The per-epoch validation line
# ---------------------------------------------------------------------------


MEASURED = {
    "water": {"loss": 0.5, "rmse_energy_per_atom": 0.002, "rmse_forces": 0.05},
    "salt": {"loss": 0.7, "rmse_energy_per_atom": 0.004, "rmse_forces": 0.09},
}


def test_there_is_one_line_per_head_and_each_names_its_own(caplog):
    with caplog.at_level("INFO"):
        log_validation(3, MEASURED)
    printed = [record.getMessage() for record in caplog.records]
    assert len(printed) == 2
    assert "head water" in printed[0]
    assert "head salt" in printed[1]


def test_the_line_does_not_depend_on_an_error_table(caplog):
    """`log_validation` takes no table type, so no configuration can silence
    it. The frozen tree writes one branch per type with no fallback, so
    `DipoleMAE` prints nothing at all and the virials branches are unreachable;
    both are pinned in `tests/unit/test_valid_err_log.py`."""
    import inspect

    from mace_torch.train import log_validation as function

    assert set(inspect.signature(function).parameters) == {"epoch", "per_head"}


def test_the_line_reports_what_was_measured(caplog):
    with caplog.at_level("INFO"):
        log_validation(3, {"water": MEASURED["water"]})
    printed = caplog.records[0].getMessage()
    assert "loss=0.5" in printed
    assert "rmse_forces=0.05" in printed
