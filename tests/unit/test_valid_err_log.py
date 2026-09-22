"""The per-epoch validation line: which heads it names, and which runs get one.

`valid_err_log` is what a training prints after every evaluation, and multihead
is the normal case, so the head has to be on the line rather than in a heading
several lines up. All eleven branches do carry it. Nothing asserted that, which
is what this file is for: it is a property with no cost to lose and no symptom
when lost, since the line still prints and still reads like a whole run's.

Two things came out of pinning it, both recorded rather than endorsed, and both
invisible without running the function:

* one of the ten `--error_table` choices prints **no validation line at all**;
* the two stress-or-virials tables can never print virials.

Both follow from the shape of the function rather than from any one branch. It
is a chain of `elif` with no `else`, so an unmatched table type falls out
silently; and the metrics it reads are a `defaultdict` whose factory returns an
object that is never `None`, so a branch guarded on `is not None` is taken
whether or not the quantity was measured.
"""

import logging
from collections import defaultdict

import pytest

from mace.tools.train import valid_err_log

#: The ten choices of `--error_table` (`mace/tools/arg_parser.py:113-130`).
TABLE_TYPES = [
    "PerAtomRMSE",
    "TotalRMSE",
    "PerAtomRMSEstressvirials",
    "PerAtomMAEstressvirials",
    "PerAtomMAE",
    "TotalMAE",
    "DipoleRMSE",
    "DipoleMAE",
    "DipolePolarRMSE",
    "EnergyDipoleRMSE",
]

#: The one that matches no branch of the chain.
NO_LINE = "DipoleMAE"

ENERGY_AND_FORCES = {
    "rmse_e": 0.004,
    "rmse_e_per_atom": 0.002,
    "rmse_f": 0.05,
    "mae_e": 0.003,
    "mae_e_per_atom": 0.0015,
    "mae_f": 0.04,
}

DIPOLE = {"rmse_mu_per_atom": 0.001, "rmse_polarizability_per_atom": 0.002}


class _NoneMultiply:
    """`MACELoss.compute`'s factory, reproduced so the fixture is its shape.

    Copied rather than imported because it is defined inside `compute`. What
    matters about it is the pair of properties the branches below depend on:
    it survives multiplication, and it formats as the string `None`.
    """

    def __mul__(self, other):
        return _NoneMultiply()

    def __rmul__(self, other):
        return _NoneMultiply()

    def __format__(self, spec):
        return str(None)


class _Logger:
    """The metrics logger's interface, which this function only writes to."""

    def __init__(self):
        self.entries = []

    def log(self, values):
        self.entries.append(dict(values))


def eval_metrics(**measured):
    """What `evaluate` hands over: the defaultdict straight out of `compute`.

    Built as a defaultdict rather than a plain one on purpose. `evaluate`
    (`mace/tools/train.py:607-611`) returns `aux` without converting it, so a
    branch that reads a key nobody set gets an object rather than a `KeyError`.
    """
    metrics = defaultdict(_NoneMultiply)
    metrics.update(measured)
    return metrics


def lines_for(table_type, caplog, **measured):
    """The log lines one table type produces for one head."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        valid_err_log(
            valid_loss=0.5,
            eval_metrics=eval_metrics(**{**ENERGY_AND_FORCES, **DIPOLE, **measured}),
            logger=_Logger(),
            log_errors=table_type,
            epoch=3,
            valid_loader_name="water",
        )
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Epoch")
    ]


# ---------------------------------------------------------------------------
# The head, on every line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_type", [name for name in TABLE_TYPES if name != NO_LINE]
)
def test_every_line_names_the_head(table_type, caplog):
    """Read next to the other heads' lines and the next epoch's, where a
    heading several lines up has stopped applying."""
    printed = lines_for(table_type, caplog, rmse_stress=0.01, mae_stress=0.008)
    assert printed
    assert all("head: water" in line for line in printed)


def test_the_initial_evaluation_names_the_head_too(caplog):
    """Before the first epoch there is no epoch number, and the head is the
    only thing that says which of several the line belongs to."""
    caplog.clear()
    with caplog.at_level(logging.INFO):
        valid_err_log(
            valid_loss=0.5,
            eval_metrics=eval_metrics(**ENERGY_AND_FORCES),
            logger=_Logger(),
            log_errors="PerAtomRMSE",
            epoch=None,
            valid_loader_name="water",
        )
    assert any("Initial: head: water" in r.getMessage() for r in caplog.records)


def test_the_head_reaches_the_metrics_logger_as_well(caplog):
    """The line is for a person and the entry is for the plots, and the plot
    subcommand groups by this key."""
    logger = _Logger()
    with caplog.at_level(logging.INFO):
        valid_err_log(
            valid_loss=0.5,
            eval_metrics=eval_metrics(**ENERGY_AND_FORCES),
            logger=logger,
            log_errors="PerAtomRMSE",
            epoch=3,
            valid_loader_name="water",
        )
    assert logger.entries[0]["head"] == "water"
    assert logger.entries[0]["mode"] == "eval"
    assert logger.entries[0]["epoch"] == 3


# ---------------------------------------------------------------------------
# The choice that prints nothing
# ---------------------------------------------------------------------------


def test_one_of_the_ten_tables_logs_no_validation_line(caplog):
    """`--error_table DipoleMAE` is accepted by the parser, produces a final
    error table, and produces no per-epoch line: the chain has no branch for
    it and no `else`. A run configured that way looks, epoch by epoch, like a
    run that is not evaluating."""
    assert lines_for(NO_LINE, caplog) == []


def test_it_is_the_only_one(caplog):
    """The guard against the test above passing for a second reason."""
    silent = [name for name in TABLE_TYPES if not lines_for(name, caplog)]
    assert silent == [NO_LINE]


# ---------------------------------------------------------------------------
# The virials branches, which cannot be reached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table_type", "label"),
    [("PerAtomRMSEstressvirials", "RMSE_stress"), ("PerAtomMAEstressvirials", "MAE_stress")],
)
def test_a_virials_run_prints_a_stress_of_None(table_type, label, caplog):
    """The stress branch is guarded on `eval_metrics[...] is not None`, and
    reading a key nobody set creates an object that is not `None`. So it wins,
    the virials branch below it is dead, and the line reports a stress the run
    never measured."""
    printed = lines_for(table_type, caplog, rmse_virials_per_atom=0.02, mae_virials=0.02)
    assert len(printed) == 1
    assert f"{label}=None" in printed[0]
    assert "virials" not in printed[0]


def test_a_stress_run_prints_its_stress(caplog):
    """The guard: the `None` above is the missing quantity and not the
    formatting."""
    printed = lines_for("PerAtomRMSEstressvirials", caplog, rmse_stress=0.01)
    assert "RMSE_stress=   10.00 meV / A^3" in printed[0]


def test_the_mae_branch_is_guarded_on_a_key_nothing_sets(caplog):
    """`mae_stress_per_atom` is read by the guard and written by nobody:
    `MACELoss.compute` sets `mae_stress` (`mace/tools/train.py:786`). The
    branch is therefore taken on the strength of the default object alone."""
    printed = lines_for("PerAtomMAEstressvirials", caplog, mae_stress=0.008)
    assert "MAE_stress=    8.00 meV / A^3" in printed[0]
