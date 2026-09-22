"""The ten error tables, against the frozen tree's own column headings.

The headings are quoted from `mace/tools/tables_utils.py:23-108` rather than
built from a rule. They are what a user greps a log for, so a heading that
reads better is a heading that breaks somebody's script.
"""

from __future__ import annotations

import re

import pytest
from mace_core.tables import (
    TABLE_TYPES,
    Row,
    UnknownTableTypeError,
    error_table,
)

#: Exactly the ten `--error_table` choices of `mace/tools/arg_parser.py:113-130`.
LEGACY_TYPES = {
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
}

#: The headings, quoted. Only the ones that are not a mechanical rewriting of
#: another row, so the list stays readable and still pins every wording that
#: differs: "relative" against "rel", the parenthesised stress unit, and the
#: two dipole units that are not the same unit.
LEGACY_HEADINGS = {
    "PerAtomRMSE": [
        "config_type",
        "RMSE E / meV / atom",
        "RMSE F / meV / A",
        "relative F RMSE %",
    ],
    "TotalRMSE": [
        "config_type",
        "RMSE E / meV",
        "RMSE F / meV / A",
        "relative F RMSE %",
    ],
    "PerAtomMAEstressvirials": [
        "config_type",
        "MAE E / meV / atom",
        "MAE F / meV / A",
        "relative F MAE %",
        "MAE Stress (Virials) / meV / A (A^3)",
    ],
    "DipolePolarRMSE": [
        "config_type",
        "RMSE MU / me A / atom",
        "relative MU RMSE %",
        "RMSE ALPHA e A^2 / V / atom",
    ],
    "EnergyDipoleRMSE": [
        "config_type",
        "RMSE E / meV / atom",
        "RMSE F / meV / A",
        "rel F RMSE %",
        "RMSE MU / mDebye / atom",
        "rel MU RMSE %",
    ],
}

ENERGY_METRICS = {
    "rmse_energy": 0.004,
    "rmse_energy_per_atom": 0.002,
    "mae_energy": 0.003,
    "mae_energy_per_atom": 0.0015,
    "rmse_forces": 0.05,
    "mae_forces": 0.04,
    "rel_rmse_forces": 12.5,
    "rel_mae_forces": 10.0,
}

DIPOLE_METRICS = {
    "rmse_dipole_per_atom": 0.001,
    "mae_dipole_per_atom": 0.0008,
    "rel_rmse_dipole": 7.5,
    "rel_mae_dipole": 6.0,
    "rmse_polarizability_per_atom": 0.002,
}


def rows(metrics, names=("valid_water",), head="water"):
    return [Row(name=name, head=head, metrics=metrics) for name in names]


def headings_of(text: str) -> list[str]:
    return [cell.strip() for cell in text.splitlines()[1].strip("|").split("|")]


def cells_of(text: str, line: int) -> list[str]:
    return [cell.strip() for cell in text.splitlines()[line].strip("|").split("|")]


# ---------------------------------------------------------------------------
# The ten types, and their headings
# ---------------------------------------------------------------------------


def test_the_ten_legacy_types_are_the_ten_here():
    assert set(TABLE_TYPES) == LEGACY_TYPES


@pytest.mark.parametrize("kind", sorted(LEGACY_HEADINGS))
def test_the_headings_are_the_frozen_trees(kind):
    metrics = {
        **ENERGY_METRICS,
        **DIPOLE_METRICS,
        "rmse_stress": 0.01,
        "mae_stress": 0.008,
    }
    assert headings_of(error_table(kind, rows(metrics))) == LEGACY_HEADINGS[kind]


def test_every_type_renders_when_the_run_measured_everything():
    metrics = {
        **ENERGY_METRICS,
        **DIPOLE_METRICS,
        "rmse_stress": 0.01,
        "mae_stress": 0.008,
    }
    for kind in TABLE_TYPES:
        assert "config_type" in error_table(kind, rows(metrics))


# ---------------------------------------------------------------------------
# The numbers and their units
# ---------------------------------------------------------------------------


def test_an_energy_is_printed_in_meV():
    text = error_table("PerAtomRMSE", rows(ENERGY_METRICS))
    assert cells_of(text, 3)[1] == "2.0"


def test_a_relative_error_is_already_a_percentage():
    text = error_table("PerAtomRMSE", rows(ENERGY_METRICS))
    assert cells_of(text, 3)[3] == "12.50"


def test_the_stress_column_takes_the_virials_when_there_is_no_stress():
    """One column, whichever of the two the run produced."""
    metrics = {**ENERGY_METRICS, "rmse_virials": 0.02}
    text = error_table("PerAtomRMSEstressvirials", rows(metrics))
    assert cells_of(text, 3)[4] == "20.0"


# ---------------------------------------------------------------------------
# Row order and skipping
# ---------------------------------------------------------------------------


def test_the_training_rows_come_before_the_validation_rows():
    names = ("valid_b", "train_b", "test_high_pressure", "train_a")
    text = error_table("PerAtomRMSE", rows(ENERGY_METRICS, names))
    printed = [cells_of(text, line)[0] for line in range(3, 7)]
    assert printed == ["train_a", "train_b", "valid_b", "test_high_pressure"]


def test_a_skipped_head_leaves_no_row():
    listed = [
        Row("valid_water", "water", ENERGY_METRICS),
        Row("valid_salt", "salt", ENERGY_METRICS),
    ]
    text = error_table("PerAtomRMSE", listed, skip_heads=["water"])
    assert "valid_water" not in text
    assert "valid_salt" in text


def test_skipping_matches_the_head_and_not_the_row_name():
    """`water` skips `valid_saltwater` in the frozen tree, because it asks
    whether the name contains the head rather than whether it is that head."""
    listed = [Row("valid_saltwater", "saltwater", ENERGY_METRICS)]
    assert "valid_saltwater" in error_table("PerAtomRMSE", listed, skip_heads=["water"])


def test_skipping_every_head_is_refused():
    listed = [Row("valid_water", "water", ENERGY_METRICS)]
    with pytest.raises(UnknownTableTypeError, match="every row was skipped"):
        error_table("PerAtomRMSE", listed, skip_heads=["water"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unknown_type_is_refused_and_lists_the_ten():
    with pytest.raises(UnknownTableTypeError, match="is not an error table"):
        error_table("PerAtomQ95", rows(ENERGY_METRICS))


def test_a_table_the_run_cannot_fill_is_refused_and_says_which_it_can():
    """The frozen tree prints the heading and no rows, and the run reports
    nothing wrong."""
    with pytest.raises(UnknownTableTypeError) as caught:
        error_table("DipoleRMSE", rows(ENERGY_METRICS))
    message = str(caught.value)
    assert "RMSE MU / mDebye / atom" in message
    assert "PerAtomRMSE" in message


def test_the_stress_table_is_refused_when_neither_quantity_was_measured():
    with pytest.raises(UnknownTableTypeError, match="Stress"):
        error_table("PerAtomRMSEstressvirials", rows(ENERGY_METRICS))


# ---------------------------------------------------------------------------
# The rendering itself
# ---------------------------------------------------------------------------


def test_the_columns_line_up():
    names = ("train_a", "valid_a_very_long_name")
    text = error_table("PerAtomRMSE", rows(ENERGY_METRICS, names))
    widths = {len(line) for line in text.splitlines()}
    assert len(widths) == 1


def test_nothing_in_a_cell_is_cut_off():
    text = error_table("PerAtomRMSE", rows(ENERGY_METRICS, ("valid_x" * 6,)))
    assert "valid_x" * 6 in text
    assert not re.search(r"\.\.\.", text)
