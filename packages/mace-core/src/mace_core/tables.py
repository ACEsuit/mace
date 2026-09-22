"""The error table a finished run prints.

Ten table types, because the frozen tree has ten and every one of them is
somebody's habit. What they are is a choice of columns over the same metrics,
so they are ten rows of a table here rather than ten branches: legacy spells
them as a chain of ``elif`` on the type name, twice over, once to name the
columns and once to format them, and the two chains can disagree.

Two differences from the frozen tree, both deliberate.

**A type that names a quantity the run does not produce is refused.** Legacy's
formatting chain falls through, so the table prints with a header and no rows
and the run reports nothing wrong. Asking for a dipole table from a run that
fits energies is a mistake in the configuration, and it is cheaper to hear
about it than to read an empty table.

**Skipping a head matches the head, not the row name.** Legacy asks
``any(skip in name for skip in skip_heads)`` over names like ``valid_water``,
so a head called ``water`` also skips ``valid_saltwater`` and a head called
``a`` skips everything. The head is carried beside the row here, so the match
is on what it is rather than on how the row was spelled.

Rendering is plain text rather than ``prettytable``. The columns and their
units are the frozen tree's, character for character; the borders are not, and
a dependency for drawing them is not worth carrying.

The module is here rather than beside the training loop because it is
arithmetic over a mapping of floats and imports no framework, and because the
configuration validates the requested type against :data:`TABLE_TYPES` when a
run starts. A name checked only where the table is rendered is checked after
the two days of training that produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "TABLE_TYPES",
    "Column",
    "Row",
    "UnknownTableTypeError",
    "error_table",
]


class UnknownTableTypeError(KeyError):
    """A table type nobody defines, or one this run cannot fill."""


@dataclass(frozen=True)
class Column:
    """One column: its heading, where its number comes from, and its scale.

    Attributes:
        label: The heading, including the unit. The frozen tree's wording.
        metrics: Candidate metric names, first present wins. More than one
            because the stress and virial column is whichever of the two the
            run produced, and a run producing neither should not get the
            column at all.
        scale: What the metric is multiplied by to reach the unit in the
            label. ``1000`` turns eV into meV; a relative error is already a
            percentage and carries ``1``.
        decimals: Digits after the point.
    """

    label: str
    metrics: tuple[str, ...]
    scale: float = 1000.0
    decimals: int = 1

    def value(self, metrics: Mapping[str, float]) -> float | None:
        """This column's number, or ``None`` when the run produced none."""
        for name in self.metrics:
            if name in metrics:
                return metrics[name] * self.scale
        return None


def _energy(statistic: str, per_atom: bool) -> Column:
    suffix = "_per_atom" if per_atom else ""
    unit = " / atom" if per_atom else ""
    return Column(
        f"{statistic.upper()} E / meV{unit}", (f"{statistic}_energy{suffix}",)
    )


def _forces(statistic: str) -> Column:
    return Column(f"{statistic.upper()} F / meV / A", (f"{statistic}_forces",))


def _relative_forces(statistic: str, short: bool = False) -> Column:
    label = "rel" if short else "relative"
    return Column(
        f"{label} F {statistic.upper()} %", (f"rel_{statistic}_forces",), 1.0, 2
    )


def _stress_or_virials(statistic: str) -> Column:
    return Column(
        f"{statistic.upper()} Stress (Virials) / meV / A (A^3)",
        (f"{statistic}_stress", f"{statistic}_virials"),
    )


#: The ten types the frozen tree offers, by the name its `--error_table` flag
#: takes. `PerAtomRMSE` is the default there and here.
TABLE_TYPES: dict[str, tuple[Column, ...]] = {
    "TotalRMSE": (
        _energy("rmse", per_atom=False),
        _forces("rmse"),
        _relative_forces("rmse"),
    ),
    "PerAtomRMSE": (
        _energy("rmse", per_atom=True),
        _forces("rmse"),
        _relative_forces("rmse"),
    ),
    "PerAtomRMSEstressvirials": (
        _energy("rmse", per_atom=True),
        _forces("rmse"),
        _relative_forces("rmse"),
        _stress_or_virials("rmse"),
    ),
    "PerAtomMAEstressvirials": (
        _energy("mae", per_atom=True),
        _forces("mae"),
        _relative_forces("mae"),
        _stress_or_virials("mae"),
    ),
    "TotalMAE": (
        _energy("mae", per_atom=False),
        _forces("mae"),
        _relative_forces("mae"),
    ),
    "PerAtomMAE": (
        _energy("mae", per_atom=True),
        _forces("mae"),
        _relative_forces("mae"),
    ),
    "DipoleRMSE": (
        Column("RMSE MU / mDebye / atom", ("rmse_dipole_per_atom",), 1000.0, 2),
        Column("relative MU RMSE %", ("rel_rmse_dipole",), 1.0, 1),
    ),
    "DipoleMAE": (
        Column("MAE MU / mDebye / atom", ("mae_dipole_per_atom",), 1000.0, 2),
        Column("relative MU MAE %", ("rel_mae_dipole",), 1.0, 1),
    ),
    "DipolePolarRMSE": (
        Column("RMSE MU / me A / atom", ("rmse_dipole_per_atom",), 1000.0, 2),
        Column("relative MU RMSE %", ("rel_rmse_dipole",), 1.0, 1),
        Column(
            "RMSE ALPHA e A^2 / V / atom",
            ("rmse_polarizability_per_atom",),
            1000.0,
            2,
        ),
    ),
    "EnergyDipoleRMSE": (
        _energy("rmse", per_atom=True),
        _forces("rmse"),
        _relative_forces("rmse", short=True),
        Column("RMSE MU / mDebye / atom", ("rmse_dipole_per_atom",)),
        Column("rel MU RMSE %", ("rel_rmse_dipole",), 1.0, 1),
    ),
}

#: Row names starting with these are the training and validation splits, and
#: they are printed before the test sets whatever they are called.
_SPLIT_ORDER = ("train_", "valid_")


@dataclass(frozen=True)
class Row:
    """One loader's errors, and which head they belong to.

    Attributes:
        name: What the row is called, such as ``valid_water``.
        head: The head, carried rather than parsed back out of the name.
        metrics: What that loader measured.
    """

    name: str
    head: str
    metrics: Mapping[str, float]


def error_table(
    kind: str, rows: Sequence[Row], *, skip_heads: Sequence[str] = ()
) -> str:
    """Render one table.

    Args:
        kind: A key of :data:`TABLE_TYPES`.
        rows: One per loader. Sorted with the training rows first and the
            validation rows second, which is the order the frozen tree's sort
            key asks for and does not achieve: it compares against the bare
            words ``train`` and ``valid``, and a multi-head run's rows are
            named ``train_<head>``, so none of them ever matches.
        skip_heads: Heads to leave out.

    Raises:
        UnknownTableTypeError: On a type nobody defines, and on one whose
            columns the run produced no metric for.
    """
    if kind not in TABLE_TYPES:
        raise UnknownTableTypeError(
            f"{kind!r} is not an error table. They are {sorted(TABLE_TYPES)}."
        )
    columns = TABLE_TYPES[kind]
    kept = [row for row in rows if row.head not in set(skip_heads)]
    if not kept:
        raise UnknownTableTypeError(
            f"every row was skipped, so the {kind!r} table has nothing in it. "
            f"The heads asked to be skipped are {sorted(set(skip_heads))}."
        )
    _check_filled(kind, columns, kept)

    headings = ["config_type", *(column.label for column in columns)]
    body = [
        [row.name, *(_format(column, row.metrics) for column in columns)]
        for row in sorted(kept, key=_sort_key)
    ]
    return _render(headings, body)


def _check_filled(kind: str, columns: Sequence[Column], rows: Sequence[Row]) -> None:
    """Refuse a table whose columns this run produced nothing for."""
    missing = [
        column.label
        for column in columns
        if all(column.value(row.metrics) is None for row in rows)
    ]
    if missing:
        fits = sorted(
            name
            for name, candidate in TABLE_TYPES.items()
            if all(
                any(column.value(row.metrics) is not None for row in rows)
                for column in candidate
            )
        )
        raise UnknownTableTypeError(
            f"the {kind!r} table has columns {missing} and this run measured "
            f"nothing for them. The types it can fill are {fits}."
        )


def _format(column: Column, metrics: Mapping[str, float]) -> str:
    value = column.value(metrics)
    return "" if value is None else f"{value:.{column.decimals}f}"


def _sort_key(row: Row) -> tuple[int, str]:
    for position, prefix in enumerate(_SPLIT_ORDER):
        if row.name.startswith(prefix):
            return position, row.name
    return len(_SPLIT_ORDER), row.name


def _render(headings: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Plain aligned text. The numbers are right aligned, the names are not."""
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in body))
        if body
        else len(headings[index])
        for index in range(len(headings))
    ]
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(cells: Sequence[str], *, heading: bool) -> str:
        padded = [
            cell.ljust(width) if heading or index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    return "\n".join(
        [rule, line(headings, heading=True), rule]
        + [line(row, heading=False) for row in body]
        + [rule]
    )
