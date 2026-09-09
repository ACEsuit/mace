"""Read `debt_book.md`, and turn an open row into an xfail marker.

The debt book is the authored source: prose belongs next to the compromise it
describes, and a reviewer of a burn-step pull request has to be able to read
why the row existed. This module parses that file and provides the one piece
of machinery the markdown cannot carry, the marker itself.

The mechanism, restated because it is easy to get backwards. A row's fitness
test asserts the state of the tree *after* the debt is burned, so while the
debt lives that assertion fails. `xfail(strict=True)` makes the expected
failure green and, crucially, makes an unexpected **pass** red: the moment the
burn-step ticket's work lands and the row is still here, the suite fails. That
is the "overdue" trigger, and it needs no date and no network call. The
burn-step pull request deletes the row and the decorator together; deleting
either one alone is also red, from the two directions the meta-tests cover.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOK = Path(__file__).with_name("debt_book.md")
ARCHITECTURE = Path(__file__).parent

#: The five columns, in order. The schema is asserted rather than assumed: a
#: sixth column, or a renamed one, changes what every reader below means.
COLUMNS: Tuple[str, ...] = (
    "debt_id",
    "description",
    "burn-step (ticket)",
    "burn-check",
    "fitness-test",
)

#: `RET-1 (#1597)` / `DEP-2 (#1594)`: a ticket id and the issue it resolves to.
#: Both halves are required. The id alone cannot be looked up by a reader who
#: was not in the room; the issue number alone says nothing about which phase
#: of the migration owes the work.
BURN_STEP = re.compile(r"^[A-Z]+-\d+[a-z]? \(#\d+\)$")

#: `test_name` or `test_name[param]`.
FITNESS_TEST = re.compile(r"^(?P<function>test_[a-z0-9_]+)(?:\[(?P<param>[^]]+)])?$")


@dataclass(frozen=True)
class DebtRow:
    debt_id: str
    description: str
    burn_step: str
    burn_check: str
    fitness_test: str

    @property
    def fitness_function(self) -> str:
        """The test function name, without a parametrization suffix."""
        match = FITNESS_TEST.match(self.fitness_test)
        assert match is not None, self.fitness_test
        return match.group("function")

    @property
    def fitness_param(self) -> str:
        """The parametrization id, or the empty string for a plain test."""
        match = FITNESS_TEST.match(self.fitness_test)
        assert match is not None, self.fitness_test
        return match.group("param") or ""


def _ledger_lines() -> List[str]:
    """The rows of the ledger table, as raw markdown lines.

    The file holds two tables: the mechanics table in the prose above, and the
    ledger. Only the ledger is data, and it is found by its heading rather than
    by being the last table, so that adding a section below it is harmless.
    """
    text = BOOK.read_text(encoding="utf-8")
    heading = "## The ledger"
    assert heading in text, f"{BOOK.name} has no '{heading}' section"
    return [
        line.strip()
        for line in text[text.index(heading) :].splitlines()
        if line.strip().startswith("|")
    ]


def _cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def header() -> List[str]:
    """The ledger's column names, as written."""
    return _cells(_ledger_lines()[0])


def rows() -> Dict[str, DebtRow]:
    """The open debt rows, keyed by `debt_id`, in the order the book lists them."""
    lines = _ledger_lines()
    result: Dict[str, DebtRow] = {}
    for line in lines[2:]:  # [0] header, [1] the |---| separator
        cells = _cells(line)
        if len(cells) != len(COLUMNS):
            raise ValueError(
                f"a ledger row has {len(cells)} cells, not {len(COLUMNS)}: {line!r}"
            )
        debt_id = cells[0].strip("`")
        result[debt_id] = DebtRow(
            debt_id=debt_id,
            description=cells[1],
            burn_step=cells[2],
            burn_check=cells[3],
            fitness_test=cells[4].strip("`"),
        )
    return result


def open_debt(debt_id: str) -> "pytest.MarkDecorator":
    """Mark a fitness test `xfail(strict=True)` while its debt row is open.

    Returns a mark decorator, so the same call serves both shapes the ledger
    needs: `@open_debt("...")` above a plain test, and
    `pytest.param(..., marks=open_debt("..."))` for the engine-default rows,
    which are one parametrized test over the axes of the manifest rather than
    six near-identical functions.

    An unknown id raises at import time, which is a collection error rather
    than a skipped test. That is the case worth failing loudly: it is what a
    burn-step pull request that deleted the row and forgot the decorator looks
    like, and a decorator quietly degrading to a no-op would leave the tree
    with an unmarked test asserting a state it has not reached.
    """
    book = rows()
    if debt_id not in book:
        raise LookupError(
            f"{debt_id} is not a row in {BOOK.name}. If the debt was just "
            f"burned, delete this @open_debt decorator in the same change that "
            f"deleted the row -- the test then stays as a permanent guard. If "
            f"the id is a typo, the book lists {sorted(book)}."
        )
    row = book[debt_id]
    return pytest.mark.xfail(
        strict=True,
        reason=(
            f"{debt_id} is an open debt, burned by {row.burn_step}. If this "
            f"test now PASSES, the debt is gone: delete its row from "
            f"{BOOK.name} and the @open_debt decorator here. strict=True is "
            f"what turns that pass into a failure instead of a silent xpass."
        ),
    )


def debt_claims() -> Dict[str, List[str]]:
    """Every `open_debt(...)` id claimed in `tests/architecture`, and where.

    Read by syntax tree from the test sources rather than by importing them:
    an unknown id raises out of `open_debt`, so importing in order to find the
    ids would fail on exactly the tree this is meant to describe.

    Two argument shapes are recognised, because the ledger has two kinds of
    row. A string literal claims one id. An f-string claims a **prefix**,
    reported with a trailing `*`: the engine-default rows are one parametrized
    test over the axes of `capabilities.toml`, so their ids are built as
    `f"DEBT-ENGINE-DEFAULT-{axis}"` and no literal for them exists anywhere.
    Resolving the f-string would mean evaluating it; matching its constant
    head is enough to tell a claimed row from an unclaimed one.
    """
    found: Dict[str, List[str]] = {}
    for path in sorted(ARCHITECTURE.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name != "open_debt" or not node.args:
                continue
            argument = node.args[0]
            claim = None
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                claim = argument.value
            elif isinstance(argument, ast.JoinedStr):
                head = argument.values[0] if argument.values else None
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    claim = f"{head.value}*"
            if claim:
                found.setdefault(claim, []).append(path.name)
    return found


def _claims(claim: str, debt_id: str) -> bool:
    if claim.endswith("*"):
        return debt_id.startswith(claim[:-1])
    return claim == debt_id


def defined_test_functions() -> Dict[str, str]:
    """Test function name -> the file that defines it, over `tests/architecture`."""
    found: Dict[str, str] = {}
    for path in sorted(ARCHITECTURE.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    found[node.name] = path.name
    return found


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def problems() -> List[str]:
    """Everything wrong with the debt book, as sentences."""
    found: List[str] = []

    if header() != list(COLUMNS):
        found.append(
            f"the ledger's columns are {header()} but the schema is "
            f"{list(COLUMNS)}"
        )

    try:
        book = rows()
    except ValueError as error:
        return found + [str(error)]

    if not book:
        found.append(
            "the ledger is empty. An empty debt book means either the "
            "migration is finished or nobody is writing the rows down, and "
            "the two look identical from here"
        )

    defined = defined_test_functions()
    for debt_id, row in book.items():
        if not FITNESS_TEST.match(row.fitness_test):
            found.append(
                f"{debt_id}: fitness-test {row.fitness_test!r} is not a test "
                f"name (`test_x` or `test_x[param]`)"
            )
            continue
        if row.fitness_function not in defined:
            found.append(
                f"{debt_id}: fitness-test names {row.fitness_function}, which "
                f"is not defined in tests/architecture. A row whose test does "
                f"not exist reads as coverage and is worse than no row"
            )
        if not BURN_STEP.match(row.burn_step):
            found.append(
                f"{debt_id}: burn-step {row.burn_step!r} is not a ticket id and "
                f"issue number, e.g. 'RET-1 (#1597)'"
            )
        if len(row.burn_check.split()) < 5:
            found.append(
                f"{debt_id}: burn-check is {row.burn_check!r}. It has to say "
                f"what would be true of the tree once the debt is gone, in "
                f"terms the reviewer of the burn-step pull request can check"
            )
        if len(row.description.split()) < 6:
            found.append(f"{debt_id}: the description is too short to be one")

    claims = debt_claims()
    for debt_id in book:
        if not any(_claims(claim, debt_id) for claim in claims):
            found.append(
                f"{debt_id}: a row with no open_debt marker anywhere in "
                f"tests/architecture, so its fitness test is not xfailed and "
                f"the suite is red for an expected reason"
            )
    for claim, where in sorted(claims.items()):
        if not any(_claims(claim, debt_id) for debt_id in book):
            found.append(
                f"{claim}: marked open_debt in {where} but matches no row in "
                f"the book"
            )

    return found
