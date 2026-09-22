"""The v1 sign conventions are the characterization suite's, character for
character.

`tests/unit/test_physics_glue.py` pinned the force, stress and virial
conventions against finite differences, and its module docstring is the
normative statement of each one. `mace_core.units` restates them for the
rewrite, which is only safe while the two texts are identical: a convention
that has been re-worded has been re-derived, and re-deriving the virial's sign
is exactly how a port loses it.

The text is read out of both files rather than imported, so this test needs
neither tree installed and works while the two are governed by different
toolchains.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARACTERIZATION = REPO_ROOT / "tests" / "unit" / "test_physics_glue.py"
UNITS = REPO_ROOT / "packages" / "mace-core" / "src" / "mace_core" / "units.py"

#: The v1 constant holding each statement, and the bullet the characterization
#: docstring introduces it with. Both halves are spelled out: deriving either
#: from the other would let a rename on one side pass unnoticed.
STATEMENTS = {
    "FORCE_SIGN_CONVENTION": "forces",
    "STRESS_SIGN_CONVENTION": "stress",
    "VIRIAL_SIGN_CONVENTION": "virials",
}


def module_docstring(path: Path) -> str:
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)


def assigned_strings(path: Path) -> dict[str, str]:
    """Module-level ``NAME = "..."`` assignments, with implicit concatenation
    already joined, which is how a long statement is written in the source."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                found[target.id] = node.value.value
    return found


def test_both_files_are_where_this_test_thinks_they_are():
    assert CHARACTERIZATION.exists(), CHARACTERIZATION
    assert UNITS.exists(), UNITS


@pytest.mark.parametrize(("constant", "quantity"), sorted(STATEMENTS.items()))
def test_the_statement_is_the_characterization_suite_s_verbatim(constant, quantity):
    statement = assigned_strings(UNITS)[constant]
    assert statement.startswith(f"{quantity} = ")

    # The docstring writes each one as a markdown bullet with the quantity
    # emboldened. Everything after that is the statement, and it must match.
    bullet = f"* **{quantity}** = "
    docstring = module_docstring(CHARACTERIZATION)
    assert bullet in docstring, (
        f"the characterization docstring no longer introduces {quantity} with "
        f"{bullet!r}. It is the normative text, so this test has to be taught "
        f"the new shape rather than the statement reworded to match."
    )
    pinned = docstring.split(bullet, 1)[1].split("\n", 1)[0].strip()

    assert statement == f"{quantity} = {pinned}", (
        f"mace_core.units.{constant} and the characterization docstring "
        f"disagree.\n  units: {statement}\n  pinned: {quantity} = {pinned}\n"
        f"The docstring is the authority: it is what was checked against "
        f"finite differences. Reword the constant, never the pin."
    )


def test_the_prose_around_the_constants_also_carries_the_three_statements():
    """The units module reproduces them in its own docstring as well, so a
    reader sees them without chasing three constants. That copy is checked too,
    or it becomes the one that drifts."""
    docstring = module_docstring(UNITS)
    for constant in STATEMENTS:
        assert assigned_strings(UNITS)[constant] in docstring
