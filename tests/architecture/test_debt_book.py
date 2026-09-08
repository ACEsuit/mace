"""One fitness test per debt row, failing until the debt is burned.

Each test below asserts the state of the tree *after* its debt is gone, and
carries `@open_debt(...)` for as long as its row is in `debt_book.md`. That
decorator is `xfail(strict=True)`, so:

* debt still there, row open      -> xfail, green
* debt burned, row open           -> XPASS(strict), **red**
* debt burned, row deleted        -> passed, and the assertion stays forever
* debt still there, row deleted   -> failed

The third line is what the ledger is for and the second is what makes it
work. "The burn-step ticket closed while the row is still open" needs no date
and no call to GitHub: it is exactly an unexpected pass, and `strict` turns
that into a failure. A burn-step pull request therefore deletes the row and
the decorator in one change.

The tests are ordinary asserts about the frozen tree and about the manifest.
Nothing here imports the legacy package: reading the source is enough, and the
architecture job is a fast CPU gate on every pull request.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture import capabilities, debt, v1_surface

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The five cross-backend weight converters. Their existence is what the
#: accelerated-backend tolerance carries a cross-layout term for.
CONVERT_CLIS = (
    "mace/cli/convert_e3nn_cueq.py",
    "mace/cli/convert_cueq_e3nn.py",
    "mace/cli/convert_e3nn_oeq.py",
    "mace/cli/convert_oeq_e3nn.py",
    "mace/cli/convert_e3nn_hybrid.py",
)


def _calls_jit_compile(path: Path) -> bool:
    """True when the file calls `jit.compile` or `torch.jit.save`.

    Read by syntax tree, not by grep: `run_train.py` mentions compilation in
    log messages and in a comment, and a textual match would report the debt
    burned the day the calls go and the prose stays.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr not in {"compile", "save", "script"}:
            continue
        owner = function.value
        if isinstance(owner, ast.Name) and owner.id == "jit":
            return True
        if isinstance(owner, ast.Attribute) and owner.attr == "jit":
            return True
    return False


# ---------------------------------------------------------------------------
# Parity tolerances
# ---------------------------------------------------------------------------


@debt.open_debt("DEBT-TOL-PARITY-FP64-CPU")
def test_debt_parity_fp64_cpu_tolerance_is_gone():
    """The in-process legacy-vs-v1 comparison no longer exists to be toleranced.

    Not "the tolerance row was deleted": `fp64_cpu_reference` outlives the
    migration as the row every committed golden is asserted at. What ends is
    the *comparison against a live legacy model*, and it ends when no file
    under `tests/parity/` imports the legacy package any more.
    """
    parity = REPO_ROOT / "tests" / "parity"
    assert parity.is_dir(), (
        "tests/parity/ does not exist yet, so there is no legacy-vs-v1 "
        "comparison and this debt cannot be burned. PAR-1 creates it."
    )
    legacy_importers = []
    for path in sorted(parity.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "mace" for name in names):
                legacy_importers.append(str(path.relative_to(REPO_ROOT)))
                break
    assert not legacy_importers, (
        f"these parity files still import the legacy package, so the fp64 "
        f"tolerance is still carrying a live comparison: {legacy_importers}"
    )


@debt.open_debt("DEBT-TOL-PARITY-FP64-ACCEL")
def test_debt_parity_fp64_accelerated_tolerance_is_gone():
    """The cross-layout term in the accelerated tolerance is gone.

    While the five converters exist, an accelerated comparison can cross two
    weight layouts as well as a kernel and a device, and 1e-5 is carrying all
    three. Once they go, v1 has one canonical layout and the comparison is
    backend against backend.
    """
    surviving = [path for path in CONVERT_CLIS if (REPO_ROOT / path).is_file()]
    assert not surviving, (
        f"the cross-layout converters are still in the tree: {surviving}"
    )


# ---------------------------------------------------------------------------
# Engine defaults, one parameter per axis of the manifest
#
# The parametrization and its markers both come from `capabilities.toml`: an
# axis whose state reaches `v1-default` needs its debt row deleted, and until
# then the row supplies the xfail. That is the "generated from the manifest"
# direction, applied to the markers themselves rather than to a file on disk.
# ---------------------------------------------------------------------------


def _axis_parameters():
    """One parameter per axis, xfailed while its engine-default row is open.

    The `open_debt` call is written with the f-string inline rather than
    through a local, because `debt.debt_claims()` reads these ids out of the
    syntax tree: the constant head `DEBT-ENGINE-DEFAULT-` is what tells the
    meta-check that these six rows are claimed by a test.
    """
    book = debt.rows()
    parameters = []
    for name in capabilities.axes():
        marks = (
            [debt.open_debt(f"DEBT-ENGINE-DEFAULT-{name}")]
            if f"DEBT-ENGINE-DEFAULT-{name}" in book
            else []
        )
        parameters.append(pytest.param(name, marks=marks, id=name))
    return parameters


@pytest.mark.parametrize("axis", _axis_parameters())
def test_debt_engine_default_is_v1(axis):
    """This axis is what a user gets without asking for it.

    The manifest is the record, and it is the record on purpose: the flag
    default lives in `mace_launcher`, the CI configuration is generated from
    the manifest, and a flip that changed one without the other would leave
    the two disagreeing about what shipped.
    """
    capability = capabilities.load()[axis]
    assert capability.defaults_to_v1, (
        f"the {axis} axis is at {capability.state!r}; it defaults to v1 at "
        f"{sorted(capabilities.V1_BY_DEFAULT)}, and {capability.burn_step} is "
        f"the ticket that gets it there"
    )


def test_every_axis_has_an_engine_default_row_or_none_at_all():
    """No axis may quietly lack a debt row while still defaulting to legacy.

    Without this, deleting a row is a way to make an axis stop being counted
    rather than a way to record that it was migrated. The two directions are
    both covered: an open axis must have a row, and a migrated one must not.
    """
    book = debt.rows()
    for name, capability in capabilities.axes().items():
        debt_id = f"DEBT-ENGINE-DEFAULT-{name}"
        if capability.defaults_to_v1:
            assert debt_id not in book, (
                f"{name} is at {capability.state!r} but {debt_id} is still a "
                f"row in the debt book; delete the row"
            )
        else:
            assert debt_id in book, (
                f"{name} is at {capability.state!r} and owes a default flip, "
                f"but there is no {debt_id} row in the debt book"
            )


# ---------------------------------------------------------------------------
# TorchScript on the legacy path
# ---------------------------------------------------------------------------


@debt.open_debt("DEBT-JIT-MODULES")
def test_debt_no_compile_mode_in_legacy_modules():
    """`mace/modules/` carries no TorchScript decorator any more."""
    modules = REPO_ROOT / "mace" / "modules"
    if not modules.is_dir():
        return
    problems, _ = v1_surface.scan(v1_surface.torchscript_violations, [modules])
    compile_modes = [problem for problem in problems if "@compile_mode" in problem]
    assert not compile_modes, (
        f"{len(compile_modes)} @compile_mode decorators remain under "
        f"mace/modules/"
    )


@debt.open_debt("DEBT-JIT-CALCULATORS")
def test_debt_no_lammps_calculators():
    """Neither legacy LAMMPS calculator is in the tree."""
    surviving = [
        path
        for path in (
            "mace/calculators/lammps_mace.py",
            "mace/calculators/lammps_mliap_mace.py",
        )
        if (REPO_ROOT / path).is_file()
    ]
    assert not surviving, f"the legacy LAMMPS calculators remain: {surviving}"


@debt.open_debt("DEBT-JIT-EXPORT")
def test_debt_no_torchscript_lammps_export():
    """No call site in the tree scripts a model for LAMMPS."""
    export = REPO_ROOT / "mace" / "cli" / "create_lammps_model.py"
    assert not export.is_file(), (
        "mace/cli/create_lammps_model.py still scripts the model with "
        "e3nn.util.jit.compile, and TorchScript is banned under packages/, so "
        "that call site can never be pointed at a v1 model"
    )


@debt.open_debt("DEBT-COMPILED-ARTIFACT-BEST-EFFORT")
def test_debt_no_best_effort_compiled_artifact():
    """Training no longer emits a side artifact that may silently not appear.

    The two emissions sit inside `except Exception`, so a scripting failure
    leaves no artifact and a zero exit status. v1 emits no side artifact at
    all; a deployment bundle is produced by a step that fails when it fails.
    """
    run_train = REPO_ROOT / "mace" / "cli" / "run_train.py"
    if not run_train.is_file():
        return
    assert not _calls_jit_compile(run_train), (
        "mace/cli/run_train.py still calls jit.compile inside an except "
        "Exception handler, so a run whose _compiled.model could not be "
        "written still exits 0 with the artifact absent"
    )


# ---------------------------------------------------------------------------
# The gap on the v1 side
# ---------------------------------------------------------------------------


@debt.open_debt("DEBT-V1-NO-DEPLOYMENT")
def test_debt_v1_has_no_deployment_path():
    """`mace_torch` ships a deployment surface of its own.

    The only row that is a hole in v1 rather than a leftover of legacy. v1
    blocks are born without `@compile_mode`, so the legacy `e3nn.util.jit`
    export cannot script them, and the `--engine v1` opt-in therefore excludes
    LAMMPS export and the compiled checkpoint for as long as this is open.
    """
    deploy = REPO_ROOT / "packages" / "mace-torch" / "src" / "mace_torch" / "deploy"
    assert deploy.is_dir(), "packages/mace-torch/src/mace_torch/deploy/ does not exist"
    modules = [path.name for path in deploy.glob("*.py") if path.name != "__init__.py"]
    assert modules, (
        f"{deploy.relative_to(REPO_ROOT)} exists but is empty of modules, so "
        f"there is still no export entry point"
    )


# ---------------------------------------------------------------------------
# The book itself
# ---------------------------------------------------------------------------


def test_the_debt_book_is_valid():
    """Schema, resolvable fitness tests, burn-step tickets, both directions.

    The same checks `check_debt_book.py` runs. They are here as well because
    the architecture job gates every pull request and the script is a
    convenience, not a second opinion.
    """
    problems = debt.problems()
    assert not problems, "\n".join(f"- {problem}" for problem in problems)


def test_every_row_names_a_fitness_test_that_exists():
    """The acceptance criterion, stated on its own so it fails by name.

    A row pointing at a test nobody wrote reads as coverage, which is worse
    than a row that admits there is none.
    """
    defined = debt.defined_test_functions()
    missing = {
        debt_id: row.fitness_function
        for debt_id, row in debt.rows().items()
        if row.fitness_function not in defined
    }
    assert not missing, f"debt rows whose fitness test does not exist: {missing}"


def test_the_ledger_has_no_date_column():
    """Deliberately, and it is the kind of thing a later edit would add back.

    A date here would be a promise to nobody: `main` stays on v0.3 until v1
    ships and no end of life is announced. The trigger is the burn-step
    ticket, and the mechanism is `strict=True`.
    """
    columns = [column.lower() for column in debt.header()]
    assert not [column for column in columns if "date" in column or "due" in column], (
        f"the ledger grew a date column: {debt.header()}"
    )


def test_an_unknown_debt_id_is_an_error_rather_than_a_no_op():
    """The failure mode of a decorator that reads a file.

    A burn-step pull request that deletes the row and forgets the decorator
    must not leave `open_debt` degrading quietly to a no-op: the test would
    then be unmarked while still asserting a state the tree has not reached,
    and it would be red for a reason nobody could place.
    """
    # Built rather than written as a literal: `debt.debt_claims()` reads
    # open_debt ids out of this file's syntax tree, and a literal here would
    # be reported as a marker claiming a row that does not exist.
    unknown = "-".join(["DEBT", "THAT", "WAS", "NEVER", "WRITTEN"])
    with pytest.raises(LookupError, match="not a row"):
        debt.open_debt(unknown)


# ---------------------------------------------------------------------------
# The mechanism itself
#
# Everything above trusts that `xfail(strict=True)` turns an unexpected pass
# into a failure. That is the whole gate, and it is a property of the pytest
# configuration this repository actually runs under, not of the marker in the
# abstract: `xfail_strict` in an ini file, or a plugin, could change the
# answer. So it is exercised, in a subprocess, against a config file written
# for the purpose -- the repository's own conftest imports torch, which a test
# of a marker has no business paying for.
# ---------------------------------------------------------------------------

MECHANISM_INI = "[pytest]\naddopts = --strict-markers\n"

MECHANISM_TEST = '''
import pytest


@pytest.mark.xfail(strict=True, reason="an open debt")
def test_debt_still_open():
    assert False, "the debt is still there"


@pytest.mark.xfail(strict=True, reason="an open debt")
def test_debt_already_burned():
    assert True
'''


def _run_pytest_in(directory: Path, *arguments: str):
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    (directory / "pytest.ini").write_text(MECHANISM_INI, encoding="utf-8")
    (directory / "test_mechanism.py").write_text(MECHANISM_TEST, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(directory / "pytest.ini"),
            "--rootdir",
            str(directory),
            *arguments,
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_debt_that_is_still_open_is_green(tmp_path):
    """The everyday state: the assertion fails, and the suite is fine with it."""
    result = _run_pytest_in(tmp_path, "-q", "test_mechanism.py::test_debt_still_open")
    assert result.returncode == 0, result.stdout
    assert "1 xfailed" in result.stdout


def test_a_debt_that_has_been_burned_turns_the_row_red(tmp_path):
    """The gate. An unexpected pass is a failure, not a quiet xpass.

    This is what "the burn-step ticket closed while the row is still open"
    looks like from inside the suite, and it is the reason the ledger needs no
    date column and no call to GitHub.
    """
    result = _run_pytest_in(
        tmp_path, "-q", "test_mechanism.py::test_debt_already_burned"
    )
    assert result.returncode != 0, (
        f"a strict xfail that passed did not fail the run; the whole overdue "
        f"mechanism is inert.\n{result.stdout}"
    )
    assert "XPASS(strict)" in result.stdout, result.stdout


def test_the_marker_this_module_applies_is_strict():
    """The decorator has to carry `strict=True`, or the gate above is unused."""
    marker = debt.open_debt("DEBT-JIT-MODULES").mark
    assert marker.name == "xfail"
    assert marker.kwargs.get("strict") is True
    assert "DEBT-JIT-MODULES" in marker.kwargs["reason"]
