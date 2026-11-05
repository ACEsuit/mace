"""PAR-2 nightly campaign; failures are replayable without Hypothesis search."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from mace_torch.backends.reference import ReferenceBackend

from tests.parity.fuzz_cases import INPUT_CLASSES, _case, strategies
from tests.parity.fuzz_parity import check_case, compare, save_case

pytestmark = [pytest.mark.nightly, pytest.mark.slow]
CAMPAIGN = strategies(ReferenceBackend().capabilities().max_lmax)


def require_coverage(counts):
    missing = [name for name in INPUT_CLASSES if counts.get(name, 0) == 0]
    assert not missing, f"PAR-2 input classes not executed: {missing}"


@pytest.fixture(scope="module")
def campaign(request):
    root = Path(os.environ.get("MACE_PAR2_ARTIFACT_DIR", ".pytest_cache/par2"))
    root.mkdir(parents=True, exist_ok=True)
    seed = request.config.getoption("hypothesis_seed", default=None)
    counts = dict.fromkeys(INPUT_CLASSES, 0)
    variants = {name: set() for name in INPUT_CLASSES}
    errors = {}
    yield root, seed, counts, variants, errors
    record = {
        "hypothesis_seed": seed,
        "executed": counts,
        "variants": {key: sorted(value) for key, value in variants.items()},
        "max_abs_error": errors,
    }
    (root / "coverage.json").write_text(json.dumps(record, indent=2) + "\n")
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_sep("=", "PAR-2 executed examples")
        reporter.write_line(json.dumps(record, sort_keys=True))
    require_coverage(counts)
    assert variants["effective_cells"] == {"aperiodic", "physical", "zero_slab"}
    assert variants["mixed_precision"] == {
        "float32/float32",
        "float32/float64",
        "float64/float64",
    }
    maximum = ReferenceBackend().capabilities().max_lmax or 4
    assert variants["high_lmax"] == {str(max(2, maximum - 1)), str(maximum)}
    # The artifact assertion is about serialized inputs produced by this run,
    # not merely a nonempty directory or a database Hypothesis might not use.
    for name in INPUT_CLASSES:
        witness = json.loads((root / f"{name}.json").read_text())
        assert witness["case"]["input_class"] == name


@pytest.mark.parametrize("input_class", INPUT_CLASSES)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_fuzz_parity(input_class, campaign, data):
    root, seed, counts, variants, errors = campaign
    case = data.draw(CAMPAIGN[input_class], label=input_class)
    measured = check_case(case, root, seed)
    counts[input_class] += 1
    variants[input_class].add(
        f"{case.dtype}/{case.constants_dtype}"
        if input_class == "mixed_precision"
        else str(case.lmax)
        if input_class == "high_lmax"
        else case.regime
    )
    for quantity, error in measured.items():
        key = f"{case.dtype}/{quantity}"
        errors[key] = max(errors.get(key, 0), error)
    if counts[input_class] == 1:
        save_case(root / f"{input_class}.json", case, seed)


@pytest.mark.parametrize("disabled", INPUT_CLASSES)
def test_coverage_counters_reject_a_disabled_strategy(disabled):
    counts = dict.fromkeys(INPUT_CLASSES, 0)
    for name in CAMPAIGN:
        if name != disabled:
            counts[name] += 1
    with pytest.raises(AssertionError, match=disabled):
        require_coverage(counts)


def test_coverage_counters_accept_all_six():
    require_coverage(dict.fromkeys(INPUT_CLASSES, 1))


@pytest.mark.parametrize("perturbation", ["drop_self_images", "physical_open_cell"])
def test_weak_input_failure_is_caught_and_replays(tmp_path, perturbation):
    if perturbation == "drop_self_images":
        case = _case(
            "periodic_self_images", 0, count=1, cell=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        )
    else:
        case = _case("effective_cells", 0, pbc=[False] * 3, regime="aperiodic")
    compare(case)
    root = tmp_path / ".examples"
    with pytest.raises(AssertionError, match=f"--perturbation {perturbation}"):
        check_case(case, root, 0, perturbation=perturbation)
    (path,) = root.glob("failure-*.json")
    command = [sys.executable, "-m", "tests.parity.fuzz_parity", str(path)]
    good = subprocess.run(command, capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stdout + good.stderr
    bad = subprocess.run(
        [*command, "--perturbation", perturbation],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0 and "AssertionError" in bad.stderr
    assert "Hypothesis seed: 0" in bad.stdout
    assert path.is_file() and path.stat().st_size > 0


def test_high_lmax_strategy_obeys_finite_capabilities():
    from hypothesis import find

    from tests.parity.fuzz_cases import high_lmax

    for maximum in (2, 3, 4):
        case = find(
            high_lmax(max_lmax=maximum), lambda case, cap=maximum: case.lmax == cap
        )
        assert case.lmax == maximum
    with pytest.raises(ValueError, match="lmax >= 2"):
        find(high_lmax(max_lmax=1), lambda case: True)


def test_nightly_upload_keeps_hidden_replays_and_other_jobs_exclude_campaign():
    import yaml

    workflows = Path(__file__).resolve().parents[2] / ".github/workflows"
    nightly = yaml.safe_load((workflows / "nightly.yaml").read_text())["jobs"]
    job = nightly["parity-fuzz"]
    upload = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["include-hidden-files"] is True
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"] == ".pytest_cache/par2"
    for name in ("coverage-full", "durations-refresh"):
        run = next(
            step
            for step in nightly[name]["steps"]
            if step.get("uses") == "./.github/actions/run-tests"
        )
        assert "not nightly" in run["with"]["markers"]
    core = yaml.safe_load((workflows / "ci-core.yaml").read_text())["jobs"]
    parity = next(
        step
        for step in core["parity"]["steps"]
        if step.get("uses") == "./.github/actions/run-tests"
    )
    assert "not nightly" in parity["with"]["markers"]
