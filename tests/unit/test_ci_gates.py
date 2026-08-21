"""The CI gates that cannot check themselves.

Everything this file guards failed the same way once: green, and measuring
nothing.

* The nightly `benchmarks` job ran a directory whose only test was marked
  `gpu` AND `network`, on a CPU runner with no network. All sixteen
  parametrizations skipped, the job passed, and the uploaded benchmark.json
  was reproducibly a **0-byte file**. Nothing said so -- and note which guard
  does *not* help here: pytest-benchmark creates the `--benchmark-json` path
  whatever happens, so `if-no-files-found` on the upload never fires. Only a
  check on the artifact's contents can tell a baseline from a green job that
  measured nothing, which is why the tests below run the job's own check
  against artifacts built to be empty, holed and unlabelled.
* A per-file coverage floor named after a module that has been renamed or
  moved selects no files. `coverage report --include=<gone> --fail-under=N`
  does exit 1 on "No data to report", so the nightly would catch it -- but a
  night later, and with a message that reads like a broken coverage run
  rather than a floor whose subject moved. The floors are explicitly supposed
  to migrate to `mace_core`/`mace_torch` as capabilities port, so the
  rename-without-moving-the-floor case is the expected one, not a freak.
* A capability whose tests exist but which no job names in `require-caps` is
  the skip-o-fail contract with nothing on the other end: every test that
  needs it skips in every job, and the suite reports success by absence. That
  is how the `--wandb` flags stayed untested while a test file for them was
  in the tree -- the client is an extra, nothing installed it, and the file
  skipped everywhere.

These live in tests/unit rather than beside what they describe because they
belong in a PR-gating job: tests/benchmarks is nightly-only by construction,
so a guard placed there would find the hole a day after it was dug.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NIGHTLY = REPO_ROOT / ".github" / "workflows" / "nightly.yaml"
GITLAB_PIPELINE = REPO_ROOT / ".github" / "gitlab" / "ci.yml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

pytestmark = pytest.mark.skipif(
    not NIGHTLY.is_file(), reason="workflow definitions are not part of an install"
)


def _nightly() -> dict:
    # `on:` is parsed by PyYAML 1.1 rules as the boolean True; harmless here,
    # nothing below reads it.
    return yaml.safe_load(NIGHTLY.read_text(encoding="utf-8"))


def _steps(job: str) -> List[dict]:
    return _nightly()["jobs"][job]["steps"]


def _floor_table() -> Dict[str, int]:
    """The floors, read back out of the job definition that enforces them.

    Parsing the shell heredoc rather than importing a table from somewhere is
    deliberate. The floors have to be readable in the job that enforces them,
    next to the reasoning for why those files and not others -- a reviewer
    should not have to open a second file to see what the gate is. So the job
    definition is the only copy, and this test reads that one rather than
    creating a second that could drift from it.
    """
    runs = "\n".join(step.get("run", "") for step in _steps("coverage-report"))
    body = re.search(r"<<'FLOORS'\n(.*?)\n\s*FLOORS\b", runs, re.DOTALL)
    assert body is not None, "the coverage-report job has no FLOORS heredoc"
    table: Dict[str, int] = {}
    for line in body.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, floor = line.split()
        table[path] = int(floor)
    return table


# ---------------------------------------------------------------------------
# Coverage floors
# ---------------------------------------------------------------------------


def test_every_floor_names_a_file_that_exists():
    """A floor whose module moved must move with it, not silently stop biting.

    `coverage report --include` takes an fnmatch pattern, so a stale path is
    not an error at parse time -- it just selects nothing.
    """
    table = _floor_table()
    assert table, "the floor table is empty"
    missing = [path for path in table if not (REPO_ROOT / path).is_file()]
    assert not missing, (
        f"coverage floors name files that no longer exist: {missing}. If the "
        f"behaviour moved to mace_core/mace_torch, move the floor with it -- "
        f"the floor protects the behaviour, not the legacy file."
    )


def test_every_floor_is_a_usable_percentage():
    for path, floor in _floor_table().items():
        assert 0 < floor <= 100, f"{path}: {floor} is not a coverage percentage"


def test_the_floors_are_enforced_after_the_shards_are_combined():
    """Not inside a shard, and not in the PR job -- both measure something else.

    A shard holds a third of the suite, so a module exercised entirely by
    tests that landed in another group would fail its own floor; the ci-core
    coverage job measures only tests/unit + tests/workflows, a different
    denominator for the same number.
    """
    report_runs = "\n".join(step.get("run", "") for step in _steps("coverage-report"))
    assert "coverage combine" in report_runs
    assert report_runs.index("coverage combine") < report_runs.index("--fail-under")

    for job in ("coverage-full", "benchmarks", "workflows-full"):
        runs = "\n".join(step.get("run", "") for step in _steps(job))
        assert "--fail-under" not in runs, f"a coverage floor leaked into {job}"

    core = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci-core.yaml").read_text(
            encoding="utf-8"
        )
    )
    core_runs = "\n".join(
        step.get("run", "") for step in core["jobs"]["coverage"]["steps"]
    )
    assert "--fail-under" not in core_runs, (
        "the per-PR coverage job is informative by design: it measures the "
        "deterministic PR slice, which is a different denominator"
    )


def test_the_floors_carry_their_reasoning_in_the_job_that_enforces_them():
    """A bare list of percentages is not maintainable by anyone but its author.

    Three things have to survive next to the numbers, because each is a
    question the next person will otherwise answer wrongly: why only these
    files (the rest is pinned by goldens and contracts, not line coverage),
    which selection the percentages are measured under (the same floor reads
    76 or 87 on `mace/modules/utils.py` depending on it), and what to do when
    a module moves to the new stack (the floor moves with it). Comments are
    not in the parsed YAML, so this reads the raw file.
    """
    text = NIGHTLY.read_text(encoding="utf-8")
    gate = text[text.index("THE coverage gate") : text.index("<<'FLOORS'")]
    for topic, needle in (
        ("why only these files", "WHY ONLY THESE FILES"),
        ("the migration invariant", "MIGRATION INVARIANT"),
        ("which selection is measured", "THE SELECTION THESE FLOORS ARE MEASURED"),
        ("why not in a shard", "shard"),
        ("why not in the PR job", "denominator"),
    ):
        assert needle in gate, (
            f"the floor gate no longer explains {topic}; the list of "
            f"percentages has to carry its reasoning or it cannot be "
            f"maintained, raised or migrated by anyone but whoever wrote it"
        )


def test_the_floor_gate_runs_only_when_every_shard_reported():
    """`coverage-report` must stay dependent on, and not independent of, the shards.

    Combining a subset of the shards produces a number that reads as the
    project's while missing whatever the failed shard covered -- and a floor
    computed on it would fail for the wrong reason.
    """
    job = _nightly()["jobs"]["coverage-report"]
    assert job.get("needs") == "coverage-full"
    assert "if" not in job, "coverage-report must not run on a partial shard set"


# ---------------------------------------------------------------------------
# Benchmark baseline
# ---------------------------------------------------------------------------


def test_every_declared_benchmark_size_has_a_cpu_runnable_case():
    from tests.benchmarks.test_inference_cpu import (  # noqa: PLC0415
        SYSTEM_SIZES,
        iter_case_marks,
    )

    runnable = {
        regime
        for regime, marks in iter_case_marks()
        if not marks & {"gpu", "network", "cueq", "oeq"}
    }
    missing = set(SYSTEM_SIZES) - runnable
    assert not missing, (
        f"no benchmark case a CPU-only nightly can run at sizes "
        f"{sorted(missing)}: the published artifact would have a hole in it"
    )


def test_the_nightly_benchmarks_job_selects_the_benchmarks_and_can_fail():
    job = _nightly()["jobs"]["benchmarks"]
    assert not job.get("continue-on-error"), "a job that cannot fail is not a baseline"

    run_tests = [
        step
        for step in job["steps"]
        if str(step.get("uses", "")).endswith("actions/run-tests")
    ]
    assert len(run_tests) == 1
    step = run_tests[0]
    assert not step.get("continue-on-error"), (
        "continue-on-error on the run-tests step hides exactly the failure "
        "this baseline exists to notice"
    )
    with_ = step.get("with", {})
    assert "tests/benchmarks" in with_["tests"]
    # A marker expression here could only narrow the selection, and every
    # narrowing so far has ended with the artifact empty.
    assert "benchmark" not in with_.get("markers", "")
    assert "--benchmark-json" in with_.get("extra-args", "")
    # pytest-benchmark disables itself under xdist, so a parallel run of this
    # job would publish an artifact full of nothing.
    assert str(with_.get("numprocs")) == "0"
    # Half the baseline is downloaded; a lost network must be red, not half
    # an artifact.
    assert "network" in with_.get("require-caps", "")
    assert with_.get("allow-network") == "true"

    upload = [
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/upload")
    ]
    assert len(upload) == 1
    assert upload[0]["with"].get("if-no-files-found") == "error"


def _artifact_check_source() -> str:
    """The python body of the job's artifact content check, as text."""
    for step in _steps("benchmarks"):
        if "actually contain a baseline" in str(step.get("name", "")):
            body = re.search(r"<<'PY'\n(.*?)\n\s*PY", step["run"], re.DOTALL)
            assert body is not None, "the content check is not a PY heredoc"
            return body.group(1)
    raise AssertionError(
        "the benchmarks job has no artifact content check: without one the "
        "job is green whenever every case skips"
    )


def _run_artifact_check(tmp_path: Path, payload) -> int:
    """Run the job's own check, verbatim, against an artifact we construct."""
    (tmp_path / "benchmark.json").write_text(
        "" if payload is None else json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "check.py").write_text(_artifact_check_source(), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "check.py"], cwd=tmp_path, capture_output=True, check=False
    ).returncode


def _benchmark_entry(regime: str, **overrides) -> dict:
    info = {
        "regime": regime,
        "dtype": "float64",
        "device": "cpu",
        "torch_version": "2.11.0",
        "backend": "e3nn",
    }
    info.update(overrides)
    return {"name": f"test[{regime}]", "extra_info": info}


ALL_SIZES = ("subdomain216", "subdomain512", "kernel1728")


def test_an_empty_artifact_fails_the_benchmarks_job(tmp_path):
    """The exact regression: a session where every case skipped.

    `--benchmark-json` is created regardless of what ran, so that session
    leaves a **0-byte file** -- verified against a real run of
    tests/benchmarks/test_benchmark.py, whose sixteen parametrizations are all
    `gpu` and `network`. `if-no-files-found` therefore never fires, and only a
    check on the contents can tell a baseline from an empty artifact.
    """
    assert _run_artifact_check(tmp_path, None) != 0
    assert _run_artifact_check(tmp_path, {"benchmarks": []}) != 0


def test_a_size_hole_fails_the_benchmarks_job(tmp_path):
    """Skipping is per-parametrization, so "non-empty" is not enough.

    One surviving case would satisfy a bare emptiness check while the size
    spread the downstream comparison is judged in had quietly gone.
    """
    for dropped in ALL_SIZES:
        kept = [_benchmark_entry(r) for r in ALL_SIZES if r != dropped]
        assert _run_artifact_check(tmp_path, {"benchmarks": kept}) != 0, (
            f"an artifact missing the {dropped} size passed the check"
        )
    full = [_benchmark_entry(r) for r in ALL_SIZES]
    assert _run_artifact_check(tmp_path, {"benchmarks": full}) == 0


@pytest.mark.parametrize("field", ["dtype", "device", "torch_version", "backend"])
def test_a_result_without_its_metadata_fails_the_benchmarks_job(tmp_path, field):
    """A timing whose denominator is unrecorded is not a baseline.

    These four are what the frozen comparison reads back years from now, when
    the stack that produced the number no longer exists to be asked.
    """
    runs = [_benchmark_entry(r) for r in ALL_SIZES]
    runs[0]["extra_info"][field] = ""
    assert _run_artifact_check(tmp_path, {"benchmarks": runs}) != 0


def test_the_benchmark_cases_record_every_field_the_job_demands():
    """The producer and the checker must agree, or the job fails every night."""
    from tests.benchmarks import test_inference_cpu as cases  # noqa: PLC0415

    source = Path(cases.__file__).read_text(encoding="utf-8")
    for field in ("dtype=", "device=", "torch_version=", "backend=", "regime="):
        assert field in source, f"benchmark cases never record {field}"
    assert set(cases.SYSTEM_SIZES) == set(ALL_SIZES), (
        "the sizes the job demands and the sizes the cases declare have "
        "diverged: one of them will be a hole"
    )


def test_benchmarks_never_gate_a_correctness_job():
    """No correctness job may select them, and none may set a perf threshold."""
    coverage_full = _nightly()["jobs"]["coverage-full"]
    markers = [
        step.get("with", {}).get("markers", "")
        for step in coverage_full["steps"]
        if str(step.get("uses", "")).endswith("actions/run-tests")
    ]
    assert markers and all("not benchmark" in m for m in markers)

    pipeline = GITLAB_PIPELINE.read_text(encoding="utf-8")
    assert "and not benchmark" in pipeline, (
        "the GPU pipeline runs -n 2 on a contended shared runner under a 2h "
        "cap: a timing taken there is noise, and it is a correctness gate"
    )

    # pytest-benchmark only turns a timing into a pass/fail with an explicit
    # comparison flag. Nothing may pass one.
    for text in (NIGHTLY.read_text(encoding="utf-8"), pipeline):
        assert "--benchmark-compare-fail" not in text


def _setup_step(job: str) -> dict:
    """The setup-mace step of a job, which decides what is installed."""
    for step in _steps(job):
        if str(step.get("uses", "")).endswith("setup-mace"):
            return step.get("with", {})
    raise AssertionError(f"job {job!r} has no setup-mace step")


def _run_tests_step(job: str) -> dict:
    for step in _steps(job):
        if str(step.get("uses", "")).endswith("run-tests"):
            return step.get("with", {})
    raise AssertionError(f"job {job!r} has no run-tests step")


def test_the_coverage_job_installs_every_capability_it_requires():
    """A required capability the install cannot provide is a floor that cannot be met.

    coverage-full is where the project's real coverage number is produced, and
    the per-file floors are enforced against it. It once required `polar` and
    `les` -- which arrive through pip-packages rather than extras -- while the
    `magnetic` extra was absent entirely. Every test under
    tests/extensions/magnetic therefore skipped in the one job whose numbers
    the floors judge, and the mace/modules/utils.py floor counts
    compute_forces_virials_magforces and compute_forces_magforces, reachable
    from nowhere else. That is ~8 points of a file with 2 points of headroom:
    the floor could not have been met, and the failure would have read as a
    coverage regression rather than a missing install.
    """
    setup = _setup_step("coverage-full")
    required = {
        cap.strip()
        for cap in str(_run_tests_step("coverage-full")["require-caps"]).split(",")
        if cap.strip()
    }
    # a capability is satisfiable if it is an extra, or arrives via pip-packages
    provided = {e.strip() for e in str(setup.get("extras", "")).split(",") if e.strip()}
    pip = str(setup.get("pip-packages", ""))
    # `network` is not a package; it is the allow-network switch
    unprovidable = {
        cap
        for cap in required - provided - {"network"}
        if f"requirements/{cap}.txt" not in pip
    }
    assert not unprovidable, (
        f"coverage-full requires {sorted(unprovidable)} but installs neither the "
        f"extra nor requirements/<cap>.txt for it. Its tests will skip, and any "
        f"floor counting code they reach becomes unmeetable."
    )
    assert str(_run_tests_step("coverage-full")["allow-network"]) == "true", (
        "coverage-full requires the network capability, so it must allow network"
    )


# ---------------------------------------------------------------------------
# Capabilities and the jobs that guarantee them
# ---------------------------------------------------------------------------

_CAPS_KEYS = ("require-caps", "MACE_REQUIRE_CAPS")


def _guaranteed_capabilities() -> Set[str]:
    """Every capability some pipeline promises to have installed.

    A recursive walk of the workflows rather than a per-job reader, because the
    promise is spelled two ways there: the `require-caps` input of the run-tests
    action, and a plain `MACE_REQUIRE_CAPS` env on a job that calls pytest
    itself. The GitLab pipeline is read as text instead -- it uses GitLab's
    `!reference` tag, which `yaml.safe_load` refuses -- so only its quoted
    `REQUIRE_CAPS` values are matched, never the prose that mentions the name.

    The other direction needs no test here: a required capability that is not a
    capability at all is a `pytest.UsageError` from `pytest_configure`, so the
    job that misspells one fails on its first collection.
    """
    found: Set[str] = set()

    def record(value: str) -> None:
        found.update(cap.strip() for cap in value.split(",") if cap.strip())

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _CAPS_KEYS and isinstance(value, str):
                    record(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for path in sorted(WORKFLOWS.glob("*.yaml")) + sorted(WORKFLOWS.glob("*.yml")):
        walk(yaml.safe_load(path.read_text(encoding="utf-8")))

    pipeline = GITLAB_PIPELINE.read_text(encoding="utf-8")
    for match in re.finditer(r'REQUIRE_CAPS:\s*"([^"]*)"', pipeline):
        record(match.group(1))
    return found


def test_every_capability_is_guaranteed_by_some_job():
    """A capability no job requires can only ever skip, and skips read as green.

    The probe is the point: locally an absent optional dependency has to skip.
    What makes the skip a hole is a capability that is *never* required
    anywhere, because then no run distinguishes "this feature works" from "this
    feature was never installed".
    """
    from tests.conftest import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        CAPABILITY_PROBES,
    )

    orphans = sorted(set(CAPABILITY_PROBES) - _guaranteed_capabilities())
    assert not orphans, (
        f"no CI job guarantees {orphans} via require-caps / MACE_REQUIRE_CAPS, "
        f"so every test marked with one of them skips in every job and the "
        f"suite is green by absence. Either give the capability a job that "
        f"installs it and names it, or drop the marker."
    )


# ---------------------------------------------------------------------------
# the cueq wheel extras
#
# The `cueq*` extras name wheels the suite cannot assert anything about: the only
# failure mode of an extras group is a job resolving it, and no test inside the
# suite can resolve the suite's own dependencies. The GPU pipeline installs
# cueq-cuda-13 and nothing installs the rest, so the job is the only place the
# others are checked at all.
#
# The job derives its list from setup.cfg rather than naming the extras, and
# these tests hold it to that: an extras group added or retired there is then
# covered without an edit in the workflow, and neither file can drift from the
# other while nobody is looking.
# ---------------------------------------------------------------------------


def _declared_cueq_extras() -> list:
    import configparser  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    config = configparser.ConfigParser()
    config.read(REPO_ROOT / "setup.cfg", encoding="utf-8")
    return [
        name
        for name in config["options.extras_require"]
        if name.startswith("cueq")
    ]


def _cueq_job() -> dict:
    """The job is a PR gate rather than a nightly one: resolving a handful of
    extras is seconds, so there is no reason to make a contributor wait until
    tomorrow to learn that an extras group they touched no longer resolves."""
    core = yaml.safe_load(
        (WORKFLOWS / "ci-core.yaml").read_text(encoding="utf-8")
    )
    return core["jobs"]["cueq-wheel-extras"]


def _cueq_job_script() -> str:
    return "\n".join(step.get("run", "") for step in _cueq_job()["steps"])


def test_a_pr_job_resolves_the_cueq_extras():
    core = yaml.safe_load((WORKFLOWS / "ci-core.yaml").read_text(encoding="utf-8"))

    assert "cueq-wheel-extras" in core["jobs"]
    assert "cueq-wheel-extras" not in _nightly()["jobs"], (
        "one copy only, and the PR gate is the useful one"
    )


def test_setup_cfg_declares_the_extras_the_job_resolves():
    """setup.cfg is the list, and there has to be one to read: a section rename
    would leave the loop iterating over nothing and the job passing on air."""
    declared = _declared_cueq_extras()

    assert "cueq" in declared
    assert [name for name in declared if name.startswith("cueq-cuda-")], (
        "no CUDA-major ops extra is declared, so the job resolves the frontend only"
    )


def test_the_job_fails_when_the_derived_list_is_empty():
    """The failure mode a derived list adds: an empty `for` runs no pip, leaves
    `status` at 0, and reports green having resolved nothing. Reading setup.cfg
    from the test proves the file is fine, not that the job's own parse found
    anything, so the guard has to live in the script."""
    script = _cueq_job_script()

    assert '-z "$extras"' in script, "an empty derivation is not caught"
    guard = script.split('-z "$extras"', 1)[1].split("fi", 1)[0]
    assert "exit 1" in guard, "the empty case is reported but not failed"


def test_the_job_reads_the_extras_out_of_setup_cfg():
    """Rather than naming them. A list written in the workflow goes stale in the
    direction nothing checks: an extra retired from setup.cfg leaves the job
    resolving a name pip only warns about, so it stays green while covering one
    fewer group than it claims."""
    script = _cueq_job_script()

    assert "options.extras_require" in script and "setup.cfg" in script
    # The bare `cueq` is exempt: it is the prefix the derivation filters on, so
    # the string is in the script either way. The per-major names are the ones
    # that would be a written-out list.
    named = [
        name
        for name in _declared_cueq_extras()
        if name.startswith("cueq-cuda-") and name in script
    ]
    assert not named, (
        f"the job names {named} instead of deriving them, so setup.cfg and the "
        f"workflow can disagree"
    )


def test_the_job_needs_no_gpu():
    """The reason this can gate a PR at all: the ops wheels are manylinux
    artifacts, so resolving them wants no device. If it ever moves to a GPU
    runner, the rows that point at it are pointing somewhere much more expensive.
    """
    assert _cueq_job()["runs-on"] == "ubuntu-latest"


def test_the_job_names_no_platform_tag():
    """pip matches a platform tag exactly when it is given one, so naming
    `manylinux_2_17` resolves nothing for the packages that ship
    `manylinux_2_28`, and naming several resolves nothing at all. The runner's
    own tags are the only set that is right for every one of these wheels.
    """
    assert "--platform" not in _cueq_job_script()


def test_the_job_only_resolves_and_does_not_install():
    """`--dry-run` is the whole assertion: nothing here needs the wheels on
    disk, only proof that a resolver can find them."""
    assert "--dry-run" in _cueq_job_script()


def test_a_failure_to_resolve_fails_the_job():
    """The loop must not swallow the failure: a green job that resolved nothing is
    the state these rows were already in.
    """
    script = _cueq_job_script()

    assert "status=1" in script
    assert 'exit "$status"' in script
