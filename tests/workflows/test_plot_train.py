"""`mace_plot_train` against a real results log.

The plotting CLI had no test at all, and the results log format it parses had
none either, so nothing connected the two: `mace_run_train` is free to rename a
column and the only symptom is a plot nobody generates in CI.

That absence was hiding a CLI that could not run. `plot` aggregated every column
of the log with `mean` and `std`, including `head`, which is a string. pandas used
to drop such columns from an aggregation by itself; 3.0 raises `dtype 'str' does
not support operation 'mean'`, and `pandas` is unpinned, so the command has been
dead on any recent install. It is fixed in this branch, and these tests are what
would have caught it.

The log is produced by an actual training rather than hand-written, because a
hand-written one pins the test's idea of the format instead of the trainer's.
Every flag is checked by the plot changing, not merely by the command exiting
zero: the output is byte-deterministic here, so "this flag does something" is a
real assertion, and a flag silently ignored is a live risk (`--start_stage_two`
and `--start_swa` are two spellings that must stay one flag).
"""

import json
import shutil
from pathlib import Path

import ase.io
import pytest

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_TRAIN = REPO_ROOT / "mace" / "cli" / "plot_train.py"

#: `plot` writes `{name}_{head}.{format}` next to the working directory, and
#: `name` comes off the log's filename, not from a flag.
DEFAULT_PLOT = "plotme_default.png"


@pytest.fixture(name="results_log", scope="module")
def fixture_results_log(tmp_path_factory):
    """A real `<name>_run-<seed>_train.txt`, from a real six-epoch training."""
    tmp = tmp_path_factory.mktemp("plot_train")
    ase.io.write(tmp / "fit.xyz", make_fitting_configs())
    params = base_mace_params()
    params.update(
        {
            "name": "plotme",
            "hidden_irreps": "16x0e",
            "checkpoints_dir": str(tmp),
            "model_dir": str(tmp),
            "results_dir": str(tmp),
            "log_dir": str(tmp),
            "train_file": str(tmp / "fit.xyz"),
            "max_num_epochs": 6,
            "start_swa": 4,
            "seed": 7,
        }
    )
    run_mace_train(params)
    log = tmp / "plotme_run-7_train.txt"
    assert log.exists(), "the trainer did not write the results log this CLI reads"
    return log


@pytest.fixture(name="workdir")
def fixture_workdir(results_log, tmp_path):
    """A directory holding only the log, since the CLI writes into the cwd."""
    shutil.copy(results_log, tmp_path / results_log.name)
    return tmp_path


def plot_train(workdir, *argv):
    """Run the CLI in `workdir` and return the bytes of the plot it wrote."""
    run_mace_train(
        {"path": str(workdir / "plotme_run-7_train.txt")},
        extra_argv=list(argv),
        script=PLOT_TRAIN,
        cwd=workdir,
        env_extra={"MPLBACKEND": "Agg"},
    )
    return workdir


def read(workdir, name=DEFAULT_PLOT):
    path = workdir / name
    assert path.exists(), f"the CLI wrote no {name}; got {sorted(p.name for p in workdir.iterdir())}"
    return path.read_bytes()


# ---------------------------------------------------------------------------
# The log the CLI reads
# ---------------------------------------------------------------------------


def test_the_results_log_carries_the_columns_the_plot_reads(results_log):
    """The format contract, stated where a rename would break it."""
    rows = [json.loads(line) for line in results_log.read_text().splitlines()]

    assert rows, "the results log is empty"
    assert {"epoch", "mode", "loss", "head"} <= set(rows[0])
    assert {"opt", "eval"} <= {row["mode"] for row in rows}
    assert any(row.get("rmse_e") is not None for row in rows)


def test_the_log_carries_a_string_column(results_log):
    """`head` is why the aggregation needs the numeric columns selected. A test
    that only ever saw numeric columns would pass against the broken version."""
    rows = [json.loads(line) for line in results_log.read_text().splitlines()]

    assert any(isinstance(row.get("head"), str) for row in rows)


# ---------------------------------------------------------------------------
# The plot
# ---------------------------------------------------------------------------


def test_a_real_results_log_becomes_a_plot(workdir):
    plot_train(workdir)

    content = read(workdir)
    assert content[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert len(content) > 5_000, "a PNG this small is an empty figure"


def test_a_directory_is_searched_for_logs(workdir):
    """`--path` takes a directory as well as a file, and globs `*_train.txt`."""
    run_mace_train(
        {"path": str(workdir)},
        script=PLOT_TRAIN,
        cwd=workdir,
        env_extra={"MPLBACKEND": "Agg"},
    )

    assert (workdir / DEFAULT_PLOT).exists()


def test_the_same_input_gives_the_same_plot(workdir):
    """The premise the flag tests below rely on."""
    first = read(plot_train(workdir))
    (workdir / DEFAULT_PLOT).unlink()
    second = read(plot_train(workdir))

    assert first == second


def test_the_output_format_is_honoured(workdir):
    plot_train(workdir, "--output_format", "pdf")

    content = read(workdir, "plotme_default.pdf")
    assert content[:5] == b"%PDF-"


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param(["--linear"], id="linear"),
        pytest.param(["--error_bars"], id="error_bars"),
        pytest.param(["--keys", "rmse_e"], id="keys"),
        pytest.param(["--min_epoch", "3"], id="min_epoch"),
        pytest.param(["--start_swa", "4"], id="start_swa"),
    ],
)
def test_each_flag_changes_the_plot(workdir, flag):
    """Not "it exits zero": a flag that is parsed and then dropped would."""
    baseline = read(plot_train(workdir))
    (workdir / DEFAULT_PLOT).unlink()

    assert read(plot_train(workdir, *flag)) != baseline


def test_start_stage_two_is_the_same_flag_as_start_swa(workdir):
    """Two spellings, one `dest`. They must produce the same figure, or the
    rename left one of them landing somewhere else."""
    swa = read(plot_train(workdir, "--start_swa", "4"))
    (workdir / DEFAULT_PLOT).unlink()
    stage_two = read(plot_train(workdir, "--start_stage_two", "4"))

    assert swa == stage_two


def test_heads_names_the_plot_after_the_head(workdir):
    """The multihead path aggregates per head and writes one file per head."""
    plot_train(workdir, "--heads", "Default")

    assert (workdir / "plotme_Default.png").exists()


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


def test_a_filename_it_cannot_parse_is_refused(workdir):
    """The seed and the run name come from the filename, so an unparseable one
    has to fail rather than be plotted under a guessed name."""
    stray = workdir / "not-a-results-log.txt"
    stray.write_text('{"epoch": 1, "mode": "opt", "loss": 1.0}\n')

    result = run_mace_train(
        {"path": str(stray)},
        script=PLOT_TRAIN,
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
        env_extra={"MPLBACKEND": "Agg"},
    )

    assert result.returncode != 0
    assert "Cannot parse" in result.stderr


def test_a_directory_without_logs_is_refused(workdir, tmp_path_factory):
    empty = tmp_path_factory.mktemp("no_logs")

    result = run_mace_train(
        {"path": str(empty)},
        script=PLOT_TRAIN,
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
        env_extra={"MPLBACKEND": "Agg"},
    )

    assert result.returncode != 0
    assert "Cannot find results" in result.stderr
