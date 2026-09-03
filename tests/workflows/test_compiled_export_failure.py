"""What a run says when the compiled export fails.

`run_train` saves the model twice: the pickled checkpoint, then a TorchScript
copy next to it under `<name>_compiled.model`. The second one is wrapped in
`try/except Exception`, which is right -- a model TorchScript cannot take is not
a reason to lose a finished training -- but the handler was a bare `pass`, three
lines after an INFO line announcing the file. So the failure mode was: the log
says the file is being written, no file appears, and nothing anywhere says why.

The failure is forced here rather than simulated, by putting a directory where
the file goes: the pickled save succeeds, `torch.jit.save` cannot open its
target, and the branch under test is the real one.
"""

import ase.io
import pytest

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train


def train(tmp_path, blocked: bool):
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    if blocked:
        # Where `torch.jit.save` wants to write its file. `torch.save` of the
        # pickled model uses a different name, so it still succeeds and the run
        # reaches the compiled step with everything else intact.
        (tmp_path / "ce_compiled.model").mkdir()
    params = base_mace_params()
    params.update(
        {
            "name": "ce",
            "hidden_irreps": "16x0e",
            "checkpoints_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "results_dir": str(tmp_path),
            "log_dir": str(tmp_path),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 1,
            "seed": 4,
        }
    )
    # The stage-two branch writes `_stagetwo_compiled.model` instead, and the
    # two handlers are separate copies of the same code; this exercises the
    # plain one.
    params.pop("swa", None)
    params.pop("start_swa", None)
    return run_mace_train(params, check=False, capture_output=True, text=True)


@pytest.fixture(name="blocked_run", scope="module")
def fixture_blocked_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("compiled_blocked")
    return train(work, blocked=True), work


def test_a_compiled_export_that_cannot_be_written_does_not_fail_the_run(blocked_run):
    done, _ = blocked_run
    """The reason the handler exists: the training is finished and the pickled
    model is on disk, so a TorchScript copy that cannot be produced is not worth
    losing it over."""
    assert done.returncode == 0, done.stderr[-3000:]


def test_the_failure_is_reported(blocked_run):
    done, _ = blocked_run
    """The defect. A bare `pass` left the announcement of the file as the last
    word on the subject."""
    assert "ce_compiled.model" in done.stdout
    assert "was not written" in done.stdout, done.stdout[-3000:]


def test_the_report_names_the_reason(blocked_run):
    done, _ = blocked_run
    """A warning that says only "it failed" sends the reader back to guessing
    between a model TorchScript rejects and a path it cannot open."""
    warnings = [
        line for line in done.stdout.splitlines() if "was not written" in line
    ]

    assert warnings, done.stdout[-2000:]
    assert any("WARNING" in line for line in warnings), warnings
    assert any(
        "Is a directory" in line or "directory" in line.lower() for line in warnings
    ), warnings


def test_the_pickled_model_is_still_there(blocked_run):
    """What the run does deliver, and the reason the warning is a warning."""
    _, work = blocked_run

    assert (work / "ce.model").is_file()
    assert (work / "ce_compiled.model").is_dir()


def test_an_unobstructed_run_writes_both_and_warns_about_neither(tmp_path):
    """The control: the same run without the directory in the way."""
    done = train(tmp_path, blocked=False)

    assert done.returncode == 0
    assert (tmp_path / "ce.model").is_file()
    assert (tmp_path / "ce_compiled.model").is_file()
    assert "was not written" not in done.stdout
