"""`--foundation_model_kwargs`, and the one path that reads it.

The flag is a string holding a Python dict literal, parsed with
`ast.literal_eval` at the very top of `run()` -- before any data is read, and
whether or not a foundation model was asked for. So the parse is what every run
pays for, and it had no test.

Where the parsed dict *goes* is narrower than the flag's name suggests: it is
splatted into `mace_mp(...)`, and only on the branch that recognises the
foundation model as one of the named MACE-MP releases. A foundation model given
as a path takes a different branch, which never looks at the kwargs. Passing both
is therefore accepted and silently ineffective, which is worth a test precisely
because nothing says so.
"""

from pathlib import Path

import ase.io
import pytest

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

ANCHOR = Path(__file__).resolve().parents[1] / "golden" / "models" / "tiny_scaleshift.model"


def train(tmp_path, kwargs_value, foundation=None, check=True):
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    params = base_mace_params()
    params.update(
        {
            "name": "fk",
            "hidden_irreps": "16x0e",
            "checkpoints_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "results_dir": str(tmp_path),
            "log_dir": str(tmp_path),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 1,
            "seed": 3,
            "foundation_model_kwargs": kwargs_value,
        }
    )
    if foundation is not None:
        params["foundation_model"] = str(foundation)
        params["multiheads_finetuning"] = "False"
    return run_mace_train(
        params, check=check, capture_output=True, text=True
    )


# ---------------------------------------------------------------------------
# the parse, which every run performs
# ---------------------------------------------------------------------------


def test_the_default_is_an_empty_dict_and_a_run_survives_it(tmp_path):
    """`--foundation_model_kwargs` defaults to the string `{}`, so the parse runs
    on every training whether or not there is a foundation model to configure."""
    assert train(tmp_path, "{}").returncode == 0


def test_a_value_that_is_not_a_literal_stops_the_run(tmp_path):
    """`ast.literal_eval`, so a name or a call is refused rather than executed.
    This is what keeps the flag from being an arbitrary-code entry point."""
    done = train(tmp_path, "{'device': open('x')}", check=False)

    assert done.returncode != 0
    assert "ValueError" in done.stderr or "malformed node" in done.stderr, done.stderr


def test_an_unparseable_value_stops_the_run(tmp_path):
    done = train(tmp_path, "{not a dict", check=False)

    assert done.returncode != 0
    assert "SyntaxError" in done.stderr, done.stderr


def test_a_literal_that_is_not_a_dict_stops_the_run(tmp_path):
    """`[1, 2]` parses and then fails on the next line, where the foundation head
    is written into it. Nothing checks the shape, so the message is about list
    indices rather than about the flag -- the only hint a user gets that a mapping
    was wanted."""
    done = train(tmp_path, "[1, 2]", check=False)

    assert done.returncode != 0
    assert "list indices must be integers" in done.stderr, done.stderr


# ---------------------------------------------------------------------------
# where it is read, and where it is not
# ---------------------------------------------------------------------------


def test_the_kwargs_are_ignored_when_the_foundation_model_is_a_path(tmp_path):
    """Recorded, not endorsed. Only the named-release branch splats them into
    `mace_mp`; a checkpoint path is loaded directly. A kwarg no loader would
    accept is therefore accepted here, and the run finishes as if it had not been
    given -- so a user who spells the model as a path gets silence rather than
    the setting they asked for.
    """
    done = train(tmp_path, "{'nonexistent_kwarg': 12345}", foundation=ANCHOR)

    assert done.returncode == 0
    assert "nonexistent_kwarg" not in done.stderr
