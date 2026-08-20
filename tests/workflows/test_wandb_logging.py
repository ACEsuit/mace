"""The `--wandb` flags, exercised without a Weights and Biases account.

Nothing exercised any of the six. They are the flags whose failure mode is a run
that trains fine and logs nowhere, or logs under the wrong project, and neither
shows up as an error.

Two levels. `setup_wandb` is where five of the flags become arguments to
`wandb.init`, so that is checked by capturing the call: a flag read into the wrong
keyword is invisible in any end-to-end run, since `wandb.init` accepts all of them
happily. Then one real training with `--wandb` under `WANDB_MODE=offline`, which is
the only thing that says the flag is wired into `run_train` at all, and the only
place the offline run directory gets created.

Offline mode is not a stub: wandb writes a real run directory and a real summary,
so the smoke exercises the same code an account would, minus the upload.
"""

import argparse
import json
from pathlib import Path

import ase.io
import pytest

from mace.tools import scripts_utils
from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

pytest.importorskip("wandb")

HYPERS = ["lr", "batch_size", "max_num_epochs"]


@pytest.fixture(name="captured_init")
def fixture_captured_init(monkeypatch):
    """What `setup_wandb` passes on, without starting a run."""
    seen = {}

    def fake_init_wandb(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(scripts_utils.tools, "init_wandb", fake_init_wandb)

    # `setup_wandb` does `import wandb` in its own body, so the module itself has
    # to carry the stub run: patching the name on `scripts_utils` would be
    # rebound by that import and the test would assert against nothing.
    import wandb  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    class _Run:
        def __init__(self):
            self.summary = {}

    monkeypatch.setattr(wandb, "run", _Run())
    return seen


def args_for(**overrides):
    args = argparse.Namespace(
        wandb=True,
        wandb_project="a-project",
        wandb_entity="an-entity",
        wandb_name="a-run",
        wandb_dir="/tmp/a-dir",
        wandb_log_hypers=list(HYPERS),
        lr=0.005,
        batch_size=7,
        max_num_epochs=3,
        name="model",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Where the five flags land
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,keyword,value",
    [
        ("wandb_project", "project", "a-project"),
        ("wandb_entity", "entity", "an-entity"),
        ("wandb_name", "name", "a-run"),
        ("wandb_dir", "directory", "/tmp/a-dir"),
    ],
)
def test_each_flag_lands_on_its_own_keyword(captured_init, flag, keyword, value):
    """Four flags, four keywords, and `wandb.init` would accept them in any
    arrangement, so the mapping is what needs pinning."""
    scripts_utils.setup_wandb(args_for())

    assert captured_init[keyword] == value
    assert flag  # named in the parametrization so a failure says which flag


def test_only_the_requested_hyperparameters_are_logged(captured_init):
    """`--wandb_log_hypers` is a list of arg names, and the config is built by
    looking each one up. Logging everything, or the wrong subset, is silent."""
    scripts_utils.setup_wandb(args_for())

    assert set(captured_init["config"]) == set(HYPERS)
    assert captured_init["config"]["batch_size"] == 7
    assert captured_init["config"]["lr"] == 0.005


def test_a_narrower_hyperparameter_list_is_honoured(captured_init):
    scripts_utils.setup_wandb(args_for(wandb_log_hypers=["lr"]))

    assert set(captured_init["config"]) == {"lr"}


def test_a_hyperparameter_that_is_not_an_argument_is_refused(captured_init):
    """The lookup is `args_dict[key]`, so a typo in `--wandb_log_hypers` raises
    rather than logging one fewer hyperparameter than asked for."""
    with pytest.raises(KeyError):
        scripts_utils.setup_wandb(args_for(wandb_log_hypers=["lr", "not_an_arg"]))


def test_the_full_arguments_are_recorded_as_json(captured_init, monkeypatch):
    """Beyond the chosen hypers, the whole namespace goes into the run summary,
    which is what makes a logged run reproducible. It has to be serialisable:
    `KeySpecification` needs the custom encoder, and an ndarray needs converting.
    """
    import numpy as np  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    from mace.data.utils import KeySpecification  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    args = args_for()
    args.key_specification = KeySpecification()
    args.atomic_energies = np.array([1.0, 2.0])

    scripts_utils.setup_wandb(args)

    import wandb  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    recorded = json.loads(wandb.run.summary["params"])
    assert recorded["wandb_project"] == "a-project"
    assert recorded["atomic_energies"] == [1.0, 2.0], "an ndarray must be listified"
    assert isinstance(recorded["key_specification"], dict)


# ---------------------------------------------------------------------------
# --wandb itself, end to end
# ---------------------------------------------------------------------------


def test_a_training_run_with_wandb_writes_an_offline_run(tmp_path):
    """The only statement that `--wandb` reaches `setup_wandb` from `run_train`.

    `WANDB_MODE=offline` keeps it off the network while still writing a real run
    directory, so this fails if the flag stops being honoured, if the config keys
    stop being serialisable, or if `init_wandb` starts raising.
    """
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    wandb_dir = tmp_path / "wb"
    wandb_dir.mkdir()
    params = base_mace_params()
    params.update(
        {
            "name": "wb",
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 1,
            "wandb": None,
            "wandb_dir": str(wandb_dir),
            "wandb_project": "mace-test",
            "wandb_name": "smoke",
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)

    result = run_mace_train(
        params, env_extra={"WANDB_MODE": "offline", "WANDB_SILENT": "true"}
    )

    assert result.returncode == 0
    runs = list((wandb_dir / "wandb").glob("offline-run-*"))
    assert runs, f"no offline run in {wandb_dir}: {[p.name for p in wandb_dir.rglob('*')][:10]}"


def test_a_run_without_the_flag_writes_nothing(tmp_path):
    """Otherwise the test above would pass on a build that logged unconditionally.
    """
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    wandb_dir = tmp_path / "wb"
    wandb_dir.mkdir()
    params = base_mace_params()
    params.update(
        {
            "name": "nowb",
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 1,
            "wandb_dir": str(wandb_dir),
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)

    run_mace_train(params, env_extra={"WANDB_MODE": "offline"})

    assert not list(wandb_dir.glob("**/offline-run-*"))
