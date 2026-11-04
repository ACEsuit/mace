"""Default and periodic plotting preserve the trained model and epoch records."""

import pytest
import torch
from safetensors.torch import load_file

from tests.workflows.test_run_train_v1 import (
    needs_the_v1_engine,
    on_v1,
    tiny_task,
    training_records,
    variant,
    multihead_task,
    RESTART,
)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    outputs = {}
    for mode, arguments in (("default", ()), ("off", ("--plot", "False")),
                            ("periodic", ("--plot_frequency", "2"))):
        directory = tmp_path_factory.mktemp(f"plots_{mode}")
        config = variant(tiny_task(directory), {"training.max_num_epochs": 3})
        result = on_v1("--config", str(config), *arguments)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[mode] = directory
    return outputs


@needs_the_v1_engine
@pytest.mark.parametrize("mode", ["default", "periodic"])
def test_default_and_periodic_plots_preserve_training(runs, mode):
    plot = runs[mode] / "results/tiny.png"
    content = plot.read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n") and len(content) > 5000
    assert not list(runs["off"].rglob("*.png"))
    assert training_records(runs[mode]) == training_records(runs["off"])
    expected = load_file(runs["off"] / "tiny.safetensors")
    actual = load_file(runs[mode] / "tiny.safetensors")
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())


@needs_the_v1_engine
@pytest.mark.parametrize("custom", [False, True])
def test_head_and_stage_files_survive_resume_without_changing_training(tmp_path, custom):
    settings = {"training.max_num_epochs": 4, "runtime.plot_frequency": 0}
    if custom:
        settings["training.stages"] = [
            {"name": "warmup", "start_epoch": 0},
            {"name": "polish", "start_epoch": 2, "lr": 0.005},
        ]
        labels = ("stage-warmup", "stage-polish")
    else:
        settings.update({"training.stage_two.enabled": True,
                         "training.stage_two.start_epoch": 2, "training.stage_two.lr": 0.005})
        labels = ("stage_one", "stage_two")
    config = variant(multihead_task(tmp_path), settings)
    first = on_v1("--config", str(config), "--max_num_epochs", "2")
    assert first.returncode == 0, first.stdout + first.stderr
    directory = tmp_path / "results"
    earlier = {path.name: path.read_bytes() for path in directory.glob(f"*_{labels[0]}.png")}
    assert set(earlier) == {f"tiny_{head}_{labels[0]}.png" for head in ("small", "large")}
    resumed = on_v1("--config", str(variant(config, RESTART)))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert {path.name for path in directory.glob("*.png")} == {"tiny.png"} | {
        f"tiny_{head}_{stage}.png" for head in ("small", "large") for stage in labels
    }
    assert all((directory / name).read_bytes() == content for name, content in earlier.items())
    uninterrupted = tmp_path / "uninterrupted"
    complete = on_v1("--config", str(variant(config, {"runtime.work_dir": str(uninterrupted)})))
    assert complete.returncode == 0, complete.stdout + complete.stderr
    for name in (path.name for path in directory.glob("*.png")):
        assert (directory / name).read_bytes() == (uninterrupted / "results" / name).read_bytes()
    disabled = tmp_path / "disabled"
    off = on_v1("--config", str(variant(config, {"runtime.work_dir": str(disabled), "runtime.plot": False})))
    assert off.returncode == 0, off.stdout + off.stderr
    assert not list(disabled.rglob("*.png"))
    expected = load_file(disabled / "tiny.safetensors")
    actual = load_file(tmp_path / "tiny.safetensors")
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[name], value) for name, value in expected.items())
    assert training_records(tmp_path) == training_records(disabled)
