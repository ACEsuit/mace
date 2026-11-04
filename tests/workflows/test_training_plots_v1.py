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
