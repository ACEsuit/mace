"""A CLI device request places real training, resume and deployment there."""

import json

import numpy as np
import pytest
import torch
from conftest import assert_close
from mace_torch.calculators import MACECalculator
from mace_torch.cli import run_train
from mace_torch.deploy.loader import load_deployed
from mace_torch.train.checkpoint import latest_run_checkpoint
from safetensors.numpy import load_file
from test_mace_torch_calculator import water
from test_mace_torch_training_pipeline import configuration


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_device_flag_trains_resumes_and_loads_on_cpu(
    tmp_path, monkeypatch, dtype, device
):
    document = configuration(tmp_path).model_dump(mode="json")
    document["runtime"].update(name="placement", plot=False)
    document["precision"]["model"] = str(dtype).removeprefix("torch.")
    document["training"]["ema"] = {"enabled": True, "decay": 0.9}
    called, gradients = [], []
    build = run_train.build

    def checked_build(*args, **kwargs):
        result = build(*args, **kwargs)

        def check_forward(module, args):
            assert all(
                value.device.type == device for value in module.state_dict().values()
            )
            assert args[0]["positions"].device.type == device
            called.append(device)

        def check_gradient(value):
            assert value.device.type == device
            gradients.append(device)

        result.engine.register_forward_pre_hook(check_forward)
        for parameter in result.engine.parameters():
            if parameter.requires_grad:
                parameter.register_hook(check_gradient)
        return result

    monkeypatch.setattr(run_train, "build", checked_build)

    def fit(directory, epochs, resume=False):
        directory.mkdir(exist_ok=True)
        current = json.loads(json.dumps(document))
        current["runtime"]["work_dir"] = str(directory)
        current["training"]["max_num_epochs"] = epochs
        path = directory / "config.json"
        path.write_text(json.dumps(current))
        config = run_train.parse(
            [
                "--config",
                str(path),
                "--device",
                device,
                *(["--restart_latest"] if resume else []),
            ]
        )
        assert config.runtime.device == device
        result = run_train.run(config)
        assert all(np.isfinite(item.train_loss) for item in result.history)
        return result

    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        whole, split = tmp_path / "whole", tmp_path / "split"
        fit(whole, 2)
        fit(split, 1)
        resumed = fit(split, 2, resume=True)
        assert [item.epoch for item in resumed.history] == [1]
        assert called and gradients
        tensors, records = [], []
        for directory in (whole, split):
            path = latest_run_checkpoint(directory, "placement")
            assert path is not None
            tensors.append(load_file(path.with_suffix(".safetensors")))
            records.append(json.loads(path.read_text()))
        assert tensors[0].keys() == tensors[1].keys()
        for key in tensors[0]:
            np.testing.assert_array_equal(tensors[0][key], tensors[1][key], err_msg=key)
        for key in ("state", "optimizer", "scheduler", "ema", "model_tensors"):
            assert records[0][key] == records[1][key]
        artifact = split / "placement.json"
        cpu = load_deployed(artifact, device="cpu")
        requested = load_deployed(artifact, device=device)
        assert all(
            value.device.type == "cpu" for value in cpu.engine.state_dict().values()
        )
        assert cpu.config.runtime.device == device
        results = []
        for loaded, target in ((cpu, "cpu"), (requested, device)):
            calculator = MACECalculator(
                models=loaded,
                device=target,
                default_dtype=str(dtype).removeprefix("torch."),
            )
            calculator.calculate(water())
            results.append(calculator.results)
        for name in ("energy", "forces"):
            assert_close(results[0][name], results[1][name], name)
    finally:
        torch.use_deterministic_algorithms(deterministic)
