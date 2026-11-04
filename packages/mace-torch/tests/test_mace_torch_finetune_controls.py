"""CLI fine-tuning controls must survive actual training and interruption."""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import assert_close
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.cli import run_train
from mace_torch.deploy.loader import load_deployed
from mace_torch.serialization import canonical_state
from mace_torch.train import loop, model_stage
from mace_torch.train.checkpoint import latest_run_checkpoint, write_model
from safetensors.numpy import load_file
from test_mace_torch_training_pipeline import configuration


@pytest.mark.parametrize("ema", [False, True])
@pytest.mark.parametrize("transfer", [False, True])
def test_lora_cli_trains_resumes_and_saves_a_plain_model(
    tmp_path, monkeypatch, dtype, ema, transfer
):
    config = configuration(tmp_path, max_num_epochs=2)
    config = config.model_copy(
        update={
            "precision": replace(
                config.precision, model=str(dtype).removeprefix("torch.")
            )
        }
    )
    foundation = run_train.build(config, DEFAULT_CATALOGUE)
    artifact = write_model(
        tmp_path / "foundation", foundation.model, foundation.metadata
    )
    document = config.model_dump(mode="json")
    document["model"] = {"observables": ["energy", "forces"]}
    document["runtime"].update(name="adapted", plot=False)
    document["finetune"] = {"foundation_model": str(artifact)}
    document["training"]["ema"] = {"enabled": ema, "decay": 0.9}
    transfers, gradients = [], []
    original_transfer = model_stage.transfer_foundation

    def checked_transfer(source, target, **kwargs):
        before = {
            path: {key: value.clone() for key, value in state.items()}
            for path, state in canonical_state(target).items()
            if ".readouts." in path
        }
        assert before and kwargs["transfer_readout"] is transfer
        original_transfer(source, target, **kwargs)
        expected = canonical_state(source) if transfer else before
        after = canonical_state(target)
        for path in before:
            for key in before[path]:
                torch.testing.assert_close(
                    after[path][key], expected[path][key], atol=0, rtol=0
                )
        transfers.append(transfer)

    monkeypatch.setattr(model_stage, "transfer_foundation", checked_transfer)
    inject = loop.inject_lora
    frozen = {}

    def checked_inject(model, rank, alpha):
        assert (rank, alpha) == (3, 2.5)
        result = inject(model, rank=rank, alpha=alpha)
        frozen.clear()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert ".parametrizations." in name and not name.endswith(".original")
                parameter.register_hook(
                    lambda gradient: gradients.append(
                        gradient.detach().abs().max().item()
                    )
                )
            else:
                frozen[name] = parameter.detach().clone()
        return result

    monkeypatch.setattr(loop, "inject_lora", checked_inject)
    write_run = loop.write_run_checkpoint

    def checked_write(*args, model, **kwargs):
        parameters = dict(model.named_parameters())
        for name, expected in frozen.items():
            assert parameters[name].grad is None
            torch.testing.assert_close(parameters[name], expected, atol=0, rtol=0)
        return write_run(*args, model=model, **kwargs)

    monkeypatch.setattr(loop, "write_run_checkpoint", checked_write)

    def fit(directory, epochs, resume=False):
        directory.mkdir(exist_ok=True)
        current = json.loads(json.dumps(document))
        current["runtime"]["work_dir"] = str(directory)
        current["training"]["max_num_epochs"] = epochs
        path = directory / "config.json"
        path.write_text(json.dumps(current))
        parsed = run_train.parse(
            [
                "--config",
                str(path),
                "--lora",
                "true",
                "--lora_rank",
                "3",
                "--lora_alpha",
                "2.5",
                "--foundation_model_readout",
                str(transfer),
                *(["--restart_latest"] if resume else []),
            ]
        )
        return run_train.run(parsed)

    whole, split = tmp_path / "whole", tmp_path / "split"
    fit(whole, 2)
    fit(split, 1)
    resumed = fit(split, 2, resume=True)
    assert [item.epoch for item in resumed.history] == [1]
    assert len(transfers) == 3 and any(value > 0 for value in gradients)
    tensors, records = [], []
    for directory in (whole, split):
        path = latest_run_checkpoint(directory, "adapted")
        tensors.append(load_file(path.with_suffix(".safetensors")))
        records.append(json.loads(path.read_text()))
    assert tensors[0].keys() == tensors[1].keys()
    for key in tensors[0]:
        np.testing.assert_array_equal(tensors[0][key], tensors[1][key], err_msg=key)
    for key in ("state", "optimizer", "scheduler", "ema", "model_tensors", "lora"):
        assert records[0][key] == records[1][key]
    loaded = load_deployed(split / "adapted.json")
    assert not any(
        "parametrizations" in name for name, _ in loaded.engine.named_parameters()
    )
    graph = next(iter(foundation.data.valid_loaders["default"])).graph
    expected = resumed.model(dict(graph), compute=("forces",))
    actual = loaded.engine(dict(graph), compute=("forces",))
    assert_close(actual.total_energy.detach(), expected.total_energy.detach(), "energy")
    assert_close(actual.forces.detach(), expected.forces.detach(), "forces")


@pytest.mark.parametrize(
    "changed", ["missing_state", "missing_tensor", "rank", "alpha", "disabled"]
)
def test_incomplete_or_different_lora_state_cannot_resume(tmp_path, changed):
    from mace_torch.finetune.lora import inject_lora
    from mace_torch.serialization import CheckpointError
    from mace_torch.train import RunState, read_run_checkpoint, write_run_checkpoint
    from mace_torch.train.optimizers import build_optimizer

    config = configuration(tmp_path)
    model = run_train.build(config, DEFAULT_CATALOGUE).model
    inject_lora(model, rank=3, alpha=2.5)
    path = write_run_checkpoint(
        tmp_path,
        "adapter",
        RunState(1),
        model=model,
        optimizer=build_optimizer(model, config.training),
    )
    record = json.loads(path.read_text())
    if changed == "missing_state":
        del record["lora"]
    elif changed == "missing_tensor":
        record["lora"]["parameters"]["mapping"][0][1]["tensor"] = "absent"
    path.write_text(json.dumps(record))
    target = run_train.build(config, DEFAULT_CATALOGUE).model
    if changed != "disabled":
        inject_lora(
            target,
            rank=2 if changed == "rank" else 3,
            alpha=0.5 if changed == "alpha" else 5 / 3 if changed == "rank" else 2.5,
        )
    with pytest.raises(CheckpointError, match="LoRA"):
        read_run_checkpoint(
            path, model=target, optimizer=build_optimizer(target, config.training)
        )
