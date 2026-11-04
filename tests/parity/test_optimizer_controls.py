"""CLI optimizer controls on identical parameters, gradients and step history.

The small tensor groups isolate optimizer arithmetic from the reduced/full
Clebsch-Gordan coordinate change. Actual model training and resume are covered
by the native pipeline cases alongside this differential test.
"""

import argparse
import json
from copy import deepcopy

import pytest
import torch
from mace_torch.cli.run_train import parse
from mace_torch.train.optimizers import (
    build_optimizer,
    build_scheduler,
    set_optimizer_training,
)
from torch import nn

from tests.golden.harness import EXACT


def tensor_groups(dtype):
    model = nn.Module()
    model.node_embedding = nn.Linear(2, 3, dtype=dtype)
    interaction = nn.Module()
    interaction.linear = nn.Linear(3, 2, dtype=dtype)
    interaction.radial = nn.Linear(2, 2, dtype=dtype)
    model.interactions = nn.ModuleList([interaction])
    model.products = nn.ModuleList([nn.Linear(2, 3, dtype=dtype)])
    model.readouts = nn.ModuleList([nn.Linear(3, 1, dtype=dtype)])
    return model


def exact(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=EXACT.atol, rtol=EXACT.rtol)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "kind",
    ["adam", "adamw", pytest.param("schedulefree", marks=pytest.mark.schedulefree)],
)
@pytest.mark.parametrize("scheduler", ["ReduceLROnPlateau", "ExponentialLR"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cli_optimizer_steps_and_state_match_legacy(
    isolated, tmp_path, kind, scheduler, custom, dtype
):
    from mace.tools.scripts_utils import LRScheduler, get_optimizer, get_params_options

    torch.manual_seed(9)
    frozen = tensor_groups(dtype)
    native = nn.Module()
    native.backbone = deepcopy(frozen)
    settings = dict(
        optimizer=kind,
        scheduler=scheduler,
        lr=0.017,
        weight_decay=0.031,
        lr_factor=0.6,
        scheduler_patience=1,
        lr_scheduler_gamma=0.71,
        beta=0.87 if custom else 0.9,
        amsgrad=not custom,
        beta1_schedulefree=0.73 if custom else 0.9,
        # Preserve V1's existing default. The frozen CLI defaults to 0.98;
        # comparisons below give the oracle the same explicit configuration.
        beta2_schedulefree=0.93 if custom else 0.999,
        warmup_steps_schedulefree=3 if custom else 0,
    )
    selected = {
        key: value
        for key, value in settings.items()
        if (key not in ("beta", "amsgrad") or kind != "schedulefree")
        and ("schedulefree" not in key or kind == "schedulefree")
    }
    flags = [
        item
        for key, value in selected.items()
        if key != "amsgrad"
        and (
            custom
            or key
            not in (
                "beta",
                "beta1_schedulefree",
                "beta2_schedulefree",
                "warmup_steps_schedulefree",
            )
        )
        for item in (f"--{key}", str(value))
    ]
    if kind != "schedulefree" and settings["amsgrad"]:
        flags.append("--amsgrad")
    parsed = parse(flags)
    if kind != "schedulefree" and not settings["amsgrad"]:
        # Both historical CLIs expose a presence-only AMSGrad flag. Disabling
        # it belongs to the configuration file, not a new flag spelling.
        document = parsed.model_dump(mode="json")
        document["training"]["optimizer"]["amsgrad"] = False
        source = tmp_path / "config.json"
        source.write_text(json.dumps(document))
        parsed = parse(["--config", str(source)])
    config = parsed.training
    arguments = argparse.Namespace(**settings, freeze=0, lr_params_factors="{}")
    expected = get_optimizer(arguments, get_params_options(arguments, frozen))
    actual = build_optimizer(native, config)
    expected_schedule = LRScheduler(expected, arguments)
    actual_schedule = build_scheduler(actual, config.scheduler)
    if kind == "schedulefree":
        assert actual_schedule is None
        assert config.optimizer.beta1 == settings["beta1_schedulefree"]
        assert config.optimizer.beta2 == settings["beta2_schedulefree"]
        assert config.optimizer.warmup_steps == settings["warmup_steps_schedulefree"]
    else:
        assert config.optimizer.beta == settings["beta"]
        assert config.optimizer.amsgrad == settings["amsgrad"]
    initial = {
        name: parameter.detach().clone()
        for name, parameter in frozen.named_parameters()
    }
    for step, metric in enumerate([1.0, 0.8, 0.8, 0.9, 0.9, 0.7]):
        set_optimizer_training(actual, training=True)
        if kind == "schedulefree":
            expected.train()
        for (name, reference), (native_name, parameter) in zip(
            frozen.named_parameters(), native.backbone.named_parameters(), strict=True
        ):
            assert name == native_name
            gradient = torch.sin(
                torch.arange(reference.numel(), dtype=dtype) + step + 0.3
            ).reshape_as(reference)
            reference.grad, parameter.grad = gradient.clone(), gradient.clone()
        expected.step()
        actual.step()
        if step % 2 == 0:
            set_optimizer_training(actual, training=False)
            if kind == "schedulefree":
                expected.eval()
        expected_schedule.step(metrics=metric)
        if actual_schedule is not None:
            actual_schedule.step(
                metric
            ) if scheduler == "ReduceLROnPlateau" else actual_schedule.step()
        for reference, parameter in zip(
            frozen.parameters(), native.parameters(), strict=True
        ):
            exact(parameter, reference)
            exact(actual.state[parameter], expected.state[reference])
        for got, wanted in zip(actual.param_groups, expected.param_groups, strict=True):
            for key in wanted.keys() - {"name", "params", "initial_lr"}:
                exact(got[key], wanted[key])
        if actual_schedule is not None:
            exact(actual_schedule.state_dict(), expected_schedule.state_dict())
    assert any(
        not torch.equal(parameter, initial[name])
        for name, parameter in frozen.named_parameters()
    )
