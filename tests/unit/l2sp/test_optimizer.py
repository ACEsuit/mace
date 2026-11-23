"""Numerical contracts for decay towards pretrained weights, independent of MACE."""

from copy import deepcopy

import pytest
import torch

from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.l2sp import (
    L2SPCheckpointError,
    L2SPRegularization,
    load_optimizer_state_dict,
    optimizer_state_dict,
    optimizer_step,
)


def make_optimizer(parameter, strength=10.0, **kwargs):
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "param_names": ["weight"]}],
        weight_decay=kwargs.pop("weight_decay", 0.0),
        lr=0.02,
        **kwargs,
    )
    optimizer.l2sp = L2SPRegularization(optimizer, strength)
    return optimizer


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("strength", [0.0, 10.0])
def test_matches_adamw_on_residuals(strength, amsgrad):
    reference = torch.tensor([2.0, -3.0, 0.4], dtype=torch.float64)
    weight = torch.nn.Parameter(reference.clone())
    delta = torch.nn.Parameter(torch.zeros_like(weight))
    optimizer = make_optimizer(weight, strength, amsgrad=amsgrad)
    residual_optimizer = torch.optim.AdamW(
        [delta], lr=0.02, weight_decay=strength, amsgrad=amsgrad
    )
    for step in range(8):
        # The same nonlinear data loss in both parameterizations.
        for opt, effective in (
            (optimizer, weight),
            (residual_optimizer, reference + delta),
        ):
            opt.zero_grad(set_to_none=True)
            ((effective.sin() - 0.1 * step).square().sum()).backward()
            opt.param_groups[0]["lr"] = 0.02 / (step + 1)
            optimizer_step(opt)
        torch.testing.assert_close(weight, reference + delta, atol=1e-14, rtol=1e-14)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[weight][key],
                residual_optimizer.state[delta][key],
                atol=1e-14,
                rtol=1e-14,
            )


def test_zero_gradient_decays_to_reference_and_none_does_not():
    weight = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    optimizer = make_optimizer(weight)
    with torch.no_grad():
        weight.copy_(torch.tensor([3.0, -1.5]))
    optimizer_step(optimizer)
    torch.testing.assert_close(
        weight, torch.tensor([3.0, -1.5], dtype=torch.float64), atol=0, rtol=0
    )
    weight.grad = torch.zeros_like(weight)
    optimizer_step(optimizer)
    torch.testing.assert_close(
        weight, torch.tensor([2.8, -1.4], dtype=torch.float64), atol=1e-15, rtol=0
    )
    assert optimizer.l2sp.squared_distance().item() == pytest.approx(0.8)
    torch.testing.assert_close(
        optimizer_state_dict(optimizer)["l2sp"]["references"][0][0],
        torch.tensor([2.0, -1.0], dtype=torch.float64),
    )


def test_first_step_and_closure():
    weight = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    optimizer = make_optimizer(weight, eps=0)

    def closure():
        optimizer.zero_grad()
        loss = (weight * torch.tensor([0.5, -0.25])).sum()
        loss.backward()
        return loss

    assert closure().item() == pytest.approx(1.75)
    optimizer_step(optimizer)
    torch.testing.assert_close(
        weight, torch.tensor([1.98, -2.98], dtype=torch.float64), atol=1e-15, rtol=0
    )


def test_checkpoint_restores_original_reference_and_trajectory():
    weight = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float64))
    optimizer = make_optimizer(weight)
    weight.grad = torch.tensor([0.5, -0.25], dtype=torch.float64)
    optimizer_step(optimizer)
    saved = deepcopy(optimizer_state_dict(optimizer))
    resumed_weight = torch.nn.Parameter(weight.detach().clone())
    resumed = make_optimizer(resumed_weight)
    load_optimizer_state_dict(resumed, saved)
    for step in range(4):
        for opt, param in ((optimizer, weight), (resumed, resumed_weight)):
            param.grad = torch.tensor([0.2 * step, -0.4], dtype=torch.float64)
            optimizer_step(opt)
        torch.testing.assert_close(weight, resumed_weight, atol=0, rtol=0)
    torch.testing.assert_close(
        optimizer_state_dict(resumed)["l2sp"]["references"][0][0],
        torch.tensor([2.0, -3.0], dtype=torch.float64),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "change", ["missing", "strength", "rule", "names", "shape", "decay"]
)
def test_incompatible_optimizer_checkpoint_fails(change):
    weight = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    optimizer = make_optimizer(weight)
    saved = deepcopy(optimizer_state_dict(optimizer))
    if change == "missing":
        del saved["l2sp"]
    elif change == "strength":
        saved["l2sp"]["strength"] = 0.0
    elif change == "rule":
        saved["l2sp"]["rule"] = "loss"
    elif change == "names":
        saved["param_groups"][0]["param_names"] = ["wrong.weight"]
    elif change == "shape":
        saved["l2sp"]["references"][0][0] = torch.zeros(3)
    else:
        saved["param_groups"][0]["weight_decay"] = 1.0
    with pytest.raises(L2SPCheckpointError):
        load_optimizer_state_dict(optimizer, saved)


@pytest.mark.parametrize("strength", [-1, float("nan"), float("inf")])
def test_invalid_strength(strength):
    with pytest.raises(ValueError, match="strength"):
        make_optimizer(torch.nn.Parameter(torch.ones(1)), strength)


def test_zero_centered_decay_is_rejected():
    with pytest.raises(ValueError, match="weight_decay=0"):
        make_optimizer(torch.nn.Parameter(torch.ones(1)), weight_decay=0.1)


def test_zero_lr_preserves_absolute_weights_without_round_trip_error():
    parameter = torch.nn.Parameter(torch.tensor([1e20], dtype=torch.float64))
    optimizer = make_optimizer(parameter)
    with torch.no_grad():
        parameter.fill_(1.0)
    optimizer.param_groups[0]["lr"] = 0.0
    parameter.grad = torch.ones_like(parameter)
    optimizer_step(optimizer)
    torch.testing.assert_close(parameter, torch.ones_like(parameter), atol=0, rtol=0)


def test_failed_step_restores_absolute_coordinates_and_decay(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = make_optimizer(parameter)
    with torch.no_grad():
        parameter.fill_(3.0)
    parameter.grad = torch.ones_like(parameter)

    def fail():
        assert parameter.item() == 1.0
        assert optimizer.param_groups[0]["weight_decay"] == 10.0
        raise RuntimeError("failed update")

    monkeypatch.setattr(optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="failed update"):
        optimizer_step(optimizer)
    assert parameter.item() == 3.0
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_legacy_decoupled_checkpoint_without_mode_is_accepted():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = make_optimizer(parameter)
    saved = deepcopy(optimizer_state_dict(optimizer))
    saved["l2sp"]["version"] = 1
    del saved["l2sp"]["rule"]
    del saved["l2sp"]["optimizer_kind"]
    with torch.no_grad():
        parameter.add_(1.0)
    resumed = make_optimizer(parameter)
    load_optimizer_state_dict(resumed, saved)
    torch.testing.assert_close(
        optimizer_state_dict(resumed)["l2sp"]["references"][0][0], torch.ones(1)
    )


def test_checkpoint_builder_rejects_changing_optimizer_kind_before_model_load():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = make_optimizer(model.weight)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    saved = deepcopy(
        CheckpointBuilder.create_checkpoint(
            CheckpointState(model, optimizer, scheduler)
        )
    )
    saved["model"]["weight"].zero_()
    original = model.weight.detach().clone()
    plain_optimizer = torch.optim.AdamW(model.parameters())
    state = CheckpointState(model, plain_optimizer, scheduler)
    with pytest.raises(L2SPCheckpointError, match="requires --l2sp"):
        CheckpointBuilder.load_checkpoint(state, saved, strict=False)
    torch.testing.assert_close(model.weight, original, atol=0, rtol=0)


def test_checkpoint_model_is_strict_for_l2sp():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = make_optimizer(model.weight)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    state = CheckpointState(model, optimizer, scheduler)
    saved = deepcopy(CheckpointBuilder.create_checkpoint(state))
    del saved["model"]["weight"]
    with pytest.raises(L2SPCheckpointError, match="model is incompatible"):
        CheckpointBuilder.load_checkpoint(state, saved, strict=False)
