"""Compare MACE E/F/stress training against the Equitrain residual formulation.

The oracle freezes a copy of the base model and calls it with theta_0 + delta,
using ordinary torch AdamW on delta. This reproduces Equitrain's algorithm
without an optional Equitrain installation or any downloads.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.func import functional_call
from torch.optim.swa_utils import AveragedModel
from torch_ema import ExponentialMovingAverage

from mace.calculators import MACECalculator
from mace.modules.loss import WeightedHuberEnergyForcesStressLoss
from mace.tools import torch_tools
from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.scripts_utils import get_optimizer, get_params_options
from mace.tools.train import evaluate, take_step
from tests.golden.anchors import anchor_batch, load_anchor, load_training_structures
from tests.unit.l2sp.test_config import args_for

OUTPUT_ARGS = {"forces": True, "stress": True, "virials": True}


def train_against_residual(
    selection="products,interactions.1,readouts", use_ema=False, model=None
):
    if model is None:
        model = load_anchor("tiny_scaleshift")
    device = next(model.parameters()).device
    base = deepcopy(model)
    base.requires_grad_(False)
    args = args_for(freeze=selection)
    optimizer = get_optimizer(args, get_params_options(args, model))
    deltas = {
        name: torch.nn.Parameter(torch.zeros_like(param))
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    residual_optimizer = torch.optim.AdamW(
        deltas.values(),
        lr=args.lr,
        weight_decay=args.l2sp_strength,
        betas=(args.beta, 0.999),
        amsgrad=args.amsgrad,
    )
    structures = load_training_structures(limit=2)
    batch = anchor_batch(model, structures).to(device)
    loss_fn = WeightedHuberEnergyForcesStressLoss(
        energy_weight=10, forces_weight=100, stress_weight=1000
    ).to(device=device, dtype=torch.float64)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9) if use_ema else None
    delta_ema = (
        ExponentialMovingAverage(deltas.values(), decay=0.9) if use_ema else None
    )

    def residual_forward(training):
        params = {
            name: param + deltas[name] if name in deltas else param
            for name, param in base.named_parameters()
        }
        return functional_call(
            base,
            params,
            (batch.clone().to_dict(),),
            {"training": training, "compute_force": True, "compute_stress": True},
        )

    initial = deepcopy(model.state_dict())
    for step in range(3):
        # Exercise stage-two-style LR changes without changing the reference.
        for opt in (optimizer, residual_optimizer):
            for group in opt.param_groups:
                group["lr"] = args.lr / (step + 1)
        residual_optimizer.zero_grad(set_to_none=True)
        residual_output = residual_forward(training=True)
        residual_loss = loss_fn(batch, residual_output)
        residual_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(deltas.values()), 1.0)
        _, metrics = take_step(
            model,
            loss_fn,
            batch.clone(),
            optimizer,
            ema,
            OUTPUT_ARGS,
            1.0,
            device,
        )
        residual_optimizer.step()
        if delta_ema:
            delta_ema.update()
        assert metrics["loss"] == pytest.approx(
            residual_loss.item(), abs=1e-11, rel=1e-11
        )
        assert metrics["l2sp_squared_distance"] > 0
        for name, param in model.named_parameters():
            if name in deltas:
                torch.testing.assert_close(
                    param, initial[name] + deltas[name], atol=2e-12, rtol=2e-12
                )
            else:
                torch.testing.assert_close(param, initial[name], atol=0, rtol=0)

    if ema:
        ema.copy_to()
        delta_ema.copy_to()
    expected = residual_forward(training=False)
    actual = model(batch.clone().to_dict(), compute_force=True, compute_stress=True)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-11, rtol=2e-11)
    return model, optimizer, batch, actual, expected


@pytest.mark.parametrize("selection", [None, "products,interactions.1,readouts"])
@pytest.mark.parametrize("use_ema", [False, True])
def test_mace_training_matches_residual_adamw(selection, use_ema):
    with torch_tools.default_dtype("float64"):
        _, _, _, actual, _ = train_against_residual(selection, use_ema)
    if selection == "products,interactions.1,readouts" and not use_ema:
        reference = json.loads(
            Path(__file__).with_name("reference.json").read_text(encoding="utf-8")
        )
        for key in ("energy", "forces", "stress"):
            np.testing.assert_allclose(
                actual[key].detach().numpy(), reference[key], atol=2e-10, rtol=2e-10
            )


def test_evaluate_resume_and_export_preserve_model(tmp_path):
    with torch_tools.default_dtype("float64"):
        model, optimizer, batch, _, _ = train_against_residual()
        flags = {name: param.requires_grad for name, param in model.named_parameters()}
        loss_fn = WeightedHuberEnergyForcesStressLoss().double()
        evaluate(model, loss_fn, [batch.clone()], OUTPUT_ARGS, torch.device("cpu"))
        assert flags == {
            name: param.requires_grad for name, param in model.named_parameters()
        }

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        state = CheckpointState(model, optimizer, scheduler)
        checkpoint = deepcopy(CheckpointBuilder.create_checkpoint(state))
        resumed = load_anchor("tiny_scaleshift")
        args = args_for()
        resumed_opt = get_optimizer(args, get_params_options(args, resumed))
        resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            resumed_opt, gamma=0.9
        )
        CheckpointBuilder.load_checkpoint(
            CheckpointState(resumed, resumed_opt, resumed_scheduler),
            checkpoint,
            strict=False,
        )
        for candidate, opt in ((model, optimizer), (resumed, resumed_opt)):
            take_step(
                candidate,
                loss_fn,
                batch.clone(),
                opt,
                None,
                OUTPUT_ARGS,
                1.0,
                torch.device("cpu"),
            )
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter, dict(resumed.named_parameters())[name], atol=0, rtol=0
            )

        # Stage-two averaging uses ordinary model weights and preserves the schema.
        averaged = AveragedModel(model)
        averaged.update_parameters(model)
        torch.testing.assert_close(
            averaged.module.node_embedding.linear.weight,
            model.node_embedding.linear.weight,
        )

        path = tmp_path / "l2sp.model"
        torch.save(model, path)
        calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64")
        atoms = load_training_structures(limit=1)[0].copy()
        atoms.calc = calc
        output = model(
            anchor_batch(model, [atoms]).to_dict(),
            compute_force=True,
            compute_stress=True,
        )
        assert atoms.get_potential_energy() == pytest.approx(
            output["energy"].item(), abs=1e-12
        )
        np.testing.assert_allclose(
            atoms.get_forces(),
            output["forces"].detach().numpy(),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            atoms.get_stress(voigt=False),
            output["stress"][0].detach().numpy(),
            atol=1e-12,
            rtol=1e-12,
        )
        assert set(calc.models[0].state_dict()) == set(model.state_dict())
