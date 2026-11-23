"""L2-SP on real converted models, using CUEQ's CPU fallback or CUDA kernels."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from mace.calculators import MACECalculator
from mace.cli.convert_cueq_e3nn import run as to_e3nn
from mace.cli.convert_e3nn_cueq import run as to_cueq
from mace.modules.loss import WeightedHuberEnergyForcesStressLoss
from mace.tools import torch_tools
from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.l2sp import L2SPCheckpointError
from mace.tools.scripts_utils import (
    extract_config_mace_model,
    get_optimizer,
    get_params_options,
)
from mace.tools.train import take_step
from tests.golden.anchors import anchor_batch, load_anchor, load_training_structures
from tests.unit.l2sp.test_config import args_for
from tests.unit.l2sp.test_model import OUTPUT_ARGS, train_against_residual

pytestmark = pytest.mark.cueq
TARGETED = "products,interactions.1,readouts"


@pytest.fixture(
    name="backend",
    params=[
        pytest.param(("cpu", False), id="cpu"),
        pytest.param(("cuda", False), id="cuda", marks=pytest.mark.gpu),
        pytest.param(("cuda", True), id="cuda-fused", marks=pytest.mark.gpu),
    ],
)
def backend_fixture(request):
    device, conv_fusion = request.param
    return {"device": device, "conv_fusion": conv_fusion}


@pytest.fixture(name="model_pair", params=[False, True], ids=["unreduced", "reduced"])
def model_pair_fixture(request, backend):
    with torch_tools.default_dtype("float64"):
        source = load_anchor("tiny_scaleshift")
        config = extract_config_mace_model(source)
        if config["use_reduced_cg"] != request.param:
            config["use_reduced_cg"] = request.param
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(27)
                source = type(source)(**config)
        source = source.to(backend["device"])
        converted = to_cueq(deepcopy(source), **backend)
        assert type(converted.products[0].symmetric_contractions).__module__.startswith(
            "cuequivariance_torch."
        )
        yield source, converted


@pytest.mark.parametrize("selection", [None, TARGETED])
@pytest.mark.parametrize("use_ema", [False, True])
def test_training_matches_cueq_residual(model_pair, selection, use_ema):
    _, model = model_pair
    initial = dict(model.named_parameters())
    initial = {name: param.detach().clone() for name, param in initial.items()}
    _, optimizer, _, _, _ = train_against_residual(selection, use_ema, model=model)
    for group, refs in zip(optimizer.param_groups, optimizer.l2sp.references):
        for name, reference in zip(group["param_names"], refs):
            torch.testing.assert_close(reference, initial[name], atol=0, rtol=0)


def test_targeted_training_matches_e3nn(model_pair):
    source, converted = model_pair
    e3nn_model, _, _, expected, _ = train_against_residual(model=source)
    cueq_model, _, _, actual, _ = train_against_residual(model=converted)
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-10, rtol=2e-10)
    e3nn_params = dict(e3nn_model.named_parameters())
    for name, parameter in cueq_model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.reshape(-1),
                e3nn_params[name].reshape(-1),
                atol=2e-10,
                rtol=2e-10,
            )


@pytest.mark.parametrize("selection", [None, TARGETED])
def test_checkpoint_resume_and_e3nn_export(model_pair, backend, selection, tmp_path):
    source, converted = model_pair
    model, optimizer, batch, _, _ = train_against_residual(selection, model=converted)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        CheckpointBuilder.create_checkpoint(
            CheckpointState(model, optimizer, scheduler)
        ),
        path,
    )
    checkpoint = torch.load(path, weights_only=False, map_location=backend["device"])
    args = args_for(freeze=selection, enable_cueq=True)
    resumed = to_cueq(deepcopy(source), **backend)
    resumed_opt = get_optimizer(args, get_params_options(args, resumed))
    resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(resumed_opt, gamma=0.9)
    CheckpointBuilder.load_checkpoint(
        CheckpointState(resumed, resumed_opt, resumed_scheduler),
        checkpoint,
        strict=False,
    )
    for refs, restored_refs in zip(
        optimizer.l2sp.references, resumed_opt.l2sp.references
    ):
        for reference, restored in zip(refs, restored_refs):
            torch.testing.assert_close(reference, restored, atol=0, rtol=0)

    loss_fn = WeightedHuberEnergyForcesStressLoss().to(
        device=backend["device"], dtype=torch.float64
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
            torch.device(backend["device"]),
        )
    resumed_params = dict(resumed.named_parameters())
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, resumed_params[name], atol=0, rtol=0)
        assert param.requires_grad == resumed_params[name].requires_grad

    # A training checkpoint cannot be reinterpreted in another parameter basis.
    e3nn_opt = get_optimizer(args, get_params_options(args, source))
    e3nn_scheduler = torch.optim.lr_scheduler.ExponentialLR(e3nn_opt, gamma=0.9)
    with pytest.raises(L2SPCheckpointError):
        CheckpointBuilder.load_checkpoint(
            CheckpointState(source, e3nn_opt, e3nn_scheduler), checkpoint, strict=False
        )

    exported = to_e3nn(deepcopy(model).cpu(), device="cpu")
    path = tmp_path / "export.model"
    torch.save(exported, path)
    atoms = load_training_structures(limit=1)[0].copy()
    atoms.calc = MACECalculator(model_paths=path, device="cpu", default_dtype="float64")
    expected = model(
        anchor_batch(model, [atoms]).to(backend["device"]).to_dict(),
        compute_force=True,
        compute_stress=True,
    )
    assert atoms.get_potential_energy() == pytest.approx(
        expected["energy"].item(), abs=2e-10
    )
    np.testing.assert_allclose(
        atoms.get_forces(),
        expected["forces"].detach().cpu().numpy(),
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False),
        expected["stress"][0].detach().cpu().numpy(),
        atol=2e-10,
        rtol=2e-10,
    )
