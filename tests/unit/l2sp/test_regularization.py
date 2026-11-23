"""Native optimizer parity and exact resume semantics for L2-SP."""

from copy import deepcopy

import pytest
import torch

from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.l2sp import L2SPCheckpointError, L2SPRegularization, validate_args
from mace.tools.scripts_utils import get_optimizer, get_params_options
from mace.tools.train import take_step, take_step_lbfgs
from tests.golden.anchors import load_anchor
from tests.unit.l2sp.test_config import args_for

OUTPUTS = {"forces": False, "stress": False, "virials": False}
OPTIMIZERS = [
    "adam",
    "adamw",
    "sgd",
    pytest.param("schedulefree", marks=pytest.mark.schedulefree),
]


class Batch:
    def __init__(self, size):
        self.num_graphs = size
        self.x = torch.ones(size, dtype=torch.float64)
        self.target = 2.0 * self.x

    def to(self, _device):
        return self

    def to_dict(self):
        return {"x": self.x}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def forward(self, batch, **_kwargs):
        return {"prediction": self.weight * batch["x"]}


def data_loss(pred, ref):
    return (pred["prediction"] - ref.target).square().mean()


def make_optimizer(model, kind, strength=3.0, regularize=True):
    params = [{"params": [model.weight], "param_names": ["weight"]}]
    if kind == "sgd":
        optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9)
    elif kind == "schedulefree":
        from schedulefree import AdamWScheduleFree

        optimizer = AdamWScheduleFree(params, lr=0.05, weight_decay=0)
        optimizer.train()
    elif kind == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )
    else:
        cls = torch.optim.AdamW if kind == "adamw" else torch.optim.Adam
        optimizer = cls(params, lr=0.05, weight_decay=0)
    if regularize:
        optimizer.l2sp = L2SPRegularization(optimizer, strength)
    return optimizer


@pytest.mark.parametrize("kind", OPTIMIZERS)
@pytest.mark.parametrize("foreach", [False, True])
def test_updates_match_native_weight_decay_on_residuals(kind, foreach):
    model, oracle = Model(), Model()
    optimizer = make_optimizer(model, kind)
    reference_optimizer = make_optimizer(oracle, kind, regularize=False)
    reference = oracle.weight.detach().clone()
    with torch.no_grad():
        oracle.weight.zero_()
    for opt in (optimizer, reference_optimizer):
        opt.param_groups[0]["foreach"] = foreach
        if kind == "schedulefree":
            opt.param_groups[0]["warmup_steps"] = 3
    reference_optimizer.param_groups[0]["weight_decay"] = 3.0
    batch = Batch(3)
    for step in range(6):
        for opt in (optimizer, reference_optimizer):
            opt.param_groups[0]["lr"] = 0.05 / (step + 1)
        reference_optimizer.zero_grad(set_to_none=True)
        expected_loss = data_loss(
            {"prediction": (reference + oracle.weight) * batch.x}, batch
        )
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 0.5)
        reference_optimizer.step()
        loss, metrics = take_step(
            model, data_loss, batch, optimizer, None, OUTPUTS, 0.5, torch.device("cpu")
        )
        torch.testing.assert_close(loss, expected_loss, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(
            model.weight, reference + oracle.weight, atol=1e-13, rtol=1e-13
        )
        if kind == "schedulefree":
            torch.testing.assert_close(
                optimizer.state[model.weight]["z"],
                reference + reference_optimizer.state[oracle.weight]["z"],
                atol=1e-13,
                rtol=1e-13,
            )
            optimizer.eval()
            reference_optimizer.eval()
            torch.testing.assert_close(
                model.weight, reference + oracle.weight, atol=1e-13, rtol=1e-13
            )
            optimizer.train()
            reference_optimizer.train()
        assert metrics["l2sp_squared_distance"] == pytest.approx(
            (model.weight - reference).square().item()
        )
    torch.testing.assert_close(
        optimizer.l2sp.references[0][0], reference, atol=0, rtol=0
    )


@pytest.mark.parametrize("batch_sizes", [[8], [4, 4], [1, 3, 4]])
def test_lbfgs_counts_penalty_once_per_closure_independent_of_batch_partition(
    batch_sizes,
):
    model = Model()
    optimizer = make_optimizer(model, "lbfgs")
    _, metrics = take_step_lbfgs(
        model,
        data_loss,
        [Batch(n) for n in batch_sizes],
        optimizer,
        None,
        OUTPUTS,
        None,
        torch.device("cpu"),
        False,
        0,
    )
    # Minimizer of (w - 2)^2 + 3/2 * (w - 0.4)^2.
    assert model.weight.item() == pytest.approx(1.04, abs=1e-9)
    assert metrics["l2sp_squared_distance"] == pytest.approx(0.64**2, abs=1e-9)


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_lbfgs_distributed_normalization_for_identical_replicas(
    monkeypatch, world_size
):
    # Identical replicas have identical local gradients, so DDP's gradient
    # average is the identity. SUM collectives still multiply losses/counts.
    # This checks normalization without requiring a process-group backend.
    @torch.no_grad()
    def sum_replicas(tensor, **_kwargs):
        tensor.mul_(world_size)

    monkeypatch.setattr(torch.distributed, "all_reduce", sum_replicas)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    model = Model()
    optimizer = make_optimizer(model, "lbfgs")
    take_step_lbfgs(
        model,
        data_loss,
        [Batch(2), Batch(3)],
        optimizer,
        None,
        OUTPUTS,
        None,
        torch.device("cpu"),
        True,
        0,
    )
    assert model.weight.item() == pytest.approx(1.04, abs=1e-9)


@pytest.mark.parametrize("kind", [*OPTIMIZERS, "lbfgs"])
def test_penalty_checkpoint_restores_initial_reference_and_continuation(kind):
    model = Model()
    optimizer = make_optimizer(model, kind)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    state = CheckpointState(model, optimizer, scheduler)

    def step(candidate, opt):
        if kind == "lbfgs":
            take_step_lbfgs(
                candidate,
                data_loss,
                [Batch(3)],
                opt,
                None,
                OUTPUTS,
                None,
                torch.device("cpu"),
                False,
                0,
            )
        else:
            take_step(
                candidate,
                data_loss,
                Batch(3),
                opt,
                None,
                OUTPUTS,
                None,
                torch.device("cpu"),
            )

    step(model, optimizer)
    saved = deepcopy(CheckpointBuilder.create_checkpoint(state))
    resumed = deepcopy(model)
    resumed_optimizer = make_optimizer(resumed, kind)
    resumed_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        resumed_optimizer, gamma=0.9
    )
    CheckpointBuilder.load_checkpoint(
        CheckpointState(resumed, resumed_optimizer, resumed_scheduler),
        saved,
        strict=False,
    )
    assert resumed_optimizer.l2sp.references[0][0].item() == 0.4
    for _ in range(3):
        step(model, optimizer)
        step(resumed, resumed_optimizer)
        torch.testing.assert_close(model.weight, resumed.weight, atol=0, rtol=0)


@pytest.mark.parametrize(
    "change",
    ["rule", "strength", "optimizer_kind", "names", "shape", "missing", "model"],
)
def test_incompatible_checkpoint_fails_before_loading_model(change):
    model = Model()
    optimizer = make_optimizer(model, "adam")
    state = CheckpointState(
        model, optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    )
    saved = deepcopy(CheckpointBuilder.create_checkpoint(state))
    saved["model"]["weight"].fill_(99)
    if change in ("rule", "optimizer_kind"):
        saved["optimizer"]["l2sp"][change] = "other"
    elif change == "strength":
        saved["optimizer"]["l2sp"][change] = 123
    elif change == "names":
        saved["optimizer"]["param_groups"][0]["param_names"] = ["wrong"]
    elif change == "shape":
        saved["optimizer"]["l2sp"]["references"][0][0] = torch.zeros(3)
    elif change == "missing":
        del saved["optimizer"]["l2sp"]
    else:
        del saved["model"]["weight"]
    with pytest.raises(L2SPCheckpointError):
        CheckpointBuilder.load_checkpoint(state, saved, strict=False)
    assert model.weight.item() == 0.4


@pytest.mark.parametrize(
    "kind",
    [
        "adam",
        "adamw",
        pytest.param("schedulefree", marks=pytest.mark.schedulefree),
        "lbfgs",
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 0.75, 10.0])
def test_mace_optimizer_factory_uses_native_optimizers(kind, weight_decay):
    args = args_for(
        optimizer="adam" if kind == "lbfgs" else kind,
        lbfgs=kind == "lbfgs",
        weight_decay=weight_decay,
    )
    validate_args(args)
    model = load_anchor("tiny_scaleshift")
    optimizer = get_optimizer(args, get_params_options(args, model))
    assert isinstance(optimizer.l2sp, L2SPRegularization)
    assert optimizer.l2sp.strength == weight_decay
    assert all(group.get("weight_decay", 0.0) == 0 for group in optimizer.param_groups)
    if kind == "lbfgs":
        assert isinstance(optimizer, torch.optim.LBFGS)
    else:
        assert (
            type(optimizer).__name__
            == {"adam": "Adam", "adamw": "AdamW", "schedulefree": "AdamWScheduleFree"}[
                kind
            ]
        )


def test_zero_strength_does_not_create_gradients_for_unused_parameters():
    model = Model()
    unused = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [{"params": [model.weight, unused], "param_names": ["weight", "unused"]}]
    )
    optimizer.l2sp = L2SPRegularization(optimizer, 0.0)
    take_step(
        model, data_loss, Batch(3), optimizer, None, OUTPUTS, None, torch.device("cpu")
    )
    assert unused.grad is None
    optimizer.l2sp.add_gradients()
    assert unused.grad is None


def test_penalty_rejects_ordinary_weight_decay():
    model = Model()
    optimizer = torch.optim.AdamW(
        [{"params": [model.weight], "param_names": ["weight"]}], weight_decay=0.1
    )
    with pytest.raises(ValueError, match="weight_decay=0"):
        L2SPRegularization(optimizer, 3.0)


@pytest.mark.parametrize(
    "kind,compatible", [("adam", False), ("adamw", False), ("lbfgs", True)]
)
def test_legacy_explicit_loss_checkpoint_compatibility(kind, compatible):
    model = Model()
    optimizer = make_optimizer(model, kind)
    state = CheckpointState(
        model, optimizer, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    )
    saved = deepcopy(CheckpointBuilder.create_checkpoint(state))
    metadata = saved["optimizer"]["l2sp"]
    metadata["version"] = 2
    metadata["mode"] = "loss"
    del metadata["rule"]
    if compatible:
        CheckpointBuilder.load_checkpoint(state, saved, strict=False)
    else:
        with pytest.raises(L2SPCheckpointError, match="regularization rule"):
            CheckpointBuilder.load_checkpoint(state, saved, strict=False)
