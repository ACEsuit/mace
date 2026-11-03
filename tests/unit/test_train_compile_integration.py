import copy
import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close
from torch_ema import ExponentialMovingAverage

from mace.cli import run_train
from mace.tools import compile as mace_compile
from mace.tools.checkpoint import CheckpointBuilder

train_tools = importlib.import_module("mace.tools.train")


class _Batch:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def to(self, device: torch.device):
        self.values = self.values.to(device)
        return self

    def to_dict(self):
        return {"values": self.values}


class _EnergyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float64))

    def forward(self, data, **kwargs):
        del kwargs
        return {"energy": self.weight * data["values"]}


class _ForwardOnly(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        object.__setattr__(self, "base_model", base_model)

    def forward(self, data):
        return self.base_model(data)

    def parameters(self, recurse=True):
        del recurse
        raise AssertionError("compiled forward wrapper must not own clipping")


class _EnergyLoss(torch.nn.Module):
    def forward(self, pred, ref):
        del ref
        return pred["energy"].square().mean()


class _Logger:
    def log(self, metrics):
        del metrics


def _assert_optimizer_states_close(left, right):
    for left_group, right_group in zip(left.param_groups, right.param_groups):
        for left_param, right_param in zip(left_group["params"], right_group["params"]):
            assert left.state[left_param].keys() == right.state[right_param].keys()
            for key, left_value in left.state[left_param].items():
                right_value = right.state[right_param][key]
                if torch.is_tensor(left_value):
                    assert_close(left_value, right_value)
                else:
                    assert left_value == right_value


def _call_public_train_with_invalid_compile_configuration(
    *, monkeypatch=None, distributed=False, lbfgs=False, unsupported_output=None
):
    model = _EnergyModel()
    optimizer = (
        torch.optim.LBFGS(model.parameters())
        if lbfgs
        else torch.optim.Adam(model.parameters())
    )
    output_args = {
        "energy": True,
        "forces": True,
        "virials": False,
        "stress": False,
    }
    if unsupported_output is not None:
        output_args[unsupported_output] = True
    valid_loaders = {}
    if monkeypatch is not None:
        monkeypatch.setattr(
            train_tools,
            "evaluate",
            lambda **kwargs: (0.0, {}),
        )
        valid_loaders = {"valid": object()}
    train_tools.train(
        model=model,
        loss_fn=_EnergyLoss(),
        train_loader=[],
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=None,
        start_epoch=0,
        max_num_epochs=0,
        patience=1,
        checkpoint_handler=None,
        logger=_Logger(),
        eval_interval=1,
        output_args=output_args,
        device=torch.device("cpu"),
        log_errors="",
        distributed=distributed,
        compile_training=True,
    )


def test_take_step_uses_forward_wrapper_but_eager_model_owns_training_state():
    eager_model = _EnergyModel()
    candidate_model = copy.deepcopy(eager_model)
    eager_optimizer = torch.optim.Adam(eager_model.parameters(), lr=0.01, amsgrad=True)
    candidate_optimizer = torch.optim.Adam(
        candidate_model.parameters(), lr=0.01, amsgrad=True
    )
    eager_ema = ExponentialMovingAverage(eager_model.parameters(), decay=0.9)
    candidate_ema = ExponentialMovingAverage(candidate_model.parameters(), decay=0.9)
    loss_fn = _EnergyLoss()
    output_args = {"forces": False, "virials": False, "stress": False}

    train_tools.take_step(
        model=eager_model,
        loss_fn=loss_fn,
        batch=_Batch(torch.tensor([1.0, -2.0], dtype=torch.float64)),
        optimizer=eager_optimizer,
        ema=eager_ema,
        output_args=output_args,
        max_grad_norm=0.2,
        device=torch.device("cpu"),
    )
    train_tools.take_step(
        model=candidate_model,
        forward_model=_ForwardOnly(candidate_model),
        loss_fn=loss_fn,
        batch=_Batch(torch.tensor([1.0, -2.0], dtype=torch.float64)),
        optimizer=candidate_optimizer,
        ema=candidate_ema,
        output_args=output_args,
        max_grad_norm=0.2,
        device=torch.device("cpu"),
    )

    assert_close(eager_model.weight, candidate_model.weight)
    _assert_optimizer_states_close(eager_optimizer, candidate_optimizer)
    for eager_shadow, candidate_shadow in zip(
        eager_ema.shadow_params, candidate_ema.shadow_params
    ):
        assert_close(eager_shadow, candidate_shadow)


def test_training_conversion_disables_convolution_fusion(monkeypatch):
    calls = []
    source = object()
    converted = object()
    monkeypatch.setattr(run_train, "deepcopy", lambda model: model)
    monkeypatch.setattr(
        run_train,
        "run_e3nn_to_cueq",
        lambda model, **kwargs: calls.append((model, kwargs)) or converted,
    )

    assert run_train._convert_e3nn_to_cueq_for_training(source, "cuda") is converted
    assert calls == [
        (source, {"device": "cuda", "conv_fusion": False, "conv_fusion_node_bucket": 256})
    ]


def test_compiled_training_conversion_disables_e3nn_codegen(monkeypatch):
    events = []
    source = object()
    converted = object()

    @contextmanager
    def fake_disable_e3nn_codegen():
        events.append("enter_codegen_disabled")
        yield
        events.append("exit_codegen_disabled")

    monkeypatch.setattr(run_train, "deepcopy", lambda model: model)
    monkeypatch.setattr(
        run_train.mace_compile,
        "disable_e3nn_codegen",
        fake_disable_e3nn_codegen,
    )
    monkeypatch.setattr(
        run_train,
        "run_e3nn_to_cueq",
        lambda model, **kwargs: events.append((model, kwargs)) or converted,
    )

    assert (
        run_train._convert_e3nn_to_cueq_for_training(
            source,
            "cuda",
            compile_training=True,
        )
        is converted
    )
    assert events == [
        "enter_codegen_disabled",
        (
            source,
            {"device": "cuda", "conv_fusion": False, "conv_fusion_node_bucket": 256},
        ),
        "exit_codegen_disabled",
    ]


def test_train_one_epoch_compiles_first_real_batch_once_and_reuses_wrapper(
    monkeypatch,
):
    owner = _EnergyModel()
    optimizer = torch.optim.Adam(owner.parameters())
    wrapper = object()
    compile_calls = []
    step_calls = []

    def fake_compile(
        model,
        example_batch,
        output_args,
        mode,
        dynamic,
        validation_batch,
    ):
        compile_calls.append(
            (model, example_batch, validation_batch, output_args, mode, dynamic)
        )
        return wrapper

    def fake_take_step(**kwargs):
        step_calls.append(kwargs)
        return torch.tensor(0.0), {"loss": 0.0, "time": 0.0}

    monkeypatch.setattr(mace_compile, "compile_training_model", fake_compile)
    monkeypatch.setattr(train_tools, "take_step", fake_take_step)
    output_args = {"forces": True, "virials": False, "stress": False}

    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([1.0])), _Batch(torch.tensor([2.0]))],
        optimizer=optimizer,
        epoch=0,
        output_args=output_args,
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=False,
        compile_training=True,
        compiled_training_model=None,
    )
    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([3.0]))],
        optimizer=optimizer,
        epoch=1,
        output_args=output_args,
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=False,
        compile_training=True,
        compiled_training_model=cached,
    )

    assert cached is wrapper
    assert len(compile_calls) == 1
    assert compile_calls[0][0] is owner
    assert_close(compile_calls[0][1]["values"], torch.tensor([1.0]))
    assert_close(compile_calls[0][2]["values"], torch.tensor([2.0]))
    assert [float(call["batch"].values) for call in step_calls] == [1.0, 2.0, 3.0]
    assert all(call["model"] is owner for call in step_calls)
    assert all(call["forward_model"] is wrapper for call in step_calls)


def test_train_one_epoch_one_batch_falls_back_without_dropping_step(monkeypatch):
    owner = _EnergyModel()
    optimizer = torch.optim.Adam(owner.parameters())
    eager_facade = object()
    compile_calls = []
    step_calls = []

    def fake_compile(
        model,
        example_batch,
        output_args,
        mode,
        dynamic,
        validation_batch,
    ):
        compile_calls.append((example_batch, validation_batch))
        return eager_facade

    def fake_take_step(**kwargs):
        step_calls.append(kwargs)
        return torch.tensor(0.0), {"loss": 0.0, "time": 0.0}

    monkeypatch.setattr(mace_compile, "compile_training_model", fake_compile)
    monkeypatch.setattr(train_tools, "take_step", fake_take_step)

    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([1.0]))],
        optimizer=optimizer,
        epoch=0,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=False,
        compile_training=True,
        compiled_training_model=None,
    )

    assert cached is eager_facade
    assert len(compile_calls) == 1
    assert_close(compile_calls[0][0]["values"], torch.tensor([1.0]))
    assert compile_calls[0][1] is None
    assert len(step_calls) == 1
    assert_close(step_calls[0]["batch"].values, torch.tensor([1.0]))


def test_train_one_epoch_wraps_compiled_facade_once_with_existing_ddp(monkeypatch):
    owner = _EnergyModel()
    optimizer = torch.optim.Adam(owner.parameters())

    class Facade:
        def materialize_compiled_backward(self, batch):
            events.append(("materialize_backward", self, batch))

    facade = Facade()
    ddp_wrapper = object()
    events = []
    step_calls = []

    monkeypatch.setattr(
        mace_compile,
        "compile_training_model",
        lambda *args, **kwargs: events.append("compile") or facade,
    )

    def fake_synchronize(model, **kwargs):
        events.append(("sync", model, kwargs))
        return model

    def fake_ddp(model, device_ids):
        events.append(("ddp", model, device_ids))
        return ddp_wrapper

    def fake_take_step(**kwargs):
        events.append("step")
        step_calls.append(kwargs)
        return torch.tensor(0.0), {"loss": 0.0, "time": 0.0}

    monkeypatch.setattr(
        train_tools, "_synchronize_compiled_training_state", fake_synchronize
    )
    monkeypatch.setattr(train_tools, "DistributedDataParallel", fake_ddp)
    monkeypatch.setattr(train_tools, "take_step", fake_take_step)

    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([1.0])), _Batch(torch.tensor([2.0]))],
        optimizer=optimizer,
        epoch=0,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=True,
        distributed_device_ids=None,
        compile_training=True,
    )
    cached = train_tools.train_one_epoch(
        model=owner,
        loss_fn=_EnergyLoss(),
        data_loader=[_Batch(torch.tensor([3.0]))],
        optimizer=optimizer,
        epoch=1,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=_Logger(),
        device=torch.device("cpu"),
        distributed=True,
        distributed_device_ids=None,
        compile_training=True,
        compiled_training_model=cached,
    )

    assert cached is ddp_wrapper
    assert events.count("compile") == 1
    assert events.count("step") == 3
    assert [
        event for event in events if isinstance(event, tuple) and event[0] == "ddp"
    ] == [("ddp", facade, None)]
    materialize_events = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "materialize_backward"
    ]
    assert len(materialize_events) == 1
    assert materialize_events[0][1] is facade
    assert events.index(materialize_events[0]) < events.index(("ddp", facade, None))
    assert all(call["model"] is owner for call in step_calls)
    assert all(call["forward_model"] is ddp_wrapper for call in step_calls)
    assert [
        event[1] for event in events if isinstance(event, tuple) and event[0] == "sync"
    ] == [
        facade,
        ddp_wrapper,
        ddp_wrapper,
        ddp_wrapper,
    ]


def test_synchronize_compiled_training_state_disables_compiled_peer(monkeypatch):
    class Facade:
        is_compiled = True

        def __init__(self):
            self.reason = None

        def disable_compilation(self, reason):
            self.is_compiled = False
            self.reason = reason

    facade = Facade()

    def fake_all_reduce(flag, op):
        assert op is torch.distributed.ReduceOp.MIN
        flag.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    result = train_tools._synchronize_compiled_training_state(
        facade, device=torch.device("cpu"), distributed=True
    )

    assert result is facade
    assert not facade.is_compiled
    assert facade.reason == "another distributed rank entered eager fallback"


@pytest.mark.parametrize("compile_training", [False, True])
def test_eager_ddp_construction_is_deferred_for_compiled_training(
    monkeypatch, compile_training
):
    model = object()
    wrapped = object()
    calls = []
    monkeypatch.setattr(
        run_train,
        "DDP",
        lambda candidate, device_ids: calls.append((candidate, device_ids)) or wrapped,
    )

    result = run_train._build_eager_distributed_model(
        model,
        distributed=True,
        compile_training=compile_training,
        device_ids=[3],
    )

    if compile_training:
        assert result is None
        assert calls == []
    else:
        assert result is wrapped
        assert calls == [(model, [3])]


@pytest.mark.parametrize("unsupported", ["lbfgs"])
def test_compile_training_rejects_unsupported_modes_before_setup(unsupported):
    args = SimpleNamespace(compile_training=True, distributed=False, lbfgs=False)
    setattr(args, unsupported, True)

    with pytest.raises(ValueError, match=unsupported):
        run_train._validate_compile_training_args(args)


def test_compile_training_accepts_distributed_mode_before_setup():
    run_train._validate_compile_training_args(
        SimpleNamespace(compile_training=True, distributed=True, lbfgs=False)
    )


@pytest.mark.parametrize("model", ["PolarMACE", "MACELES", "AtomicDielectricMACE"])
def test_compile_training_rejects_non_standard_models(model):
    args = SimpleNamespace(
        compile_training=True, distributed=False, lbfgs=False, model=model
    )

    with pytest.raises(ValueError, match="standard MACE"):
        run_train._validate_compile_training_args(args)


@pytest.mark.parametrize(
    "model",
    ["MACE", "ScaleShiftMACE", "FoundationMACE", "MagneticScaleShiftMACE"],
)
def test_compile_training_accepts_supported_models(model):
    run_train._validate_compile_training_args(
        SimpleNamespace(
            compile_training=True, distributed=False, lbfgs=False, model=model
        )
    )


@pytest.mark.parametrize("unsupported", ["lbfgs"])
def test_public_train_rejects_unsupported_compiled_modes(unsupported):
    with pytest.raises(ValueError, match=unsupported):
        _call_public_train_with_invalid_compile_configuration(
            distributed=unsupported == "distributed",
            lbfgs=unsupported == "lbfgs",
        )


def test_public_train_accepts_distributed_compiled_mode(monkeypatch):
    _call_public_train_with_invalid_compile_configuration(
        monkeypatch=monkeypatch,
        distributed=True,
    )


def test_compiled_training_checkpoint_optimizer_and_ema_keep_base_ownership(
    monkeypatch,
):
    owner = _EnergyModel()
    optimizer = torch.optim.Adam(owner.parameters())
    ema = ExponentialMovingAverage(owner.parameters(), decay=0.9)
    saved_states = []
    evaluation_losses = iter([1.0, 0.5])

    monkeypatch.setattr(
        train_tools,
        "evaluate",
        lambda **kwargs: (next(evaluation_losses), {}),
    )
    monkeypatch.setattr(
        train_tools,
        "train_one_epoch",
        lambda **kwargs: object(),
    )
    checkpoint_handler = SimpleNamespace(
        save=lambda **kwargs: saved_states.append(kwargs["state"])
    )
    scheduler = SimpleNamespace(state_dict=lambda: {})

    train_tools.train(
        model=owner,
        loss_fn=_EnergyLoss(),
        train_loader=[],
        valid_loaders={"valid": object()},
        optimizer=optimizer,
        lr_scheduler=scheduler,
        start_epoch=0,
        max_num_epochs=1,
        patience=2,
        checkpoint_handler=checkpoint_handler,
        logger=_Logger(),
        eval_interval=1,
        output_args={
            "energy": True,
            "forces": False,
            "virials": False,
            "stress": False,
        },
        device=torch.device("cpu"),
        log_errors="",
        ema=ema,
        compile_training=True,
    )

    assert len(saved_states) == 1
    saved_state = saved_states[0]
    assert saved_state.model is owner
    checkpoint = CheckpointBuilder.create_checkpoint(saved_state)
    assert checkpoint["model"].keys() == owner.state_dict().keys()
    assert not any(
        key.startswith(("module.", "base_model.")) for key in checkpoint["model"]
    )
    owner_parameters = list(owner.parameters())
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(optimizer_parameters) == len(owner_parameters)
    assert all(
        optimizer_parameter is owner_parameter
        for optimizer_parameter, owner_parameter in zip(
            optimizer_parameters, owner_parameters
        )
    )
    ema_parameters = [parameter_ref() for parameter_ref in ema._params_refs]
    assert len(ema_parameters) == len(owner_parameters)
    assert all(
        ema_parameter is owner_parameter
        for ema_parameter, owner_parameter in zip(ema_parameters, owner_parameters)
    )


@pytest.mark.parametrize(
    "unsupported_output",
    ["dipoles", "polarizabilities", "magforces", "hessian"],
)
def test_public_train_rejects_truthy_unsupported_compiled_outputs(
    unsupported_output,
):
    with pytest.raises(ValueError, match=unsupported_output):
        _call_public_train_with_invalid_compile_configuration(
            unsupported_output=unsupported_output
        )


@pytest.mark.parametrize("compile_training", [False, True])
def test_configure_model_disables_e3nn_codegen_only_for_compiled_training(
    monkeypatch, compile_training
):
    events = []
    args = SimpleNamespace(compile_training=compile_training)

    @contextmanager
    def fake_disable_e3nn_codegen():
        events.append("enter")
        yield
        events.append("exit")

    def fake_configure_model(*args):
        events.append("configure")
        assert args == (
            expected_args,
            "loader",
            "energies",
            "foundation",
            "heads",
            "z",
            "configs",
        )
        return "model", "outputs"

    monkeypatch.setattr(
        run_train.mace_compile,
        "disable_e3nn_codegen",
        fake_disable_e3nn_codegen,
    )
    monkeypatch.setattr(run_train, "configure_model", fake_configure_model)
    expected_args = args

    result = run_train._configure_model_for_training(
        args,
        "loader",
        "energies",
        "foundation",
        "heads",
        "z",
        "configs",
    )

    assert result == ("model", "outputs")
    assert events == (
        ["enter", "configure", "exit"] if compile_training else ["configure"]
    )


def test_configure_model_restores_e3nn_codegen_when_construction_raises(monkeypatch):
    optimization_defaults = {"jit_script_fx": True}

    monkeypatch.setattr(
        run_train.mace_compile,
        "get_optimization_defaults",
        lambda: dict(optimization_defaults),
    )
    monkeypatch.setattr(
        run_train.mace_compile,
        "set_optimization_defaults",
        lambda **kwargs: optimization_defaults.update(kwargs),
    )

    def raise_from_configure_model(*args):
        del args
        raise RuntimeError("synthetic construction failure")

    monkeypatch.setattr(run_train, "configure_model", raise_from_configure_model)

    with pytest.raises(RuntimeError, match="synthetic construction failure"):
        run_train._configure_model_for_training(
            SimpleNamespace(compile_training=True), "loader"
        )

    assert optimization_defaults["jit_script_fx"] is True
