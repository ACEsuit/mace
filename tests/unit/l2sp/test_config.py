"""Selection and argument contracts for the paper's first-two-layer recipe."""

import json
from types import SimpleNamespace

import pytest
import torch

from mace.tools import build_default_arg_parser
from mace.tools.l2sp import L2SPRegularization, validate_args
from mace.tools.scripts_utils import get_optimizer, get_params_options
from tests.golden.anchors import load_anchor


def args_for(**overrides):
    args = build_default_arg_parser().parse_args(["--name=test"])
    vars(args).update(
        l2sp=True,
        freeze="products,interactions.1,readouts",
        optimizer="adamw",
        weight_decay=10.0,
        foundation_model="foundation.model",
        multiheads_finetuning=False,
    )
    vars(args).update(overrides)
    return args


def test_selected_parameters_and_learning_rates():
    model = load_anchor("tiny_scaleshift")
    args = args_for(
        lr_params_factors=json.dumps(
            {"embedding_lr_factor": 0.5, "interactions_lr_factor": 0.25}
        )
    )
    validate_args(args)
    optimizer = get_optimizer(args, get_params_options(args, model))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(optimizer.l2sp, L2SPRegularization)
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(("node_embedding.", "interactions.0."))
    }
    assert {
        name for name, param in model.named_parameters() if param.requires_grad
    } == expected
    assert {
        name for group in optimizer.param_groups for name in group["param_names"]
    } == expected
    assert {group["name"]: group["lr"] for group in optimizer.param_groups} == {
        "embedding": 0.005,
        "interactions": 0.0025,
    }
    assert all(group["weight_decay"] == 0 for group in optimizer.param_groups)


def test_all_includes_every_trainable_parameter():
    model = load_anchor("tiny_scaleshift")
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    options = get_params_options(args_for(freeze=None), model)
    assert {
        name for group in options["params"] for name in group["param_names"]
    } == expected


@pytest.mark.parametrize("freeze", [None, 0, 1, 6, "products,interactions.1,readouts"])
def test_l2sp_accepts_existing_and_named_freezing(freeze):
    args = args_for(freeze=freeze)
    validate_args(args)
    model = load_anchor("tiny_scaleshift")
    options = get_params_options(args, model)
    assert {name for group in options["params"] for name in group["param_names"]} == {
        name for name, p in model.named_parameters() if p.requires_grad
    }


def test_oeq_backend_inherited_from_foundation_is_checked():
    model = load_anchor("tiny_scaleshift")
    model.oeq_config = SimpleNamespace(enabled=True)
    with pytest.raises(ValueError, match="OEQ backend"):
        get_params_options(args_for(), model)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_cueq": True},
        {"only_cueq": True},
        {"enable_cueq": True, "enable_oeq": True},
    ],
)
def test_cueq_arguments_follow_existing_backend_selection(overrides):
    validate_args(args_for(**overrides))


@pytest.mark.parametrize(
    "selection",
    [
        "",
        "interactions.99",
        "node_embedding,",
        "all,readouts",
        "atomic_energies_fn",
        "interactions.0.linear.weight",
    ],
)
def test_invalid_selection_is_not_silently_ignored(selection):
    with pytest.raises(ValueError, match="--freeze"):
        get_params_options(args_for(freeze=selection), load_anchor("tiny_scaleshift"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_decay": None},
        {"weight_decay": -1},
        {"weight_decay": float("nan")},
        {"weight_decay": float("inf")},
        {"foundation_model": None},
        {"multiheads_finetuning": True},
        {"lora": True},
        {"foundation_model_readout": False},
        {"enable_oeq": True},
        {"embedding_specs": "{}"},
    ],
)
def test_incompatible_options_fail_early(overrides):
    with pytest.raises(ValueError, match="l2sp"):
        validate_args(args_for(**overrides))


def test_disabled_defaults_preserve_existing_training():
    args = build_default_arg_parser().parse_args(["--name=test"])
    validate_args(args)
    assert args.l2sp is False
    assert not hasattr(args, "l2sp_strength")
    assert not hasattr(args, "l2sp_mode")
    assert args.optimizer == "adam"
    assert args.weight_decay == 5e-7


def test_l2sp_uses_existing_weight_decay_default():
    args = build_default_arg_parser().parse_args(
        [
            "--name=test",
            "--l2sp=True",
            "--foundation_model=foundation.model",
            "--multiheads_finetuning=False",
        ]
    )
    validate_args(args)
    model = load_anchor("tiny_scaleshift")
    optimizer = get_optimizer(args, get_params_options(args, model))
    assert args.weight_decay == 5e-7
    assert optimizer.l2sp.strength == args.weight_decay


@pytest.mark.parametrize(
    "argv, expected", [([], True), (["--amsgrad"], True), (["--amsgrad=False"], False)]
)
def test_amsgrad_can_match_equitrain_without_changing_default(argv, expected):
    args = build_default_arg_parser().parse_args(["--name=test", *argv])
    assert args.amsgrad is expected


def test_yaml_config_selects_same_recipe(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: test\nl2sp: true\nfreeze: products,interactions.1,readouts\noptimizer: adamw\namsgrad: false\nweight_decay: 10.0\nfoundation_model: foundation.model\nmultiheads_finetuning: false\n"
    )
    args = build_default_arg_parser().parse_args(["--config", str(path)])
    validate_args(args)
    assert args.l2sp is True
    assert args.weight_decay == 10.0
    assert args.amsgrad is False
    assert args.freeze == "products,interactions.1,readouts"
