"""Named-module freezing works independently of the model and fine-tuning method."""

from copy import deepcopy

import pytest
import torch

# Initialise tools before the plotter, which is imported by tools.train.
import mace.tools  # noqa: F401  # pylint: disable=unused-import
from mace.cli.visualise_train import model_inference
from mace.modules.lora import inject_LoRAs
from mace.modules.loss import WeightedEnergyForcesLoss
from mace.tools import build_default_arg_parser, torch_tools
from mace.tools.parameter_freezing import freeze_named_modules
from mace.tools.scripts_utils import get_optimizer, get_params_options
from mace.tools.train import take_step
from tests.golden.anchors import anchor_batch, load_anchor, load_training_structures


def args_for(**overrides):
    args = build_default_arg_parser().parse_args(["--name=selection"])
    args.freeze = "products,interactions.1,readouts"
    vars(args).update(overrides)
    return args


@pytest.mark.parametrize("method", ["adam", "adamw", "lora"])
def test_selected_layers_train_and_remain_frozen_through_plotting(method):
    with torch_tools.default_dtype("float64"):
        torch.manual_seed(17)
        model = load_anchor("tiny_scaleshift")
        if method == "lora":
            model = inject_LoRAs(model, rank=2, alpha=1)
        eligible = {
            name for name, param in model.named_parameters() if param.requires_grad
        }
        initial = deepcopy(model.state_dict())
        args = args_for(optimizer="adamw" if method == "lora" else method)
        optimizer = get_optimizer(args, get_params_options(args, model))
        expected = {
            name
            for name in eligible
            if name.startswith(("node_embedding.", "interactions.0."))
        }
        parameters = dict(model.named_parameters())
        assert {
            name for name, param in parameters.items() if param.requires_grad
        } == expected
        registered = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        assert registered == {id(p) for p in parameters.values()}
        assert {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
            if p.requires_grad
        } == {id(parameters[name]) for name in expected}
        batch = anchor_batch(model, load_training_structures(limit=2))
        loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=10.0).double()
        outputs = {"forces": True, "stress": False, "virials": False}
        for _ in range(2):
            take_step(
                model,
                loss,
                batch.clone(),
                optimizer,
                None,
                outputs,
                1.0,
                torch.device("cpu"),
            )
            results = model_inference({"train": [batch.clone()]}, model, outputs, "cpu")
            assert results["train"]["energy"] is not None
            assert {
                name for name, param in parameters.items() if param.requires_grad
            } == expected
        assert any(
            not torch.equal(parameters[name], initial[name]) for name in expected
        )
        for name, param in parameters.items():
            if name not in expected:
                torch.testing.assert_close(param, initial[name], atol=0, rtol=0)


def test_selection_is_generic_and_handles_shared_parameters():
    layer = torch.nn.Linear(2, 2)
    model = torch.nn.ModuleDict(
        {"first": layer, "tied": layer, "last": torch.nn.Linear(2, 1)}
    )
    layer.bias.requires_grad_(False)
    freeze_named_modules(model, "tied")
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert all(param.requires_grad for param in model["last"].parameters())


def test_default_keeps_existing_freezing_and_default_groups():
    model = load_anchor("tiny_scaleshift")
    model.node_embedding.requires_grad_(False)
    assert all(not p.requires_grad for p in model.node_embedding.parameters())
    args = args_for()
    args.freeze = None
    options = get_params_options(args, model)
    assert [group["name"] for group in options["params"]][:5] == [
        "embedding",
        "interactions_decay",
        "interactions_no_decay",
        "products",
        "readouts",
    ]


def test_ordinary_weight_decay_and_lr_multipliers_are_preserved():
    model = load_anchor("tiny_scaleshift")
    args = args_for(
        weight_decay=0.1,
        lr=0.02,
        lr_params_factors='{"embedding_lr_factor":0.5,"interactions_lr_factor":0.25}',
    )
    groups = {
        group["name"]: group for group in get_params_options(args, model)["params"]
    }
    assert groups["embedding"]["weight_decay"] == 0.0
    assert groups["interactions_decay"]["weight_decay"] == 0.1
    assert groups["interactions_no_decay"]["weight_decay"] == 0.0
    assert groups["embedding"]["lr"] == 0.01
    assert groups["interactions_decay"]["lr"] == 0.005


def test_plotting_failure_restores_selection():
    model = load_anchor("tiny_scaleshift")
    freeze_named_modules(model, "readouts")
    before = {name: param.requires_grad for name, param in model.named_parameters()}

    class FailedLoader:
        def __iter__(self):
            raise RuntimeError("failed batch")

    with pytest.raises(RuntimeError, match="failed batch"):
        model_inference({"train": FailedLoader()}, model, {}, "cpu")
    assert {
        name: param.requires_grad for name, param in model.named_parameters()
    } == before


def test_freeze_yaml_does_not_require_a_finetuning_method(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("name: selection\nfreeze: readouts\n")
    args = build_default_arg_parser().parse_args(["--config", str(path)])
    assert args.freeze == "readouts"
    assert not args.l2sp and not args.lora and args.foundation_model is None


@pytest.mark.parametrize("level", [-7, -1, 0, 1, 2, 4, 5, 6, 7, 8])
def test_legacy_freeze_still_parses_as_an_integer_in_cli_and_yaml(tmp_path, level):
    parser = build_default_arg_parser()
    cli_args = parser.parse_args(["--name=legacy", "--freeze", str(level)])
    path = tmp_path / "legacy.yaml"
    path.write_text(f"name: legacy\nfreeze: {level}\n", encoding="utf-8")
    yaml_args = parser.parse_args(["--config", str(path)])
    for args in (cli_args, yaml_args):
        assert isinstance(args.freeze, int)
        assert args.freeze == level


@pytest.mark.parametrize(
    "level,frozen_modules",
    [
        (None, ()),
        (-7, ()),
        (-1, ()),
        (0, ()),
        (1, ("node_embedding",)),
        (2, ("node_embedding",)),
        (4, ("node_embedding",)),
        (5, ("node_embedding", "interactions")),
        (6, ("node_embedding", "interactions", "products")),
        (7, ("node_embedding", "interactions", "products", "readouts")),
        (8, ("node_embedding", "interactions", "products", "readouts")),
    ],
)
def test_legacy_freeze_keeps_masks_learning_rates_and_optimizer_layout(
    level, frozen_modules
):
    model = load_anchor("tiny_scaleshift")
    baseline = get_params_options(args_for(freeze=None), model)
    expected_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    for name in expected_flags:
        if name.startswith(tuple(f"{module}." for module in frozen_modules)):
            expected_flags[name] = False
    actual = get_params_options(args_for(freeze=level), model)
    assert {
        name: p.requires_grad for name, p in model.named_parameters()
    } == expected_flags
    assert len(actual["params"]) == len(baseline["params"])
    for old, new in zip(baseline["params"], actual["params"]):
        assert [id(p) for p in new["params"]] == [id(p) for p in old["params"]]
        group = new["name"]
        module = "node_embedding" if group == "embedding" else group.split("_")[0]
        expected_lr = 0.0 if module in frozen_modules else old.get("lr", baseline["lr"])
        assert new.get("lr", actual["lr"]) == expected_lr
        assert {k: v for k, v in new.items() if k not in ("params", "lr")} == {
            k: v for k, v in old.items() if k not in ("params", "lr")
        }


@pytest.mark.parametrize("level", [1, 6])
@pytest.mark.parametrize("named", [False, True])
def test_legacy_optimizer_checkpoint_and_updates_remain_compatible(level, named):
    with torch_tools.default_dtype("float64"):
        model = load_anchor("tiny_scaleshift")
        options = get_params_options(args_for(freeze=None), model)
        # Construct the historical optimizer: keep frozen tensors in their
        # original groups and zero the corresponding group learning rates.
        frozen = (
            ("node_embedding",)
            if level == 1
            else ("node_embedding", "interactions", "products")
        )
        for name in frozen:
            getattr(model, name).requires_grad_(False)
        for group in options["params"]:
            module = (
                "node_embedding"
                if group["name"] == "embedding"
                else group["name"].split("_", maxsplit=1)[0]
            )
            if module in frozen:
                group["lr"] = 0.0
        optimizer = torch.optim.Adam(**options)
        batch = anchor_batch(model, load_training_structures(limit=2))
        loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=10.0).double()
        outputs = {"forces": True, "stress": False, "virials": False}
        take_step(
            model,
            loss,
            batch.clone(),
            optimizer,
            None,
            outputs,
            1.0,
            torch.device("cpu"),
        )
        resumed = deepcopy(model)
        args = args_for(freeze=",".join(frozen) if named else level)
        resumed_optimizer = get_optimizer(args, get_params_options(args, resumed))
        resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        for candidate, opt in ((model, optimizer), (resumed, resumed_optimizer)):
            take_step(
                candidate,
                loss,
                batch.clone(),
                opt,
                None,
                outputs,
                1.0,
                torch.device("cpu"),
            )
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(
                tensor, resumed.state_dict()[name], atol=0, rtol=0
            )


def test_named_freezing_preserves_existing_flags_and_accepts_overlaps():
    model = load_anchor("tiny_scaleshift")
    model.node_embedding.requires_grad_(False)
    freeze_named_modules(model, "interactions,interactions.0")
    once = {name: p.requires_grad for name, p in model.named_parameters()}
    freeze_named_modules(model, "interactions.0")
    assert once == {name: p.requires_grad for name, p in model.named_parameters()}
    assert not any(p.requires_grad for p in model.node_embedding.parameters())
    assert not any(p.requires_grad for p in model.interactions.parameters())
    assert all(p.requires_grad for p in model.products.parameters())


@pytest.mark.parametrize(
    "selection",
    [
        "",
        "products,",
        "interactions.99",
        "atomic_energies_fn",
        "node_embedding,interactions,products,readouts",
    ],
)
def test_invalid_named_freeze_fails_before_changing_parameters(selection):
    model = load_anchor("tiny_scaleshift")
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="--freeze"):
        get_params_options(args_for(freeze=selection), model)
    assert before == {name: p.requires_grad for name, p in model.named_parameters()}
