"""The named-module freeze option works without enabling L2-SP."""

import pytest
import torch

from tests.golden.anchors import anchor_path
from tests.helpers import run_mace_train
from tests.workflows.test_cli_contracts import base_training_params


@pytest.mark.timeout(180)
@pytest.mark.parametrize("method", ["adam", "lbfgs", "lora"])
def test_trainable_layers_survive_training_and_plotting(
    tmp_path, regression_set, method
):
    overrides = {}
    if method == "lbfgs":
        overrides["lbfgs"] = None
    elif method == "lora":
        overrides.update(
            foundation_model=str(anchor_path("tiny_scaleshift")),
            foundation_model_elements=True,
            multiheads_finetuning=False,
            E0s="foundation",
            lora=True,
        )
    else:
        overrides["start_swa"] = 1
    args = base_training_params(
        tmp_path,
        regression_set,
        max_num_epochs=2,
        freeze="products,interactions.1,readouts",
        save_all_checkpoints=None,
        plot_frequency=1,
        **overrides,
    )
    result = run_mace_train(args, capture_output=True, text=True)
    assert "Plotting failed" not in result.stdout + result.stderr
    checkpoints = []
    for epoch in (0, 1):
        paths = list(tmp_path.glob(f"contract_run-7_epoch-{epoch}*.pt"))
        assert len(paths) == 1
        checkpoints.append(torch.load(paths[0], map_location="cpu", weights_only=False))
    before, after = (checkpoint["model"] for checkpoint in checkpoints)
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed
    assert all(
        name.startswith(("node_embedding.", "interactions.0.")) for name in changed
    )
    assert all("l2sp" not in checkpoint["optimizer"] for checkpoint in checkpoints)
    if method == "lora":
        assert all("lora_" in name for name in changed)
    else:
        export_names = ["contract.model"]
        if method == "adam":
            export_names.append("contract_stagetwo.model")
        for name in export_names:
            model = torch.load(tmp_path / name, map_location="cpu", weights_only=False)
            for parameter_name, param in model.named_parameters():
                assert param.requires_grad == parameter_name.startswith(
                    ("node_embedding.", "interactions.0.")
                )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_legacy_level_six_matches_named_freezing_and_keeps_optimizer_layout(
    tmp_path, regression_set, method
):
    models = {}
    checkpoints = {}
    for name, freeze in (
        ("legacy", 6),
        ("named", "node_embedding,interactions,products"),
    ):
        work = tmp_path / name
        work.mkdir()
        options = {"lbfgs": None} if method == "lbfgs" else {}
        args = base_training_params(
            work,
            regression_set,
            freeze=freeze,
            max_num_epochs=2,
            save_all_checkpoints=None,
            plot_frequency=1,
            **options,
        )
        run_mace_train(args, capture_output=True, text=True)
        models[name] = torch.load(
            work / "contract.model", map_location="cpu", weights_only=False
        )
        checkpoints[name] = torch.load(
            work / "contract_run-7_epoch-1.pt", map_location="cpu", weights_only=False
        )
    for name, value in models["legacy"].state_dict().items():
        torch.testing.assert_close(
            value, models["named"].state_dict()[name], atol=1e-11, rtol=1e-11
        )
    groups = checkpoints["legacy"]["optimizer"]["param_groups"]
    assert sum(len(group["params"]) for group in groups) == len(
        list(models["legacy"].parameters())
    )
    groups = checkpoints["named"]["optimizer"]["param_groups"]
    assert sum(len(group["params"]) for group in groups) == len(
        list(models["named"].parameters())
    )
