"""The normal CUEQ conversion path with L2-SP, resume, and e3nn export."""

import pytest
import torch

from mace.cli.convert_e3nn_cueq import run as to_cueq
from mace.tools import torch_tools
from tests.golden.anchors import anchor_batch, load_anchor, load_training_structures
from tests.helpers import run_mace_train
from tests.workflows.l2sp.test_training import latest_checkpoint, training_args

pytestmark = pytest.mark.cueq


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "device,conv_fusion",
    [
        ("cpu", False),
        pytest.param("cuda", False, marks=pytest.mark.gpu),
        pytest.param("cuda", True, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "optimizer,freeze,averaging",
    [("adamw", "products,interactions.1,readouts", True), ("adam", 0, False)],
)
def test_cli_training_resume_and_export(
    tmp_path, device, conv_fusion, optimizer, freeze, averaging
):
    options = {"ema": None, "start_swa": 1} if averaging else {}
    args = training_args(
        tmp_path,
        device=device,
        enable_cueq=True,
        cueq_conv_fusion=conv_fusion,
        optimizer=optimizer,
        freeze=freeze,
        **options,
    )
    run_mace_train(args, capture_output=True, text=True)
    initial = latest_checkpoint(tmp_path)
    with torch_tools.default_dtype("float64"):
        source = to_cueq(
            load_anchor("tiny_scaleshift"), device=device, conv_fusion=conv_fusion
        )
    source_params = dict(source.named_parameters())
    selected = {
        name
        for group in initial["optimizer"]["param_groups"]
        for name in group["param_names"]
    }
    assert "products.0.symmetric_contractions.weight" in initial["model"]
    if freeze:
        assert selected == {
            name
            for name in source_params
            if name.startswith(("node_embedding.", "interactions.0."))
        }
    else:
        assert selected == set(source_params)
    changed = []
    for name, param in source_params.items():
        if name not in selected:
            torch.testing.assert_close(
                initial["model"][name], param.cpu(), atol=0, rtol=0
            )
        elif not torch.equal(initial["model"][name], param.cpu()):
            changed.append(name)
    assert any(name.startswith("node_embedding.") for name in changed)
    assert any(name.startswith("interactions.0.") for name in changed)
    if not freeze:
        assert any(name.startswith("products.") for name in changed)
    for group, refs in zip(
        initial["optimizer"]["param_groups"], initial["optimizer"]["l2sp"]["references"]
    ):
        for name, reference in zip(group["param_names"], refs):
            torch.testing.assert_close(
                reference, source_params[name].cpu(), atol=0, rtol=0
            )

    export_name = "l2sp_stagetwo.model" if averaging else "l2sp.model"
    exported = torch.load(
        tmp_path / export_name, weights_only=False, map_location="cpu"
    )
    assert hasattr(exported.products[0].symmetric_contractions, "contractions")
    exported.requires_grad_(False)
    output = exported(
        anchor_batch(exported, load_training_structures(limit=1)).to_dict(),
        compute_force=True,
        compute_stress=True,
    )
    assert all(
        torch.isfinite(output[key]).all() for key in ("energy", "forces", "stress")
    )

    run_mace_train(
        {**args, "max_num_epochs": 3, "restart_latest": None},
        capture_output=True,
        text=True,
    )
    resumed = latest_checkpoint(tmp_path)
    for old_refs, new_refs in zip(
        initial["optimizer"]["l2sp"]["references"],
        resumed["optimizer"]["l2sp"]["references"],
    ):
        for old, new in zip(old_refs, new_refs):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
    assert any(
        not torch.equal(initial["model"][name], resumed["model"][name])
        for name in selected
    )

    bad = run_mace_train(
        {**args, "enable_cueq": False, "max_num_epochs": 4, "restart_latest": None},
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert "L2SPCheckpointError" in bad.stderr
