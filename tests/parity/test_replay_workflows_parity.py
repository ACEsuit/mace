"""Element transfer and actual replay counts against the frozen stack."""

import json
import re

import pytest
import torch
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_torch.finetune.stages import build

from tests.golden.anchors import anchor_batch
from tests.golden.harness import FP64_CPU_REFERENCE
from tests.helpers import base_mace_params, run_mace_train
from tests.parity.anchors import load_anchor
from tests.parity.test_anchor_as_foundation import TRAIN_SET, write_anchor_checkpoint
from tests.parity.test_element_table_parity import legacy_subset, water_structures
from tests.workflows.test_replay_elements_v1 import (
    built_data,
    replay_document,
)
from tests.workflows.test_replay_elements_v1 import (
    replay_foundation as replay_foundation,
)

pytestmark = pytest.mark.usefixtures("isolated", "fp64")


@pytest.mark.parametrize("element_table", ["data", "foundation"])
def test_multihead_element_transfer_preserves_the_frozen_foundation(
    tmp_path, element_table
):
    heads = ("memory", "pbe", "mp2")
    legacy = load_anchor("tiny_scaleshift.model")
    parent = write_anchor_checkpoint(legacy, tmp_path)
    waters = water_structures()
    source = tmp_path / "waters.xyz"
    write(source, waters)
    config = ResolvedConfig.from_dict(
        {
            "runtime": {"work_dir": str(tmp_path)},
            "finetune": {
                "foundation_model": str(parent),
                "element_table": element_table,
            },
            "data": {
                "heads": {
                    name: {
                        "train_file": str(source),
                        "valid_file": str(source),
                        "readout_from": "default",
                        "e0s": {"kind": "foundation", "head": "default"},
                    }
                    for name in heads
                },
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces"]},
            "training": {"batch_size": 2, "valid_batch_size": len(waters)},
        }
    )
    built = build(config)
    expected_elements = (
        [1, 8] if element_table == "data" else [int(z) for z in legacy.atomic_numbers]
    )
    assert list(built.data.z_table.zs) == expected_elements
    if element_table == "data":
        legacy = legacy_subset(legacy, expected_elements)
    frozen_batch = anchor_batch(legacy, waters)
    expected = legacy(frozen_batch.to_dict(), compute_force=True)
    for loader in built.data.valid_loaders.values():
        actual = built.model(next(iter(loader)).graph, compute=("forces",))
        for value, reference in (
            (actual.total_energy, expected["energy"]),
            (actual.forces, expected["forces"]),
        ):
            torch.testing.assert_close(
                value,
                reference,
                atol=FP64_CPU_REFERENCE.atol,
                rtol=FP64_CPU_REFERENCE.rtol,
            )


@pytest.mark.parametrize(
    "threshold,copies", [(0.1, 1), (0.5, 1), (0.500001, 2), (1.0, 3)]
)
def test_replay_counts_match_the_frozen_training_pipeline(
    replay_foundation, tmp_path, threshold, copies
):
    """Run legacy through loader construction; do not transcribe its arithmetic."""
    root = replay_foundation
    params = base_mace_params()
    params.update(
        {
            "name": "counts",
            "foundation_model": str(
                TRAIN_SET.parents[1] / "models/tiny_scaleshift.model"
            ),
            "multiheads_finetuning": True,
            "force_mh_ft_lr": True,
            "heads": json.dumps(
                {
                    name: {
                        "train_file": str(root / f"{name}.xyz"),
                        "valid_file": str(root / "valid.xyz"),
                        "E0s": "foundation",
                    }
                    for name in ("pbe", "mp2")
                }
            ),
            "pt_train_file": str(root / "memory.xyz"),
            "pt_valid_file": str(root / "valid.xyz"),
            "num_samples_pt": 10,
            "subselect_pt": "random",
            "filter_type_pt": "none",
            "real_pt_data_ratio_threshold": threshold,
            "batch_size": 2,
            "valid_batch_size": 2,
            "max_num_epochs": 1,
            "dry_run": None,
            "seed": 23,
            **{
                name: str(tmp_path)
                for name in ("checkpoints_dir", "model_dir", "log_dir", "results_dir")
            },
        }
    )
    # The frozen path has no canonical checkpoint input. Its committed anchor
    # supplies the real legacy loader; only data counts are compared here.
    done = run_mace_train(
        params,
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env_extra={"MACE_ENGINE": "legacy"},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    counts = {
        name: int(count)
        for name, count in re.findall(
            r"Head '([^']+)' training dataset size: (\d+)", done.stdout + done.stderr
        )
    }
    assert counts == {"pt_head": 10, "pbe": 2 * copies, "mp2": 3 * copies}
    counts["memory"] = counts.pop("pt_head")
    document = replay_document(root)
    document["data"]["ratio_guard"] = {"reference": "memory", "threshold": threshold}
    built = built_data(document)
    assert {
        name: len(dataset) for name, dataset in built.data.train_loader.datasets.items()
    } == counts
