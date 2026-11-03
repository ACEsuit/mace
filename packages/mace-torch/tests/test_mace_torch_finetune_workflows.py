"""Selection and replay honor their inputs across the callable stage boundaries."""

import sys
from dataclasses import replace

import numpy as np
import pytest
from ase.io import read, write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import KeySpecification
from mace_core.data.xyz import read_configurations
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data import GraphDataset
from mace_torch.finetune import SelectionError
from mace_torch.finetune.foundation import read_foundation
from mace_torch.finetune.pseudolabels import (
    PseudolabelError,
    load_pseudolabel_artifact,
    relabelling,
)
from mace_torch.finetune.stages import build, pseudolabel, select_structures
from mace_torch.finetune.subselect import farthest_point_indices, select
from test_mace_torch_extend_elements import water_foundation
from test_mace_torch_pseudolabels import fine_tune, replay_structures
from test_mace_torch_subselect import SIX

pytestmark = fp64_only


@pytest.fixture(scope="module")
def foundation(tmp_path_factory):
    return water_foundation(tmp_path_factory.mktemp("selection-foundation"))


def unlabelled(path):
    frames = replay_structures()
    for atoms in frames:
        atoms.info.clear()
        atoms.arrays.pop("REF_forces")
    write(path, frames)


def test_selection_accepts_structures_without_labels(tmp_path):
    path = tmp_path / "pool.xyz"
    unlabelled(path)
    config = ResolvedConfig.model_validate(
        {
            "data": {
                "heads": {
                    "source": {
                        "train_file": str(path),
                        "subselect": {"num_samples": 3, "method": "random"},
                    }
                }
            }
        }
    )
    output = select_structures(config, "source", tmp_path / "selected.xyz")
    assert len(read(output, ":")) == 3


def test_unlabelled_replay_is_labelled_before_training(foundation, tmp_path):
    config = fine_tune(tmp_path, foundation)
    unlabelled(tmp_path / "replay.xyz")
    built = build(config)
    assert len(built.data.train_loader.datasets["replay"]) > 0
    labels, _ = load_pseudolabel_artifact(tmp_path / "pseudolabels/replay")
    assert len(labels) == 6
    assert all(item.properties["energy"] is not None for item in labels)
    assert all(item.property_weights["forces"] == 1.0 for item in labels)


def test_explicit_validation_replay_uses_the_same_label_source(foundation, tmp_path):
    config = fine_tune(tmp_path, foundation)
    valid = tmp_path / "valid.xyz"
    write(valid, replay_structures(count=3, seed=19))
    document = config.model_dump(mode="json", exclude_unset=True)
    document["data"]["heads"]["replay"]["valid_file"] = str(valid)
    config = ResolvedConfig.model_validate(document)
    built = build(config)
    dataset = built.data.valid_loaders["replay"].dataset
    assert isinstance(dataset, GraphDataset)
    validation = dataset.configurations
    assert len(validation) == 3
    assert all(item.properties["energy"] != 99.0 for item in validation)
    labelled, record = load_pseudolabel_artifact(tmp_path / "pseudolabels/replay")
    assert record.n_configs == len(labelled) == 9
    alone = pseudolabel(config, "replay", tmp_path / "standalone")
    selected, _ = load_pseudolabel_artifact(alone / "replay")
    for first, second in zip(labelled, selected, strict=True):
        np.testing.assert_array_equal(
            first.properties["forces"], second.properties["forces"]
        )


def test_exact_fps_does_not_repeat_indices_when_descriptors_tie(monkeypatch):
    monkeypatch.setitem(sys.modules, "fpsample", None)
    picked = farthest_point_indices(np.zeros((5, 3)), 4, np.random.RandomState(2))
    assert len(set(picked)) == len(picked) == 4


@pytest.mark.parametrize("count", [None, len(SIX), len(SIX) + 1])
def test_unknown_selection_method_is_rejected_on_every_path(count):
    with pytest.raises(SelectionError, match="not a method"):
        select(SIX, num_samples=count, method="typo", filtering="none")  # ty: ignore[invalid-argument-type]


def test_empty_input_does_not_hide_an_unknown_filter():
    with pytest.raises(SelectionError, match="not a filter"):
        select([], filtering="typo", elements=[1])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "change", ["foundation_head", "properties", "stress_if_missing"]
)
def test_reused_labels_must_match_the_requested_generation(
    foundation, tmp_path, change
):
    config = fine_tune(tmp_path, foundation)
    pseudolabel(config, "replay", tmp_path / "labels")
    document = config.model_dump(mode="json", exclude_unset=True)
    document["finetune"]["pseudolabels"].update(
        enabled=False, labels_from=str(tmp_path / "labels")
    )
    loaded = read_foundation(foundation, DEFAULT_CATALOGUE)
    if change == "foundation_head":
        document["data"]["heads"]["replay"]["readout_from"] = "different"
        loaded = replace(loaded, heads=("pbe", "different"))
    elif change == "properties":
        document["finetune"]["pseudolabels"]["properties"] = [
            "energy",
            "forces",
            "stress",
        ]
    else:
        document["finetune"]["pseudolabels"]["stress_if_missing"] = True
    config = ResolvedConfig.model_validate(document)
    items = read_configurations(
        tmp_path / "replay.xyz", KeySpecification.from_defaults(), head_name="replay"
    ).configurations
    relabel = relabelling(config, loaded)
    assert relabel is not None
    with pytest.raises(PseudolabelError, match=change):
        relabel("replay", items)


def test_fps_needs_no_model_when_no_subset_is_drawn(tmp_path):
    source = tmp_path / "pool.xyz"
    unlabelled(source)
    config = ResolvedConfig.model_validate(
        {
            "data": {
                "heads": {
                    "source": {
                        "train_file": str(source),
                        "subselect": {"method": "fps", "num_samples": 6},
                    }
                }
            }
        }
    )
    output = select_structures(config, "source", tmp_path / "selected.xyz")
    assert len(read(output, ":")) == 6


@pytest.mark.parametrize("model_dtype", ["float32", "float64"])
def test_fps_descriptors_follow_the_foundations_dtype(foundation, model_dtype):
    import torch
    from mace_core.config.precision import PrecisionConfig
    from mace_core.data.xyz import configuration_from_atoms

    loaded = read_foundation(
        foundation,
        DEFAULT_CATALOGUE,
        precision=PrecisionConfig(model=model_dtype, accumulate=model_dtype),
    )
    items = [
        configuration_from_atoms(atoms, KeySpecification.from_defaults())
        for atoms in replay_structures(count=3)
    ]
    descriptors = loaded.describe(items)
    assert descriptors.shape[0] == 3
    assert np.isfinite(descriptors).all()
    assert next(loaded.engine.parameters()).dtype == getattr(torch, model_dtype)


def test_labelled_training_still_rejects_an_unlabelled_source(foundation, tmp_path):
    from mace_torch.train import DataStageError

    config = fine_tune(tmp_path, foundation, pseudolabels={})
    unlabelled(tmp_path / "replay.xyz")
    with pytest.raises(DataStageError, match="none of"):
        build(config)


def test_fps_padding_needs_no_descriptors_when_no_structure_passes(tmp_path):
    source = tmp_path / "pool.xyz"
    unlabelled(source)
    config = ResolvedConfig.model_validate(
        {
            "data": {
                "heads": {
                    "source": {
                        "train_file": str(source),
                        "subselect": {
                            "method": "fps",
                            "num_samples": 3,
                            "filtering": "combinations",
                            "elements": [6],
                            "allow_random_padding": True,
                        },
                    }
                }
            }
        }
    )
    selected = select_structures(config, "source", tmp_path / "selected.xyz")
    assert len(read(selected, ":")) == 3


@pytest.mark.parametrize(
    "points,count",
    [(np.zeros((3, 2)), 0), (np.zeros(3), 2), (np.full((3, 2), np.nan), 2)],
)
def test_fps_refuses_invalid_descriptors_and_empty_requests(points, count):
    with pytest.raises(SelectionError):
        farthest_point_indices(points, count, np.random.RandomState(1))


def test_selection_refuses_an_output_its_reader_cannot_identify(tmp_path):
    from mace_torch.train import DataStageError

    source = tmp_path / "pool.xyz"
    unlabelled(source)
    config = ResolvedConfig.model_validate(
        {
            "data": {
                "heads": {
                    "source": {
                        "train_file": str(source),
                        "subselect": {"method": "random", "num_samples": 3},
                    }
                }
            }
        }
    )
    with pytest.raises(DataStageError, match=r"\.xyz or \.extxyz"):
        select_structures(config, "source", tmp_path / "selected.json")
    assert not (tmp_path / "selected.json").exists()
