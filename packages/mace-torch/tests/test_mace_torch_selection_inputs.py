"""Selection inputs and execution settings survive each fine-tuning entry point."""

import numpy as np
import pytest
import torch
from ase.io import read, write
from conftest import TOLERANCES, fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.cli.migrate import selection_document
from mace_torch.finetune.foundation import Foundation, read_foundation
from mace_torch.finetune.pseudolabels import (
    generate_pseudolabels,
    load_pseudolabel_artifact,
)
from mace_torch.finetune.stages import build, pseudolabel, select_structures
from mace_torch.train import DataStageError
from mace_torch.train.ddp import DistributedContext
from test_mace_torch_extend_elements import water_foundation
from test_mace_torch_pseudolabels import fine_tune, replay_structures

pytestmark = fp64_only


@pytest.fixture(scope="module")
def foundation(tmp_path_factory):
    return water_foundation(tmp_path_factory.mktemp("selection-input-foundation"))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("stage", ["select", "pseudolabel", "build", "build_context"])
def test_stages_apply_the_requested_execution_settings(
    foundation, tmp_path, monkeypatch, device, precision, stage
):
    config = fine_tune(tmp_path, foundation)
    document = config.model_dump(mode="json", exclude_unset=True)
    document["runtime"]["device"] = "cuda:73" if stage == "build_context" else device
    document["precision"] = {"model": precision}
    document["data"]["heads"]["replay"]["subselect"] = {
        "method": "fps",
        "num_samples": 3,
    }
    config = ResolvedConfig.model_validate(document)
    observed = []
    structure_descriptors = Foundation.structure_descriptors

    def observe(self, items):
        parameter = next(self.engine.parameters())
        observed.append((parameter.device.type, str(parameter.dtype)))
        return structure_descriptors(self, items)

    monkeypatch.setattr(Foundation, "structure_descriptors", observe)
    if stage == "select":
        selected = select_structures(config, "replay", tmp_path / "kept.xyz")
        assert len(read(selected, ":")) == 3
    elif stage == "pseudolabel":
        pseudolabel(config, "replay", tmp_path / "pseudolabels")
    elif stage == "build_context":
        build(config, context=DistributedContext(device=device))
    else:
        build(config)
    assert observed == [(device, f"torch.{precision}")]
    if stage != "select":
        labels, record = load_pseudolabel_artifact(tmp_path / "pseudolabels/replay")
        assert record.device.split(":")[0] == device
        assert record.dtype == precision
        assert record.deterministic == (device == "cpu")
        assert len(labels) == 3
        assert all(np.isfinite(item.properties["forces"]).all() for item in labels)
        if device == "cuda":
            reference = read_foundation(
                foundation, DEFAULT_CATALOGUE, precision=config.precision
            )
            expected = generate_pseudolabels(
                reference.engine,
                labels,
                config.finetune.pseudolabels,
                z_table=reference.z_table,
                cutoff=reference.config.model.r_max,
                head_index=0,
                batch_size=config.training.batch_size,
            )
            atol, rtol = TOLERANCES[getattr(torch, precision)]
            for actual, wanted in zip(labels, expected, strict=True):
                for name in ("energy", "forces"):
                    np.testing.assert_allclose(
                        actual.properties[name],
                        wanted.properties[name],
                        atol=atol,
                        rtol=rtol,
                    )


def cached_config(tmp_path, values):
    frames = replay_structures()
    for index, atoms in enumerate(frames):
        atoms.info["sample_id"] = index
    source, descriptors = tmp_path / "pool.xyz", tmp_path / "descriptors.npy"
    write(source, frames)
    np.save(descriptors, values)
    return ResolvedConfig.model_validate(
        {
            "runtime": {"seed": 11},
            "data": {
                "heads": {
                    "source": {
                        "train_file": str(source),
                        "subselect": {
                            "method": "fps",
                            "num_samples": 3,
                            "structure_descriptors_file": str(descriptors),
                        },
                    }
                }
            },
        }
    )


def test_cached_descriptors_select_without_loading_a_model(tmp_path):
    config = cached_config(tmp_path, np.arange(6, dtype=float).reshape(-1, 1))
    output = select_structures(config, "source", tmp_path / "selected.xyz")
    # First point is seeded at 1; 5 is farthest, then 3. No distance ties.
    expected = read(tmp_path / "pool.xyz", ":")
    assert isinstance(expected, list)
    actual = read(output, ":")
    assert len(actual) == 3
    for kept, index in zip(actual, (1, 5, 3), strict=True):
        np.testing.assert_array_equal(kept.positions, expected[index].positions)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((5, 2)),
        np.zeros(6),
        np.zeros((6, 0)),
        np.full((6, 2), np.nan),
        np.full((6, 2), "wrong"),
        np.array([{"x": 1}] * 6, dtype=object),
    ],
)
def test_cached_descriptors_must_be_a_numeric_row_per_source(tmp_path, values):
    config = cached_config(tmp_path, values)
    with pytest.raises(DataStageError, match="descriptors"):
        select_structures(config, "source", tmp_path / "selected.xyz")
    assert not (tmp_path / "selected.xyz").exists()


def test_selection_rejects_invalid_device_even_without_model(tmp_path):
    config = cached_config(tmp_path, np.zeros((6, 2)))
    document = config.model_dump(mode="json", exclude_unset=True)
    document["runtime"]["device"] = "cpu0"
    config = ResolvedConfig.model_validate(document)
    with pytest.raises((ValueError, RuntimeError), match="device"):
        select_structures(config, "source", tmp_path / "selected.xyz")


def test_legacy_descriptor_option_maps_to_typed_selection(tmp_path):
    document, output = selection_document(
        [
            "--configs_pt",
            "pool.xyz",
            "--output",
            "selected.xyz",
            "--structure-descriptors-file",
            "descriptors.npy",
            "--default_dtype",
            "float32",
        ]
    )
    config = ResolvedConfig.model_validate(document)
    subselect = config.data.heads["selected"].subselect
    assert (
        subselect is not None
        and str(subselect.structure_descriptors_file) == "descriptors.npy"
    )
    assert config.precision.model == "float32"
    assert output == "selected.xyz"


def test_cached_descriptors_keep_source_order_across_files_and_filtering(tmp_path):
    config = cached_config(tmp_path, np.array([[0], [1], [4], [5], [10], [11]]))
    frames = read(tmp_path / "pool.xyz", ":")
    assert isinstance(frames, list)
    for index in (1, 3, 5):
        frames[index].numbers = [1, 1, 1]
    write(tmp_path / "a.xyz", frames[:3])
    write(tmp_path / "b.xyz", frames[3:])
    document = config.model_dump(mode="json", exclude_unset=True)
    head = document["data"]["heads"]["source"]
    head["train_file"] = [str(tmp_path / name) for name in ("a.xyz", "b.xyz")]
    head["subselect"].update(filtering="inclusive", elements=[8], num_samples=2)
    output = select_structures(
        ResolvedConfig.model_validate(document), "source", tmp_path / "kept.xyz"
    )
    for kept, index in zip(read(output, ":"), (2, 4), strict=True):
        np.testing.assert_array_equal(kept.positions, frames[index].positions)


def test_cached_descriptors_are_not_silently_ignored_for_random_selection(tmp_path):
    config = cached_config(tmp_path, np.zeros((6, 2)))
    document = config.model_dump(mode="json", exclude_unset=True)
    document["data"]["heads"]["source"]["subselect"]["method"] = "random"
    with pytest.raises(ValueError, match="requires method='fps'"):
        ResolvedConfig.model_validate(document)
