"""Preparation preserves data and fits references only on training structures."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import KeySpecification, open_dataset
from mace_torch.cli.preprocess_data import main
from mace_torch.train import DataStageError, run_data_stage
from test_mace_torch_training_pipeline import CATALOGUE, configuration

from tests.golden.harness import CLOSED_FORM_FP64

pytestmark = fp64_only


def molecules(path, count=12, offset=0):
    frames = []
    for index in range(count):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + index / 100, 0, 0]])
        atoms.info.update(REF_energy=float((index + offset) ** 2), config_weight=2.0)
        atoms.arrays["REF_forces"] = np.full((2, 3), (index + 1) / 10)
        frames.append(atoms)
    write(path, frames)
    return path


def dataset(path):
    return open_dataset(
        path, key_spec=KeySpecification.from_defaults(), keep_isolated_atoms=True
    )


def test_average_references_use_only_the_training_split(tmp_path):
    source = molecules(tmp_path / "train.xyz")
    output = tmp_path / "prepared"
    assert (
        main(
            [
                "--train_file",
                str(source),
                "--output",
                str(output),
                "--E0s",
                "average",
                "--valid_fraction",
                "0.25",
                "--seed",
                "3",
                "--shuffle",
                "false",
                "--num_process",
                "1",
                "--compute_statistics",
            ]
        )
        == 0
    )
    train = list(dataset(output / "train").iter_range())
    valid = list(dataset(output / "valid").iter_range())
    expected = np.mean([item.properties["energy"] / 2 for item in train])
    all_energy = np.mean([item.properties["energy"] / 2 for item in train + valid])
    assert expected != all_energy
    stats = dataset(output / "train").statistics()
    np.testing.assert_allclose(
        [
            json.loads((output / "e0s.json").read_text())["1"],
            stats.atomic_energies[1],
            stats.mean,
        ],
        [expected, expected, 0.0],
        atol=CLOSED_FORM_FP64.atol,
        rtol=CLOSED_FORM_FP64.rtol,
    )


def test_typed_config_lists_and_cli_overrides_preserve_weights(tmp_path):
    first = molecules(tmp_path / "first.xyz", count=4)
    second = molecules(tmp_path / "second.xyz", count=5, offset=10)
    valid = molecules(tmp_path / "valid.xyz", count=2, offset=20)
    document = dict(
        train_file=[str(first), str(second)],
        valid_file=[str(valid)],
        test_file=[str(valid)],
        output="ignored",
        work_dir=str(tmp_path),
        atomic_numbers=[1],
        E0s={1: 2.0},
        config_type_weights={"Default": 0.25},
        compute_statistics=False,
        shuffle=False,
        num_process=1,
    )
    config = tmp_path / "prepare.yaml"
    config.write_text(yaml.safe_dump(document))
    assert (
        main(
            [
                "--config",
                str(config),
                "--output",
                "prepared",
                "--r_max",
                "4.0",
                "--compute_statistics",
            ]
        )
        == 0
    )
    output = tmp_path / "prepared"
    train = dataset(output / "train")
    assert [item.properties["energy"] for item in train.iter_range()] == [
        0,
        1,
        4,
        9,
        100,
        121,
        144,
        169,
        196,
    ]
    assert {item.weight for item in train.iter_range()} == {0.5}
    assert len(dataset(output / "valid")) == len(dataset(output / "test")) == 2
    assert train.statistics().r_max == 4.0
    assert train.statistics().atomic_energies == {1: 2.0}
    assert not (tmp_path / "ignored").exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("r_max", "0"),
        ("r_max", "nan"),
        ("shard_size", "0"),
        ("num_process", "0"),
        ("scaling", "typo"),
    ],
)
def test_invalid_preparation_settings_fail_before_writing(tmp_path, flag, value):
    source = molecules(tmp_path / "train.xyz")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match=flag):
        main(
            [
                "--train_file",
                str(source),
                "--output",
                str(output),
                "--E0s",
                "{1: 0.0}",
                "--num_process",
                "1",
                "--" + flag,
                value,
            ]
        )
    assert not output.exists()


def test_an_existing_output_is_not_partially_replaced(tmp_path):
    source = molecules(tmp_path / "train.xyz")
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing dataset")
    with pytest.raises(FileExistsError, match="empty"):
        main(
            [
                "--train_file",
                str(source),
                "--output",
                str(output),
                "--E0s",
                "{1: 0.0}",
                "--num_process",
                "1",
            ]
        )
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "existing dataset"


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_statistics_are_consumed_without_recomputation(
    tmp_path, monkeypatch, legacy
):
    from mace_torch.train import data_stage

    config = configuration(tmp_path)
    baseline = run_data_stage(config, CATALOGUE)
    saved = asdict(baseline.statistics)
    saved.update(mean=2.5, normalization_scale=3.0, avg_num_neighbors=7.0)
    if legacy:
        saved["std"] = saved.pop("normalization_scale")
        saved.pop("normalization_method")
        for name in ("atomic_numbers", "atomic_energies"):
            saved[name] = str(saved[name])
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(saved))
    settings = config.to_resolved_dict()
    settings["data"]["statistics_file"] = str(path)
    monkeypatch.setattr(
        data_stage,
        "compute_statistics",
        lambda *a, **kw: pytest.fail("explicit statistics must be used"),
    )
    actual = run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)
    assert (
        actual.statistics.mean,
        actual.statistics.normalization_scale,
        actual.statistics.avg_num_neighbors,
    ) == (2.5, 3.0, 7.0)
    assert actual.e0s == baseline.e0s


@pytest.mark.parametrize(
    "field,value",
    [
        ("r_max", 9.0),
        ("atomic_numbers", [1]),
        ("atomic_energies", {1: 0.0, 8: 0.0}),
        ("normalization_scale", -1.0),
        ("mean", float("nan")),
        ("avg_num_neighbors", -1.0),
    ],
)
def test_statistics_must_match_the_resolved_model_inputs(tmp_path, field, value):
    config = configuration(tmp_path)
    saved = asdict(run_data_stage(config, CATALOGUE).statistics)
    saved[field] = value
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(saved))
    settings = config.to_resolved_dict()
    settings["data"]["statistics_file"] = str(path)
    with pytest.raises(DataStageError, match=field):
        run_data_stage(ResolvedConfig.model_validate(settings), CATALOGUE)


@pytest.mark.parametrize("scaling", ["no_scaling", "std_scaling", "rms_forces_scaling"])
def test_scaling_and_reference_json_are_used_in_preparation(tmp_path, scaling):
    source = molecules(tmp_path / "train.xyz", count=5)
    output = tmp_path / "prepared"
    references = tmp_path / "references.json"
    references.write_text('{"1": 2.0}')
    main(
        [
            "--train_file",
            str(source),
            "--valid_file",
            str(source),
            "--output",
            str(output),
            "--E0s",
            str(references),
            "--num_process",
            "1",
            "--shuffle",
            "false",
            "--compute_statistics",
            "--scaling",
            scaling,
            "--shard_size",
            "3",
            "--batch_size",
            "1",
        ]
    )
    stats = dataset(output / "train").statistics()
    assert (
        stats.normalization_method
        == {
            "no_scaling": "none",
            "std_scaling": "std",
            "rms_forces_scaling": "rms_forces",
        }[scaling]
    )
    expected = {
        "no_scaling": 1.0,
        "std_scaling": np.std(np.arange(5) ** 2 / 2),
        "rms_forces_scaling": np.sqrt(np.mean((np.arange(1, 6) / 10) ** 2)),
    }
    np.testing.assert_allclose(
        [stats.mean, stats.normalization_scale],
        [np.mean(np.arange(5) ** 2 / 2 - 2), expected[scaling]],
        atol=CLOSED_FORM_FP64.atol,
        rtol=CLOSED_FORM_FP64.rtol,
    )
    assert len(list((output / "train").glob("*.h5"))) == 2
    assert stats.atomic_energies == {1: 2.0}


@pytest.mark.parametrize(
    "flag,value", [("heads", "{'a': {}}"), ("head_key", "custom_head")]
)
def test_preparation_retains_the_one_dataset_per_head_contract(tmp_path, flag, value):
    with pytest.raises(ValueError, match="one dataset per head"):
        main(
            [
                "--train_file",
                str(tmp_path / "train.xyz"),
                "--output",
                str(tmp_path / "prepared"),
                "--" + flag,
                value,
            ]
        )
    assert not (tmp_path / "prepared").exists()


def test_separately_prepared_heads_reproduce_raw_data_training(tmp_path):
    import torch
    from mace_torch.train import (
        latest_run_checkpoint,
        run_model_stage,
        run_train_stage,
    )
    from safetensors.torch import load_file

    settings = configuration(tmp_path, max_num_epochs=2).to_resolved_dict()
    raw_heads, prepared_heads = {}, {}
    for head, count, e0 in (("first", 4, 2.0), ("second", 6, -1.0)):
        source = molecules(tmp_path / f"{head}.xyz", count=count)
        valid = molecules(tmp_path / f"{head}_valid.xyz", count=2, offset=2)
        destination = tmp_path / head
        main(
            [
                "--train_file",
                str(source),
                "--valid_file",
                str(valid),
                "--output",
                str(destination),
                "--E0s",
                repr({1: e0}),
                "--r_max",
                "5.0",
                "--num_process",
                "1",
                "--shard_size",
                "3",
                "--shuffle",
                "false",
                "--compute_statistics",
            ]
        )
        raw_heads[head] = {
            "train_file": str(source),
            "valid_file": str(valid),
            "e0s": {"kind": "table", "values": {1: e0}},
        }
        prepared_heads[head] = {
            **raw_heads[head],
            "train_file": str(destination / "train"),
            "valid_file": str(destination / "valid"),
        }
        assert dataset(destination / "train").statistics().atomic_energies == {1: e0}

    results, tensors = [], []
    for name, heads in (("raw", raw_heads), ("prepared", prepared_heads)):
        config = ResolvedConfig.from_dict(
            {**settings, "data": {**settings["data"], "heads": heads}}
        )
        data = run_data_stage(config, CATALOGUE)
        assert data.e0s.values == {"first": {1: 2.0}, "second": {1: -1.0}}
        built = run_model_stage(config, data, CATALOGUE)
        results.append(run_train_stage(config, built, checkpoint_path=tmp_path / name))
        checkpoint = latest_run_checkpoint(tmp_path, name)
        assert checkpoint is not None
        tensors.append(load_file(str(checkpoint.with_suffix(".safetensors"))))
    assert [record.train_loss for record in results[0].history] == [
        record.train_loss for record in results[1].history
    ]
    assert [record.valid_loss for record in results[0].history] == [
        record.valid_loss for record in results[1].history
    ]
    assert tensors[0].keys() == tensors[1].keys()
    assert any(key.startswith("optimizer") for key in tensors[0])
    for key, value in tensors[0].items():
        assert torch.equal(value, tensors[1][key]), key


def test_ase_trajectory_input_remains_supported(tmp_path):
    from ase.io import read

    source = molecules(tmp_path / "source.xyz", count=3)
    trajectory = tmp_path / "source.traj"
    # ASE .traj preserves info and standard calculator data, not arbitrary arrays.
    write(trajectory, read(source, index=":"))
    output = tmp_path / "prepared"
    main(
        [
            "--train_file",
            str(trajectory),
            "--valid_file",
            str(trajectory),
            "--output",
            str(output),
            "--E0s",
            "{1: 0.0}",
            "--num_process",
            "1",
            "--shuffle",
            "false",
        ]
    )
    actual = list(dataset(output / "train").iter_range())
    assert [item.properties["energy"] for item in actual] == [0.0, 1.0, 4.0]
    assert all(item.properties["forces"] is None for item in actual)
    assert all(item.property_weights["forces"] == 0.0 for item in actual)
