"""Custom file labels enter training through preparation, including per-head data.

Training keeps V1's fixed property keys. The two runs below see the same
labels through independently normalized XYZ and CLI-prepared shards.
"""

import copy
import json

import numpy as np
import pytest
import yaml
from ase.io import read, write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_data_sources_v1 import close, launch
from tests.workflows.test_preprocessing_v1 import prepare
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, tiny_task, training_records

pytestmark = needs_the_v1_engine


@pytest.fixture(scope="module")
def labelled_sources(tmp_path_factory):
    from mace_core.data import KeySpecification, open_dataset

    root = tmp_path_factory.mktemp("label-sources")
    document = yaml.safe_load(tiny_task(root).read_text())
    document["training"]["max_num_epochs"] = 2
    # A native foundation for the multihead case; no download or pickle.
    launch(root, document, "foundation")
    frames = read(root / "train.xyz", ":")[2:]
    heads = {}
    for number, name in enumerate(("pbe", "mp2")):
        raw, normalized = [], []
        for index, original in enumerate(frames):
            atoms = original.copy()
            atoms.cell = np.eye(3) * 10
            atoms.pbc = True
            atoms.info["REF_energy"] += 0.2 * number
            atoms.arrays["REF_forces"] *= number + 1
            atoms.info["REF_stress"] = np.eye(3) * (index + 1) * 0.001
            # Includes one wholly unlabelled structure per four records.
            for kind, storage in (("energy", atoms.info), ("forces", atoms.arrays),
                                  ("stress", atoms.info)):
                absent = index % 4 == ("energy", "forces", "stress").index(kind) or index % 4 == 3
                if absent:
                    storage.pop(f"REF_{kind}")
            normalized.append(atoms.copy())
            for kind, storage in (("energy", atoms.info), ("forces", atoms.arrays),
                                  ("stress", atoms.info)):
                if f"REF_{kind}" in storage:
                    storage[f"{name}_{kind}"] = storage.pop(f"REF_{kind}")
            # Decoys detect accidentally reading the default key, even where
            # the requested custom label is absent.
            atoms.info["REF_energy"] = 999.0
            atoms.arrays["REF_forces"] = np.full((len(atoms), 3), 99.0)
            atoms.info["REF_stress"] = np.eye(3) * 9.0
            raw.append(atoms)
        source = root / f"{name}.xyz"
        canonical = root / f"{name}-canonical.xyz"
        write(source, raw)
        write(canonical, normalized)
        destination = root / f"{name}-prepared"
        prepare(
            "--train_file", str(source), "--valid_file", str(source),
            "--test_file", str(source), "--output", str(destination),
            "--energy_key", f"{name}_energy", "--forces_key", f"{name}_forces",
            "--stress_key", f"{name}_stress", "--E0s", "{1: -13.6, 8: -2040.0}",
            "--num_process", "1", "--shuffle", "false", "--shard_size", "5",
        )
        for split in ("train", "valid", "test"):
            prepared = list(open_dataset(destination / split, key_spec=KeySpecification.from_defaults()).iter_range())
            expected = list(open_dataset(canonical, key_spec=KeySpecification.from_defaults()).iter_range())
            assert len(prepared) == len(expected) == 12
            for actual, reference in zip(prepared, expected, strict=True):
                for kind in ("energy", "forces", "stress"):
                    assert actual.is_labelled(kind) == reference.is_labelled(kind)
                    assert actual.property_weights[kind] == reference.property_weights[kind]
                    if reference.is_labelled(kind):
                        close(actual.properties[kind], reference.properties[kind])
        heads[name] = canonical, destination
    return root, document, heads


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("finetune", [False, True])
def test_custom_and_missing_labels_preserve_training(labelled_sources, tmp_path, stress, finetune):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.data import KeySpecification
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.finetune.stages import build
    from safetensors.numpy import load_file

    root, base, sources = labelled_sources
    document = copy.deepcopy(base)
    if stress:
        document["model"]["observables"].append("stress")
        document["loss"]["weights"]["stress"] = 2.0
    if finetune:
        document["finetune"] = {"foundation_model": str(root / "foundation/tiny.safetensors")}
    names = ("pbe", "mp2") if finetune else ("pbe",)
    for mode in ("normalized", "prepared"):
        document["data"]["heads"] = {
            name: {
                **{f"{split}_file": str(sources[name][0] if mode == "normalized" else sources[name][1] / split)
                   for split in ("train", "valid", "test")},
                "e0s": {"kind": "table", "values": {1: -13.6, 8: -2040.0}},
                **({"readout_from": "default"} if finetune else {}),
            } for name in names
        }
        launch(tmp_path, document, mode)

    actual, expected = [load_file(tmp_path / mode / "tiny.safetensors") for mode in ("prepared", "normalized")]
    assert actual.keys() == expected.keys()
    for key in expected:
        close(actual[key], expected[key])
    actual, expected = [training_records(tmp_path / mode) for mode in ("prepared", "normalized")]
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left == pytest.approx(right, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol)
    assert {row["head"] for row in actual if row["mode"] == "valid"} == set(names)
    assert all(np.isfinite(row["loss"]) for row in actual)
    if finetune:
        # Exercise the in-process stages: subprocess isolation alone cannot
        # detect mutation of the caller's configuration or the default keys.
        config = ResolvedConfig.from_dict(document)
        before = config.to_resolved_dict()
        keys = copy.deepcopy(KeySpecification.from_defaults())
        build(config, DEFAULT_CATALOGUE)
        assert config.to_resolved_dict() == before
        assert KeySpecification.from_defaults() == keys
    metadata = json.loads((tmp_path / "prepared/tiny.json").read_text())
    submitted = yaml.safe_load((tmp_path / "prepared.yaml").read_text())["data"]["heads"]
    assert list(metadata["config"]["config"]["resolved"]["data"]["heads"]) == list(submitted)
