"""Element scope and replay balancing survive native training and reload."""

import copy
import gc
import hashlib
import sys

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_data_sources_v1 import close, launch, prepare
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, tiny_task, training_records

pytestmark = needs_the_v1_engine
ENERGIES = {1: -13.6, 6: -1030.0, 8: -2040.0}


@pytest.fixture(scope="module")
def replay_foundation(tmp_path_factory):
    root = tmp_path_factory.mktemp("replay-elements")
    document = yaml.safe_load(tiny_task(root).read_text())
    waters = read(root / "train.xyz", ":")[2:]
    mixed = []
    for index, water in enumerate(waters):
        atoms = water.copy()
        atoms += Atoms("C", positions=[[1.8, 0.2, 0.3 + index * 0.01]])
        atoms.info["REF_energy"] += ENERGIES[6]
        atoms.arrays["REF_forces"][-1] = [0.01, -0.02, 0.03]
        mixed.append(atoms)
    carbon = Atoms("C", positions=[[0, 0, 0]])
    carbon.info.update(REF_energy=ENERGIES[6], config_type="IsolatedAtom")
    carbon.arrays["REF_forces"] = np.zeros((1, 3))
    original = read(root / "train.xyz", ":")
    write(root / "foundation.xyz", [*original, carbon, *mixed])
    document["data"]["heads"]["default"]["train_file"] = str(root / "foundation.xyz")
    document["training"].update(max_num_epochs=2, stage_two={"enabled": True, "start_epoch": 1})
    launch(root, document, "foundation")
    for name, frames in (("pbe", waters[:2]), ("mp2", waters[2:5]),
                         ("memory", waters[:10]), ("valid", waters[10:]),
                         ("mixed", mixed)):
        write(root / f"{name}.xyz", frames)
    return root


def replay_document(root, *, multihead=True, element_table="foundation"):
    heads = {
        name: {
            "train_file": str(root / f"{name}.xyz"),
            "valid_file": str(root / "valid.xyz"),
            "e0s": {"kind": "foundation", "head": "default"},
            "readout_from": "default",
        }
        for name in (("memory", "mp2", "pbe") if multihead else ("pbe",))
    }
    return {
        "runtime": {"name": "fit", "seed": 23},
        "precision": {"model": "float64"},
        "finetune": {"foundation_model": str(root / "foundation/tiny.safetensors"),
                     "element_table": element_table},
        "data": {"heads": heads, "pin_memory": False},
        "model": {"observables": ["energy", "forces"]},
        "training": {"max_num_epochs": 2, "batch_size": 2, "valid_batch_size": 2,
                     "lr": 0.002, "scheduler": {"kind": "constant"}},
        "loss": {"weights": {"energy": 1.0, "forces": 10.0}},
    }


def built_data(document):
    from mace_core.config.resolved import ResolvedConfig
    from mace_torch.finetune.stages import build

    return build(ResolvedConfig.from_dict(document))


def verify_trained(directory, names, elements):
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed

    model = load_deployed(directory / "fit.safetensors")
    assert set(model.heads) == set(names)
    assert list(model.z_table.zs) == elements
    records = training_records(directory, "fit")
    updates = [row for row in records if row["mode"] == "opt"]
    assert updates and all(np.isfinite(row["loss"]) for row in updates)
    assert {row["head"] for row in records if row["mode"] == "valid" and row["epoch"] == 1} == set(names)
    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    for name in names:
        assert set(model.e0s[name]) == set(elements)
        water.calc = MACECalculator(models=model, head=name)
        assert np.isfinite(water.get_potential_energy())
        assert np.isfinite(water.get_forces()).all()
    return model


@pytest.mark.parametrize("multihead", [False, True])
@pytest.mark.parametrize("element_table", ["data", "foundation"])
def test_element_scope_survives_finetuning_and_reload(replay_foundation, tmp_path, multihead, element_table):
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed

    root = replay_foundation
    document = replay_document(root, multihead=multihead, element_table=element_table)
    if multihead:
        document["data"]["heads"]["memory"]["subselect"] = {"method": "random", "num_samples": 3}
    built = built_data(document)
    if multihead:
        assert len(built.data.train_loader.datasets["memory"]) == 3
    elements = [1, 8] if element_table == "data" else [1, 6, 8]
    assert list(built.data.z_table.zs) == elements
    parent = load_deployed(root / "foundation/tiny.safetensors")
    # Same water, with graph embeddings rebuilt for each model's element table.
    from mace_core.data import KeySpecification, configuration_from_atoms
    from mace_torch.data import GraphDataset, collate_training

    structures = [configuration_from_atoms(atoms, KeySpecification.from_defaults(), head_name="default")
                  for atoms in read(root / "valid.xyz", ":")]
    expected_dataset = GraphDataset(structures, cutoff=parent.config.model.r_max,
                                   z_table=parent.z_table, heads=("default",), targets=())
    parent_batch = collate_training([expected_dataset[i] for i in range(len(expected_dataset))],
                                   float_dtype="float64")
    expected = parent.engine(parent_batch.graph, compute=("forces",))
    for name, loader in built.data.valid_loaders.items():
        batch = next(iter(loader))
        actual = built.model(batch.graph, compute=("forces",))
        close(actual.total_energy.detach(), expected.total_energy.detach())
        close(actual.forces.detach(), expected.forces.detach())
        assert built.data.e0s.values[name] == {number: ENERGIES[number] for number in elements}
    launch(tmp_path, document, "trained")
    model = verify_trained(tmp_path / "trained", document["data"]["heads"], elements)
    if multihead:
        calculator = MACECalculator(models=model, head="memory")
        predictions = []
        for atoms in read(root / "memory.xyz", ":"):
            atoms.calc = calculator
            predictions.append(atoms.get_potential_energy())
        assert np.isfinite(predictions).all() and len(set(predictions)) > 1


def test_filtered_curated_replay_can_add_elements_to_target_heads(replay_foundation, tmp_path, monkeypatch):
    from mace_torch.finetune.replay import legacy_cached_path

    root = replay_foundation
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cache = legacy_cached_path("mp")
    cache.parent.mkdir(parents=True)
    frames = read(root / "mixed.xyz", ":")
    for atoms in frames:
        atoms.calc = SinglePointCalculator(atoms, energy=atoms.info.pop("REF_energy"),
                                          forces=atoms.arrays.pop("REF_forces"))
    write(cache, frames, format="extxyz")
    document = replay_document(root, element_table="data")
    memory = document["data"]["heads"]["memory"]
    memory.pop("train_file")
    memory.update(curated="mp", subselect={"method": "random", "num_samples": 10,
                  "filtering": "exclusive", "elements": [1, 8], "allow_random_padding": True})
    # The target levels only declare H/O references. Replay supplies carbon.
    for name in ("pbe", "mp2"):
        document["data"]["heads"][name]["e0s"] = {
            "kind": "table", "values": {1: ENERGIES[1], 8: ENERGIES[8]}}
    built = built_data(document)
    replay = built.data.train_loader.datasets["memory"].configurations
    assert len(replay) == 10
    assert all(set(item.atomic_numbers) == {1, 6, 8} for item in replay)
    for name in ("pbe", "mp2"):
        assert built.data.e0_provenance[name].from_foundation == (6,)
        assert built.data.e0s.values[name] == ENERGIES
    launch(tmp_path, document, "trained")
    verify_trained(tmp_path / "trained", document["data"]["heads"], [1, 6, 8])


@pytest.mark.parametrize("threshold,copies", [(0.1, 1), (0.5, 1), (0.500001, 2), (1.0, 3)])
def test_replay_ratio_matches_explicit_repetition(replay_foundation, tmp_path, threshold, copies):
    from safetensors.numpy import load_file

    document = replay_document(replay_foundation)
    document["data"]["ratio_guard"] = {"reference": "memory", "threshold": threshold}
    built = built_data(document)
    counts = {name: len(dataset) for name, dataset in built.data.train_loader.datasets.items()}
    assert counts == {"memory": 10, "mp2": 3 * copies, "pbe": 2 * copies}
    assert {name: len(loader.dataset) for name, loader in built.data.valid_loaders.items()} == {
        "memory": 2, "mp2": 2, "pbe": 2}
    launch(tmp_path, document, "automatic")
    explicit = copy.deepcopy(document)
    explicit["data"].pop("ratio_guard")
    for name in ("pbe", "mp2"):
        head = explicit["data"]["heads"][name]
        head["train_file"] = [head["train_file"]] * copies
    launch(tmp_path, explicit, "explicit")
    expected = load_file(tmp_path / "explicit/fit.safetensors")
    actual = load_file(tmp_path / "automatic/fit.safetensors")
    assert actual.keys() == expected.keys()
    for key in expected:
        close(actual[key], expected[key])
    for first, second in zip(training_records(tmp_path / "automatic", "fit"),
                             training_records(tmp_path / "explicit", "fit"), strict=True):
        assert first == pytest.approx(second, abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol)


def test_prepared_replay_explicitly_skips_the_ratio_guard(replay_foundation, tmp_path):
    root = replay_foundation
    document = replay_document(root)
    document["data"]["heads"]["memory"]["train_file"] = prepare(root / "memory.xyz", tmp_path / "shards")
    document["data"]["ratio_guard"] = {"reference": "memory", "threshold": 1.0}
    built = built_data(document)
    assert {name: len(dataset) for name, dataset in built.data.train_loader.datasets.items()} == {
        "memory": 10, "mp2": 3, "pbe": 2}
    result = launch(tmp_path, document, "trained")
    assert "ratio guard is skipped" in result.stdout + result.stderr
    verify_trained(tmp_path / "trained", document["data"]["heads"], [1, 6, 8])


@pytest.fixture(scope="module")
def published_small(tmp_path_factory):
    from mace_core.artifacts import resolve_artifact
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.legacy import convert_legacy
    from tests.golden.foundation_artifacts import ARTIFACTS

    artifact = ARTIFACTS["mp_small"]
    source = resolve_artifact(artifact.origin)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == artifact.sha256
    destination = tmp_path_factory.mktemp("published-small") / "foundation"
    converted = convert_legacy(source, destination, DEFAULT_CATALOGUE, python=sys.executable)
    path = converted.checkpoint.with_suffix(".safetensors")
    del converted
    gc.collect()
    return path


@pytest.mark.network
@pytest.mark.parametrize("multihead", [False, True])
@pytest.mark.parametrize("element_table", ["data", "foundation"])
def test_published_small_trains_with_selected_element_scope(
    replay_foundation, published_small, tmp_path, multihead, element_table
):
    from mace_torch.deploy.loader import load_deployed

    parent = load_deployed(published_small)
    all_elements = list(parent.z_table.zs)
    assert {1, 6, 8} < set(all_elements)
    source_head, = parent.heads
    del parent
    gc.collect()
    document = replay_document(replay_foundation, multihead=multihead, element_table=element_table)
    document["finetune"]["foundation_model"] = str(published_small)
    for head in document["data"]["heads"].values():
        head.update(readout_from=source_head, e0s={"kind": "foundation", "head": source_head})
    launch(tmp_path, document, "trained")
    expected = [1, 8] if element_table == "data" else all_elements
    verify_trained(tmp_path / "trained", document["data"]["heads"], expected)
