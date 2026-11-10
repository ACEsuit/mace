"""A foundation model, back from its checkpoint and into a fine-tune's model.

The foundation is trained here, on waters and methanes, so it holds hydrogen,
carbon and oxygen. The fine-tune holds only hydrogen and oxygen and has two
heads. What has to survive is the function: on a water, every head of the
fine-tune's model, started from the foundation, computes the foundation's
interaction energy exactly, because the backbone read at two of its three
elements, a copied readout and a copied scale are the same model on a structure
with no carbon in it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data import GraphDataset, collate_training
from mace_torch.finetune.foundation import FoundationError, read_foundation
from mace_torch.finetune.transfer import (
    TransferError,
    readout_sources,
    transfer_foundation,
)
from mace_torch.models.build import build_model
from mace_torch.train import (
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_model,
)

CATALOGUE = DEFAULT_CATALOGUE
HYDROGEN, CARBON, OXYGEN = -13.6, -1030.0, -2040.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
METHANE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ]
)


def dataset(path, seed=0):
    generator = np.random.default_rng(seed)
    frames = []
    for number, energy in ((1, HYDROGEN), (6, CARBON), (8, OXYGEN)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(8):
        for symbols, positions, reference in (
            ("OH2", WATER, 2 * HYDROGEN + OXYGEN),
            ("CH4", METHANE, 4 * HYDROGEN + CARBON),
        ):
            molecule = Atoms(
                symbols,
                positions=positions
                + generator.normal(scale=0.03, size=positions.shape),
            )
            molecule.info["REF_energy"] = reference + 0.05 * index
            molecule.arrays["REF_forces"] = generator.normal(
                scale=0.1, size=positions.shape
            )
            frames.append(molecule)
    write(path, frames)
    return path


@pytest.fixture(
    scope="module", params=[False, True], ids=["per-element", "shared-product"]
)
def checkpoint(tmp_path_factory, request):
    """A foundation trained for two epochs and written to disk."""
    torch.set_default_dtype(torch.float64)
    directory = tmp_path_factory.mktemp("foundation")
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 3},
            "data": {
                "heads": {
                    "pbe": {
                        "train_file": str(dataset(directory / "train.xyz")),
                        "e0s": {"kind": "isolated_atoms"},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 1,
                "num_interactions": 2,
                "correlation": 2,
                "use_agnostic_product": request.param,
            },
            "training": {
                "max_num_epochs": 2,
                "batch_size": 4,
                "valid_batch_size": 4,
                "scheduler": {"kind": "constant"},
            },
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    # Written here rather than taken from the run: the run writes its best
    # epoch, and what this file compares against is the model it holds.
    path = write_model(directory / "model", trained.engine, built.metadata)
    return path, trained.engine


def water_graph(z_table, head=0):
    configuration = run_water(z_table)
    batch = collate_training([configuration])
    batch.graph["head"] = torch.tensor([head])
    return batch.graph


def run_water(z_table):
    from mace_core.data.configuration import Configuration

    dataset_ = GraphDataset(
        [
            Configuration(
                atomic_numbers=np.array([8, 1, 1]),
                positions=WATER + 0.01,
                properties={},
            )
        ],
        cutoff=3.0,
        z_table=z_table,
        targets=(),
    )
    return dataset_[0]


# ---------------------------------------------------------------------------
# The checkpoint, rebuilt
# ---------------------------------------------------------------------------


@fp64_only
def test_a_checkpoint_rebuilds_into_the_model_that_wrote_it(checkpoint):
    path, original = checkpoint
    foundation = read_foundation(path, CATALOGUE)
    graph = water_graph(foundation.z_table)
    expected = original(dict(graph), compute=()).total_energy
    got = foundation.engine(dict(graph), compute=()).total_energy
    assert torch.equal(expected, got)


@fp64_only
def test_the_record_gives_the_elements_heads_and_energies(checkpoint):
    foundation = read_foundation(checkpoint[0], CATALOGUE)
    assert list(foundation.z_table.zs) == [1, 6, 8]
    assert foundation.heads == ("pbe",)
    assert foundation.e0s["pbe"][6] == CARBON


@fp64_only
@pytest.mark.parametrize("run_checkpoint", [False, True])
def test_checkpoint_preserves_head_identity_after_reload(tmp_path, run_checkpoint):
    import json

    from mace_torch.train import RunState, read_run_checkpoint, write_run_checkpoint
    from mace_torch.train.optimizers import build_optimizer
    from test_mace_torch_training_pipeline import configuration

    document = configuration(tmp_path, max_num_epochs=0).model_dump(mode="json")
    source = document["data"]["heads"]["default"]
    document["data"]["heads"] = {
        name: {**source, "e0s": {"kind": "table", "values": {1: hydrogen, 8: OXYGEN}}}
        for name, hydrogen in (("zeta", HYDROGEN), ("alpha", HYDROGEN + 2.0))
    }
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    if run_checkpoint:
        path = write_run_checkpoint(
            tmp_path,
            "ordered",
            RunState(0),
            model=built.engine,
            optimizer=build_optimizer(built.engine, config.training),
            metadata=built.metadata,
        )
        recorded = json.loads(path.read_text())["config"]
        restored_config = ResolvedConfig.model_validate(recorded["config"]["resolved"])
        restored_data = run_data_stage(restored_config, CATALOGUE)
        restored = run_model_stage(restored_config, restored_data, CATALOGUE)
        read_run_checkpoint(
            path,
            model=restored.engine,
            optimizer=build_optimizer(restored.engine, config.training),
        )
        heads, engine = restored_data.heads, restored.engine
    else:
        path = write_model(tmp_path / "ordered", built.engine, built.metadata)
        restored = read_foundation(path, CATALOGUE)
        heads, engine = restored.heads, restored.engine
    assert heads == ("zeta", "alpha")
    energies = []
    for index in range(2):
        expected = built.engine(water_graph(data.z_table, index), compute=("forces",))
        actual = engine(water_graph(data.z_table, index), compute=("forces",))
        assert torch.equal(actual.total_energy, expected.total_energy)
        assert torch.equal(actual.forces, expected.forces)
        energies.append(actual.total_energy)
    assert not torch.equal(*energies)


# ---------------------------------------------------------------------------
# Into a smaller model with two heads
# ---------------------------------------------------------------------------


def fine_tune_model(foundation, heads=("replay", "target"), elements=(1, 8)):
    """The fine-tune's model: the foundation's architecture, fewer elements,
    its own heads, and energies nothing like the foundation's."""
    table = AtomicNumberTable(list(elements))
    energies = ResolvedE0s(
        {
            head: {z: -1.0 * (position + 1) * z for z in elements}
            for position, head in enumerate(heads)
        }
    )
    engine, _ = build_model(
        foundation.config,
        CATALOGUE,
        z_table=table,
        heads=tuple(heads),
        e0s=energies,
        statistics=DatasetStatistics(avg_num_neighbors=1.0),
    )
    return engine, table


@fp64_only
@pytest.mark.parametrize("elements", [(1, 8), (1, 6, 8)])
def test_every_head_computes_the_foundations_interaction_energy(checkpoint, elements):
    """The whole transfer in one number per head: the backbone read at two of
    three elements, the copied readout, the copied scale and shift."""
    foundation = read_foundation(checkpoint[0], CATALOGUE)
    engine, table = fine_tune_model(foundation, elements=elements)
    transfer_foundation(
        foundation.model,
        engine.get_submodule("model"),
        foundation_elements=foundation.z_table.zs,
        elements=table.zs,
        foundation_heads=foundation.heads,
        heads=("replay", "target"),
        readout_from={"replay": "pbe", "target": "pbe"},
    )
    expected = foundation.engine(
        dict(water_graph(foundation.z_table)), compute=("forces",)
    )
    for head in (0, 1):
        got = engine(dict(water_graph(table, head)), compute=("forces",))
        assert torch.allclose(
            got.extras["interaction_energy"],
            expected.extras["interaction_energy"],
            rtol=0,
            atol=1e-12,
        ), head
        from tests.golden.harness import CLOSED_FORM_FP64

        torch.testing.assert_close(
            got.forces,
            expected.forces,
            rtol=CLOSED_FORM_FP64.rtol,
            atol=CLOSED_FORM_FP64.atol,
        )


@fp64_only
def test_the_heads_keep_their_own_isolated_atom_energies(checkpoint):
    """Copied everything else, and not these: each head's were resolved for
    it, and that is what a head is."""
    foundation = read_foundation(checkpoint[0], CATALOGUE)
    engine, table = fine_tune_model(foundation)
    before = engine.get_submodule("model.outputs.energy_readout").e0_table.clone()
    transfer_foundation(
        foundation.model,
        engine.get_submodule("model"),
        foundation_elements=foundation.z_table.zs,
        elements=table.zs,
        foundation_heads=foundation.heads,
        heads=("replay", "target"),
        readout_from={"replay": "pbe", "target": "pbe"},
    )
    after = engine.get_submodule("model.outputs.energy_readout").e0_table
    assert torch.equal(before, after)


@fp64_only
def test_without_the_readout_only_the_backbone_is_copied(checkpoint):
    foundation = read_foundation(checkpoint[0], CATALOGUE)
    engine, table = fine_tune_model(foundation)
    transfer_foundation(
        foundation.model,
        engine.get_submodule("model"),
        foundation_elements=foundation.z_table.zs,
        elements=table.zs,
        foundation_heads=foundation.heads,
        heads=("replay", "target"),
        readout_from={"replay": "pbe", "target": "pbe"},
        transfer_readout=False,
    )
    expected = foundation.engine(
        dict(water_graph(foundation.z_table)), compute=()
    ).extras["interaction_energy"]
    got = engine(dict(water_graph(table, 0)), compute=()).extras["interaction_energy"]
    assert not torch.allclose(got, expected)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_head_naming_no_source_takes_the_only_one():
    assert readout_sources(("a", "b"), {}, ("pbe",)) == {"a": "pbe", "b": "pbe"}


def test_a_head_naming_no_source_among_several_is_refused():
    with pytest.raises(TransferError, match="readout_from"):
        readout_sources(("a",), {}, ("pbe", "r2scan"))


def test_a_source_the_foundation_lacks_is_refused():
    with pytest.raises(TransferError, match="does not have"):
        readout_sources(("a",), {"a": "scan"}, ("pbe",))


@fp64_only
def test_an_element_outside_the_foundation_is_refused(checkpoint):
    foundation = read_foundation(checkpoint[0], CATALOGUE)
    engine, table = fine_tune_model(foundation, elements=(1, 7))
    with pytest.raises(TransferError, match=r"elements \[7\]"):
        transfer_foundation(
            foundation.model,
            engine.get_submodule("model"),
            foundation_elements=foundation.z_table.zs,
            elements=table.zs,
            foundation_heads=foundation.heads,
            heads=("replay", "target"),
            readout_from={"replay": "pbe", "target": "pbe"},
        )


def test_a_checkpoint_recording_no_heads_is_refused(tmp_path):
    """A native checkpoint must record each head's element energies."""
    import json

    from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
    from mace_torch.serialization import VERSION

    config = ResolvedConfig.model_validate(
        {
            "data": {"heads": {"a": {"train_file": "x.xyz"}}},
            "model": {"observables": ["energy"]},
        }
    )
    record = ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(versions={"mace-torch": "0"}),
    )
    (tmp_path / "missing_heads.json").write_text(
        json.dumps(
            {
                "format": "mace-v1-checkpoint",
                "version": VERSION,
                "config": record.model_dump(mode="json"),
                "layout": {},
                "tensors": [],
            }
        )
    )
    with pytest.raises(FoundationError, match="records no isolated-atom energies"):
        read_foundation(tmp_path / "missing_heads.safetensors", CATALOGUE)


# ---------------------------------------------------------------------------
# A foundation with two heads
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_head_checkpoint(tmp_path_factory):
    """Two levels of theory on the same structures: the second head's data
    is drawn with another seed, so its readout learns something else."""
    torch.set_default_dtype(torch.float64)
    directory = tmp_path_factory.mktemp("two_heads")
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 3},
            "data": {
                "heads": {
                    "pbe": {
                        "train_file": str(dataset(directory / "pbe.xyz")),
                        "e0s": {"kind": "isolated_atoms"},
                    },
                    "r2scan": {
                        "train_file": str(dataset(directory / "r2scan.xyz", seed=1)),
                        "e0s": {"kind": "isolated_atoms"},
                    },
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 1,
                "num_interactions": 2,
                "correlation": 2,
            },
            "training": {
                "max_num_epochs": 2,
                "batch_size": 4,
                "valid_batch_size": 4,
                "scheduler": {"kind": "constant"},
            },
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    return write_model(directory / "model", trained.engine, built.metadata)


def foundation_energy(foundation, head):
    graph = dict(water_graph(foundation.z_table, head))
    return foundation.engine(graph, compute=()).extras["interaction_energy"]


@fp64_only
def test_each_head_takes_the_readout_it_names(two_head_checkpoint):
    """The frozen tree's `--foundation_head`: the new head reads out with the
    foundation head it names, here the second one, not the first."""
    foundation = read_foundation(two_head_checkpoint, CATALOGUE)
    assert foundation.heads == ("pbe", "r2scan")
    pbe, r2scan = (foundation_energy(foundation, head) for head in (0, 1))
    assert not torch.allclose(pbe, r2scan)

    engine, table = fine_tune_model(foundation)
    transfer_foundation(
        foundation.model,
        engine.get_submodule("model"),
        foundation_elements=foundation.z_table.zs,
        elements=table.zs,
        foundation_heads=foundation.heads,
        heads=("replay", "target"),
        readout_from=readout_sources(
            ("replay", "target"),
            {"replay": "pbe", "target": "r2scan"},
            foundation.heads,
        ),
    )
    replay, target = (
        engine(dict(water_graph(table, head)), compute=()).extras["interaction_energy"]
        for head in (0, 1)
    )
    assert torch.allclose(replay, pbe, rtol=0, atol=1e-12)
    assert torch.allclose(target, r2scan, rtol=0, atol=1e-12)


def test_a_named_source_is_kept_as_named():
    assert readout_sources(
        ("replay", "target"), {"target": "r2scan", "replay": "pbe"}, ("pbe", "r2scan")
    ) == {"replay": "pbe", "target": "r2scan"}
