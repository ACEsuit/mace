"""The committed anchor as a foundation model, and a fine-tune started from it.

The frozen tree's fine-tuning contract starts from a published foundation model
and trains a replay head beside a new one. Here the foundation is the committed
scale-shift anchor instead, converted into a v1 checkpoint with a record, so the
contract runs without the network and from a model whose numbers are pinned:
the checkpoint read back has to reproduce the anchor before anything is trained
on top of it.

The replay head is called ``pt_head`` because that is what the frozen tree
calls it. Nothing in v1 reads the name; the head replays because its
configuration says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase.data import chemical_symbols
from ase.io import read, write
from mace_core.config.provenance import e0_details
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import ConfigRecord, HeadSummary, ModelMetadata, Provenance
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.finetune.foundation import read_foundation
from mace_torch.train import (
    evaluate,
    metric_specs,
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_model,
)
from mace_torch.train.loss import build_loss
from mace_torch.train.model_stage import build_model

from tests.parity.anchors import load_anchor
from tests.parity.fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    transfer_weights,
)

CATALOGUE = DEFAULT_CATALOGUE
TRAIN_SET = (
    Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "tiny_train.xyz"
)
ANCHOR = "tiny_scaleshift.model"


def anchor_config(legacy, work_dir: Path) -> ResolvedConfig:
    """The anchor's architecture as a v1 configuration, read off the model."""
    built = build_config(legacy)
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(work_dir), "seed": 20260810},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(TRAIN_SET),
                        "e0s": {"kind": "isolated_atoms"},
                    }
                },
                "valid_fraction": 0.25,
            },
            "model": {
                "model": "scale_shift",
                "observables": ["energy", "forces"],
                "interaction_first": "plain",
                "r_max": built["cutoff"],
                "num_interactions": built["num_layers"],
                "num_channels": built["num_features"],
                "hidden_irreps": built["hidden_irreps"],
                "max_ell": built["lmax"],
                "correlation": built["correlation"],
                "radial_basis": {"kind": "bessel", "num_basis": built["num_radial"]},
                "cutoff": {"polynomial_order": built["cutoff_order"]},
                "pair_repulsion": built["pair_repulsion"],
                "readout": {"mlp_irreps": f"{built['readout_hidden']}x0e"},
            },
        }
    )


def write_anchor_checkpoint(legacy, directory: Path) -> Path:
    """The anchor's weights and constants, with the record a v1 run writes."""
    built = build_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy)
    config = anchor_config(legacy, directory)
    engine, _ = build_model(
        config,
        CATALOGUE,
        z_table=AtomicNumberTable(built["atomic_numbers"]),
        heads=("default",),
        e0s=ResolvedE0s(values),
        statistics=DatasetStatistics(
            avg_num_neighbors=built["avg_num_neighbors"],
            mean=shift[0],
            std=scale[0],
        ),
        initialize=False,
    )
    transfer_weights(legacy, engine.get_submodule("backbone"), built["correlation"])
    metadata = ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(versions={"mace-torch-v1": "anchor"}),
        heads={
            "default": HeadSummary(
                e0=e0_details(
                    config.data.heads["default"].e0s,
                    {
                        chemical_symbols[number]: energy
                        for number, energy in values["default"].items()
                    },
                )
            )
        },
    )
    return write_model(directory / "anchor", engine, metadata)


def a_new_level_of_theory(path: Path) -> Path:
    """The anchor's structures relabelled: energies moved by a per-atom offset
    and forces scaled, which is what a new head exists to absorb."""
    frames = []
    for atoms in read(TRAIN_SET, ":"):
        if atoms.info.get("config_type") != "IsolatedAtom":
            atoms.info["REF_energy"] += 0.1 * len(atoms)
            atoms.arrays["REF_forces"] = 1.2 * atoms.arrays["REF_forces"]
        frames.append(atoms)
    write(path, frames)
    return path


def fine_tune_config(directory: Path, foundation: Path) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 7},
            "finetune": {"foundation_model": str(foundation)},
            "data": {
                "heads": {
                    "pt_head": {
                        "train_file": str(TRAIN_SET),
                        "e0s": {"kind": "foundation"},
                        "weight": 0.5,
                    },
                    "DFT": {
                        "train_file": str(a_new_level_of_theory(directory / "dft.xyz")),
                        "e0s": {"kind": "isolated_atoms"},
                    },
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
                "skip_evaluate_heads": ["pt_head"],
            },
            "model": {"observables": ["energy", "forces"]},
            "training": {
                "max_num_epochs": 6,
                "batch_size": 4,
                "valid_batch_size": 4,
                "lr": 0.01,
                "scheduler": {"kind": {"kind": "constant"}},
            },
        }
    )


@pytest.fixture(name="anchor_foundation")
def fixture_anchor_foundation(fp64, tmp_path):
    legacy = load_anchor(ANCHOR)
    return legacy, write_anchor_checkpoint(legacy, tmp_path)


def energies(engine, legacy, structures):
    """The v1 model's total energies on the batch the anchor reads."""
    from tests.golden.anchors import anchor_batch

    batch = anchor_batch(legacy, structures, torch.float64)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    graph = {
        "positions": batch.positions,
        "node_atomic_numbers": torch.tensor(
            [numbers[index] for index in batch.node_attrs.argmax(1).tolist()]
        ),
        "node_attrs": batch.node_attrs,
        "edge_index": batch.edge_index,
        "shifts": batch.shifts,
        "unit_shifts": batch.unit_shifts,
        "cell": batch.cell.reshape(-1, 3, 3),
        "batch": batch.batch,
        "ptr": torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.bincount(batch.batch).cumsum(0))
        ),
        "head": torch.zeros(int(batch.num_graphs), dtype=torch.long),
    }
    return engine(graph, compute=("forces",)).total_energy.detach()


def test_the_checkpoint_read_back_is_the_anchor(anchor_foundation):
    from tests.golden.anchors import anchor_batch, load_training_structures

    legacy, path = anchor_foundation
    structures = load_training_structures()
    expected = legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()
    foundation = read_foundation(path, CATALOGUE)
    got = energies(foundation.engine, legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


def new_head_error(built, config) -> float:
    metrics = evaluate(
        built.model,
        built.data.valid_loaders["DFT"],
        build_loss(built.outputs, config.loss),
        metric_specs(built.outputs),
    )
    return metrics["rmse_energy_per_atom"]


def test_a_fine_tune_from_the_anchor_trains_both_heads(anchor_foundation, tmp_path):
    """The frozen tree's contract: the run completes, the model holds the
    replay head and the new one, and the new data's error goes down."""
    _, path = anchor_foundation
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(tmp_path, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    built = run_model_stage(config, data, CATALOGUE, foundation=foundation)
    before = new_head_error(built, config)
    trained = run_train_stage(config, built)
    after = new_head_error(built, config)

    assert data.heads == ("pt_head", "DFT")
    assert built.model.get_submodule("backbone.outputs.heads.energy").num_heads == 2
    assert trained.best_epoch is not None
    assert after < before, f"the new head's error went from {before:.4g} to {after:.4g}"


def test_the_replay_head_keeps_the_anchors_energies(anchor_foundation, tmp_path):
    """Its E0s are read from the record, not from the loaded module."""
    legacy, path = anchor_foundation
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(tmp_path, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    values, _, _ = energy_constants_to_canonical(legacy)
    assert data.e0s.values["pt_head"] == pytest.approx(values["default"], abs=0)


def test_the_replay_head_starts_as_the_anchor(anchor_foundation, tmp_path):
    """Before any training the replay head computes what the foundation does,
    which is what lets its data be replayed rather than relearned."""
    from tests.golden.anchors import anchor_batch, load_training_structures

    legacy, path = anchor_foundation
    structures = load_training_structures()
    expected = legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()
    foundation = read_foundation(path, CATALOGUE)
    config = fine_tune_config(tmp_path, path)
    data = run_data_stage(config, CATALOGUE, foundation=foundation.context())
    built = run_model_stage(config, data, CATALOGUE, foundation=foundation)
    assert data.heads.index("pt_head") == 0
    got = energies(built.model, legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


@pytest.mark.parametrize("selection_dtype", ["float32", "float64"])
def test_foundation_selection_descriptors_match_legacy_element_means(
    anchor_foundation, selection_dtype
):
    """Selection consumes the reviewed descriptor API, including absent elements."""
    import numpy as np
    from ase import Atoms
    from mace_core.config.precision import PrecisionConfig
    from mace_core.data.configuration import Configuration
    from mace_torch.finetune.subselect import ABSENT

    from mace.calculators import MACECalculator
    from tests.golden.harness import FP32

    legacy, path = anchor_foundation
    foundation = read_foundation(
        path,
        CATALOGUE,
        precision=PrecisionConfig(model=selection_dtype, accumulate=selection_dtype),
    )
    calculator = MACECalculator(
        models=[legacy], device="cpu", default_dtype=selection_dtype
    )
    structures = [
        Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
    ]
    configurations = [
        Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            head=foundation.heads[0],
        )
        for atoms in structures
    ]
    expected = []
    for atoms in structures:
        node_features = calculator.get_descriptors(atoms, invariants_only=True)
        rows = []
        for number in foundation.z_table.zs:
            selected = node_features[atoms.numbers == number]
            rows.append(
                selected.mean(axis=0)
                if len(selected)
                else np.full(node_features.shape[1], ABSENT)
            )
        expected.append(np.concatenate(rows))
    if selection_dtype == "float32":
        np.testing.assert_allclose(
            foundation.describe(configurations),
            expected,
            atol=FP32.atol,
            rtol=FP32.rtol,
        )
    else:
        np.testing.assert_allclose(
            foundation.describe(configurations), expected, atol=1e-12, rtol=0
        )
