"""A fine-tune's element table against the frozen tree's, in one process.

The committed scale-shift anchor, over hydrogen, carbon and oxygen, is the
foundation, and the fine-tune's data holds only hydrogen and oxygen.

**The data's table** is the frozen tree's default, and it is v1's explicit
``element_table = "data"``. The frozen tree builds a model over the two
elements and slices the anchor into it with ``load_foundations_elements``,
dividing five families of per-element tensors by
``(num_species_foundations / num_species) ** 0.5``; v1 slices the canonical
rows and rescales nothing, because a canonical weight holds its normalization
already. The two models have to agree before any training.

**The foundation's table** is v1's default. The model keeps carbon, which no
structure holds, and it has to compute on hydrogen and oxygen exactly what the
anchor does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.finetune.foundation import read_foundation
from mace_torch.train import run_data_stage, run_model_stage

from tests.parity.test_anchor_as_foundation import (
    ANCHOR,
    energies,
    write_anchor_checkpoint,
)
from tests.parity.test_fm00_training_step import load_anchor

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def waters(count: int = 8):
    generator = np.random.default_rng(3)
    frames = []
    for index in range(count):
        atoms = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.05, size=(3, 3))
        )
        atoms.info["REF_energy"] = -2.0 + 0.01 * index
        atoms.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(atoms)
    return frames


def fine_tune(tmp_path, foundation_path, element_table):
    path = tmp_path / "waters.xyz"
    write(path, waters())
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 7},
            "finetune": {
                "foundation_model": str(foundation_path),
                "element_table": element_table,
            },
            "data": {
                "heads": {
                    "default": {"train_file": str(path), "e0s": {"foundation": {}}}
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces"]},
        }
    )
    foundation = read_foundation(foundation_path, DEFAULT_CATALOGUE)
    data = run_data_stage(config, DEFAULT_CATALOGUE, foundation=foundation.context())
    return run_model_stage(config, data, DEFAULT_CATALOGUE, foundation=foundation)


def legacy_subset(anchor, numbers):
    """The frozen tree's fine-tune model over ``numbers``, sliced from the
    anchor the way ``--foundation_model_elements False`` slices it."""
    from mace.tools import AtomicNumberTable
    from mace.tools.finetuning_utils import load_foundations_elements
    from mace.tools.scripts_utils import extract_config_mace_model

    config = extract_config_mace_model(anchor)
    positions = [anchor.atomic_numbers.tolist().index(z) for z in numbers]
    config.update(
        atomic_numbers=list(numbers),
        num_elements=len(numbers),
        atomic_energies=config["atomic_energies"][..., positions],
    )
    model = type(anchor)(**config).to(torch.float64)
    load_foundations_elements(
        model,
        anchor,
        AtomicNumberTable(list(numbers)),
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=1,
    )
    return model.eval()


def water_structures():
    return waters(4)


def forces_and_energies(engine, legacy, structures):
    from tests.golden.anchors import anchor_batch

    batch = anchor_batch(legacy, structures, torch.float64)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    graph = {
        "positions": batch.positions,
        "atomic_numbers": torch.tensor(
            [numbers[index] for index in batch.node_attrs.argmax(1).tolist()]
        ),
        "element_index": batch.node_attrs.argmax(1),
        "edge_index": batch.edge_index,
        "shifts": batch.shifts,
        "unit_shifts": batch.unit_shifts,
        "cell": batch.cell.reshape(-1, 3, 3),
        "batch": batch.batch,
        "num_graphs": int(batch.num_graphs),
        "head": torch.zeros(int(batch.num_graphs), dtype=torch.long),
    }
    output = engine(graph, compute=("forces",))
    return output.total_energy.detach(), output.forces.detach()


def legacy_forces_and_energies(legacy, structures):
    from tests.golden.anchors import anchor_batch

    output = legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=True,
    )
    return output["energy"].detach(), output["forces"].detach()


@pytest.fixture(name="anchor_foundation")
def fixture_anchor_foundation(fp64, tmp_path):
    legacy = load_anchor(ANCHOR)
    return legacy, write_anchor_checkpoint(legacy, tmp_path)


def test_the_data_s_table_is_the_frozen_tree_s_sliced_model(
    anchor_foundation, tmp_path
):
    anchor, path = anchor_foundation
    built = fine_tune(tmp_path, path, "data")
    assert list(built.data.z_table.zs) == [1, 8]
    legacy = legacy_subset(anchor, [1, 8])
    structures = water_structures()
    energy, forces = forces_and_energies(built.model, legacy, structures)
    expected_energy, expected_forces = legacy_forces_and_energies(legacy, structures)
    assert torch.allclose(energy, expected_energy, atol=1e-12, rtol=0), float(
        (energy - expected_energy).abs().max()
    )
    assert torch.allclose(forces, expected_forces, atol=1e-12, rtol=0), float(
        (forces - expected_forces).abs().max()
    )


def test_the_frozen_tree_divides_the_rows_it_keeps(anchor_foundation):
    """The rescale the comparison above passes through: the frozen tree's kept
    embedding rows are the anchor's divided by ``sqrt(3 / 2)``, and the two
    models still agree, because v1's canonical rows carry the normalization the
    division compensates for."""
    anchor, _ = anchor_foundation
    rescaled = legacy_subset(anchor, [1, 8])
    kept = anchor.node_embedding.linear.weight.detach().view(3, -1)[[0, 2]]
    assert torch.allclose(
        rescaled.node_embedding.linear.weight.detach().view(2, -1) * (3 / 2) ** 0.5,
        kept,
        atol=1e-14,
    )


def test_the_foundation_s_table_computes_what_the_anchor_does(
    anchor_foundation, tmp_path
):
    anchor, path = anchor_foundation
    built = fine_tune(tmp_path, path, "foundation")
    assert list(built.data.z_table.zs) == [1, 6, 8]
    structures = water_structures()
    energy = energies(built.model, anchor, structures)
    expected, _ = legacy_forces_and_energies(anchor, structures)
    assert torch.allclose(energy, expected, atol=1e-12, rtol=0), float(
        (energy - expected).abs().max()
    )
