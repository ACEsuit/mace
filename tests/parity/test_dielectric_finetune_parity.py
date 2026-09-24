"""A dielectric fine-tune's starting model against the frozen tree's, in one process.

A small legacy ``AtomicDielectricMACE`` over hydrogen, carbon and oxygen is the
foundation. It is converted into a v1 checkpoint, and a fine-tune on waters is
built from it the way any v1 fine-tune is: no mode, only the dipole and
polarizability observables and a dielectric foundation model.

**The data's table** is compared with the frozen tree's
``load_foundations_mdp``, which builds a model over the two elements and slices
the foundation into it, dividing the node embedding and the skip connection by
``sqrt(num_species_foundations / num_species)`` and copying every angular
channel of the products. v1 has no transfer of its own for this family: the one
element transfer carries every canonical tensor whole, which is all of those
channels, and a canonical row holds its normalization already.

**The foundation's table**, v1's default, has to compute what the foundation
does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import (
    ConfigRecord,
    E0Details,
    HeadSummary,
    ModelMetadata,
    Provenance,
)
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.calculators import MACECalculator
from mace_torch.finetune.stages import build
from mace_torch.train import write_model
from mace_torch.train.model_stage import build_model

from tests.parity.dipole_convert import dipole_config, transfer_dipole_weights
from tests.parity.test_dipole_parity import (
    CASES,
    engine,
    gap,
    legacy_batch,
    legacy_dielectric,
    v1_graph,
)

OBSERVABLES = ["dipole", "polarizability"]
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def architecture(legacy) -> dict:
    """The legacy model's settings as a v1 model section."""
    config = dipole_config(legacy)
    return {
        "model": "dielectric",
        "observables": OBSERVABLES,
        "r_max": config["cutoff"],
        "num_interactions": config["num_layers"],
        "num_channels": config["num_features"],
        "hidden_irreps": config["hidden_irreps"],
        "max_ell": config["lmax"],
        "correlation": config["correlation"],
        "num_radial_basis": config["num_radial"],
        "num_cutoff_basis": config["cutoff_order"],
        "readout": {"mlp_irreps": config["readout_hidden"]},
    }


def write_foundation(legacy, directory):
    """The legacy model's weights, as a v1 checkpoint with its record."""
    train_file = waters(directory / "foundation.xyz")
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory)},
            "data": {"heads": {"default": {"train_file": str(train_file)}}},
            "model": architecture(legacy),
        }
    )
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    engine_, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(numbers),
        heads=("default",),
        e0s=ResolvedE0s({"default": {}}),
        statistics=DatasetStatistics(
            avg_num_neighbors=dipole_config(legacy)["avg_num_neighbors"]
        ),
        initialize=False,
    )
    transfer_dipole_weights(legacy, engine_.get_submodule("backbone"), "dielectric")
    metadata = ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(code_version="anchor"),
        heads={"default": HeadSummary(e0=E0Details(source="explicit"))},
        elements=[chemical_symbols[z] for z in numbers],
    )
    return write_model(directory / "foundation", engine_, metadata)


def waters(path, count=8):
    generator = np.random.default_rng(2)
    frames = []
    for _ in range(count):
        atoms = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.05, size=(3, 3))
        )
        atoms.info["dipole"] = generator.normal(size=3)
        atoms.info["polarizability"] = generator.normal(size=9)
        frames.append(atoms)
    write(path, frames)
    return path


def fine_tune_config(directory, foundation, element_table):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 4},
            "finetune": {
                "foundation_model": str(foundation),
                "element_table": element_table,
            },
            "data": {
                "heads": {"default": {"train_file": str(waters(directory / "w.xyz"))}},
                "valid_fraction": 0.25,
                "pin_memory": False,
            },
            "model": {"observables": OBSERVABLES},
        }
    )


def legacy_subset(foundation, numbers):
    """The frozen tree's dielectric fine-tune model over ``numbers``."""
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools.finetuning_utils import load_foundations_mdp

    torch.manual_seed(9)
    model = legacy_dielectric("8x0e + 8x1o + 8x2e", atomic_numbers=numbers)
    load_foundations_mdp(model, foundation, LegacyTable(list(numbers)), max_L=2)
    return model.eval()


def compare(model, legacy, name, numbers):
    atoms, charges, total_charge = CASES[name]
    batch = legacy_batch(legacy, atoms, charges, total_charge)
    reference = legacy(batch, training=False, compute_dielectric_derivatives=True)
    graph = v1_graph(atoms, numbers, float(legacy.r_max), charges, total_charge)
    result = engine(model)(graph, compute=("dmu_dr", "dalpha_dr"))
    assert gap(result.dipole, reference["dipole"]) < 1e-12
    for key in (
        "charges",
        "atomic_dipoles",
        "polarizability",
        "polarizability_sh",
        "dmu_dr",
        "dalpha_dr",
    ):
        assert gap(result.extras[key], reference[key]) < 1e-12, key


@pytest.fixture(name="foundation")
def fixture_foundation(fp64, tmp_path):
    legacy = legacy_dielectric("8x0e + 8x1o + 8x2e")
    return legacy, write_foundation(legacy, tmp_path)


def test_the_data_s_table_is_the_frozen_tree_s_dielectric_transfer(
    foundation, tmp_path
):
    legacy, path = foundation
    built = build(fine_tune_config(tmp_path, path, "data"))
    assert list(built.data.z_table.zs) == [1, 8]
    subset = legacy_subset(legacy, [1, 8])
    compare(built.model.get_submodule("backbone"), subset, "water", [1, 8])


def test_the_foundation_s_table_computes_what_the_foundation_does(foundation, tmp_path):
    legacy, path = foundation
    built = build(fine_tune_config(tmp_path, path, "foundation"))
    assert list(built.data.z_table.zs) == [1, 6, 8]
    for name in ("water", "methanol"):
        compare(built.model.get_submodule("backbone"), legacy, name, [1, 6, 8])


def test_the_converted_foundation_drives_the_calculator(foundation):
    """The foundation's own checkpoint through the one calculator, against the
    frozen model on the same structure."""
    legacy, path = foundation
    atoms, charges, total_charge = CASES["methanol"]
    atoms = atoms.copy()
    atoms.arrays["Qs"] = np.asarray(charges)
    atoms.info["charge"] = total_charge
    calculator = MACECalculator(model_paths=path)
    atoms.calc = calculator
    calculator.calculate(atoms)
    reference = legacy(legacy_batch(legacy, atoms, charges, total_charge))
    np.testing.assert_allclose(
        calculator.results["dipole"],
        reference["dipole"].detach().numpy()[0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calculator.results["polarizability"],
        reference["polarizability"].detach().numpy()[0],
        atol=1e-12,
    )
