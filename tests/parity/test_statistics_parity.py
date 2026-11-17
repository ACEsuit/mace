"""Dataset statistics against the frozen legacy functions, on the training fixture.

These values seed a model's scale, shift and neighbour normalisation, so a
silent deviation moves every model trained on v1. The fixture is labelled
throughout: on a dataset with missing labels v1 measures over the labelled
structures and legacy averages zero-filled values in, a recorded difference.
Both sides compute in float64 and differ only in the order of the arithmetic.
"""

import numpy as np
import pytest
from mace_core.data import (
    AtomicNumberTable,
    InMemoryBackend,
    KeySpecification,
    compute_statistics,
    configuration_from_atoms,
    fit_atomic_energies,
    joint_avg_num_neighbors,
)
from mace_core.neighbors import get_neighbor_backend

from mace import data as legacy_data
from mace.data.utils import compute_average_E0s
from mace.modules.utils import (
    compute_avg_num_neighbors,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
)
from mace.tools import AtomicNumberTable as LegacyAtomicNumberTable
from mace.tools import torch_geometric
from tests.golden.anchors import load_training_structures
from tests.golden.harness import tolerance

ORDER_ONLY = tolerance("closed_form_fp64")
CUTOFF = 4.0


@pytest.fixture(name="structures")
def fixture_structures():
    """The training fixture, with a dipole label added so its RMS is measured."""
    structures = load_training_structures()
    rng = np.random.default_rng(0)
    for atoms in structures:
        atoms.info["dipole"] = rng.normal(size=3)
    return structures


def _legacy_configurations(structures):
    keys = legacy_data.KeySpecification.from_defaults()
    return [legacy_data.config_from_atoms(atoms, keys) for atoms in structures]


def _legacy_loader(configurations, z_table, heads=None):
    graphs = [
        legacy_data.AtomicData.from_config(
            configuration, z_table=z_table, cutoff=CUTOFF, heads=heads
        )
        for configuration in configurations
    ]
    return torch_geometric.dataloader.DataLoader(graphs, batch_size=4, shuffle=False)


def _backend(structures, head="Default"):
    keys = KeySpecification.from_defaults()
    return InMemoryBackend(
        [configuration_from_atoms(a, keys, head_name=head) for a in structures]
    )


def _close(value, reference):
    assert value == pytest.approx(
        float(np.atleast_1d(reference)[0]), abs=ORDER_ONLY.atol, rel=ORDER_ONLY.rtol
    )


@pytest.mark.parametrize("compositions", ["one", "varied"])
def test_the_fitted_energies_are_the_legacy_ones(structures, fp64, compositions):
    """Every fixture structure is CH2O, which determines one energy of three:
    both fits are then the minimum-norm one. Removing an atom from some of
    them gives the fit full rank."""
    if compositions == "varied":
        for index, atoms in enumerate(structures):
            if index % 4:
                del atoms[index % 4]
    zs = sorted({int(z) for atoms in structures for z in atoms.numbers})
    legacy = compute_average_E0s(
        _legacy_configurations(structures), LegacyAtomicNumberTable(zs)
    )
    fitted = fit_atomic_energies(
        _backend(structures), atomic_numbers=AtomicNumberTable(zs)
    )
    assert fitted.keys() == legacy.keys()
    for z, energy in fitted.items():
        _close(energy, legacy[z])


def test_the_statistics_are_the_legacy_ones(structures, fp64):
    zs = sorted({int(z) for atoms in structures for z in atoms.numbers})
    e0s = dict(zip(zs, (-0.5, -1.2, -2.0), strict=True))
    loader = _legacy_loader(
        _legacy_configurations(structures), LegacyAtomicNumberTable(zs)
    )
    e0_array = np.array([e0s[z] for z in zs])
    legacy_mean, legacy_std = compute_mean_std_atomic_inter_energy(loader, e0_array)
    _, legacy_rms_forces = compute_mean_rms_energy_forces(loader, e0_array)

    statistics = compute_statistics(
        _backend(structures),
        cutoff=CUTOFF,
        atomic_energies=e0s,
        neighbor_backend=get_neighbor_backend("matscipy"),
    )
    _close(statistics.energy_mean, legacy_mean)
    # Legacy `scatter_std` divides by `n - 1 + 1e-6`, a guard v1 does not
    # carry: it biases the spread by about 1e-6 / (n - 1), relative. Removed
    # here so the comparison is about the formula, not the guard.
    n = len(structures)
    _close(statistics.energy_std, legacy_std * np.sqrt((n - 1 + 1e-6) / (n - 1)))
    _close(statistics.rms_forces, legacy_rms_forces)
    _close(statistics.rms_dipoles, compute_rms_dipoles(loader))
    _close(statistics.avg_num_neighbors, compute_avg_num_neighbors(loader))


def test_the_heads_combine_into_the_legacy_joint_neighbour_count(structures, fp64):
    """Legacy measures one neighbour count over every head's structures."""
    zs = sorted({int(z) for atoms in structures for z in atoms.numbers})
    # Uneven on purpose: with equal heads the two averages coincide.
    split = len(structures) // 3
    per_head = [
        compute_statistics(
            _backend(part, head),
            cutoff=CUTOFF,
            atomic_energies=dict.fromkeys(zs, 0.0),
            neighbor_backend=get_neighbor_backend("matscipy"),
        )
        for head, part in (("a", structures[:split]), ("b", structures[split:]))
    ]
    loader = _legacy_loader(
        _legacy_configurations(structures), LegacyAtomicNumberTable(zs)
    )
    _close(joint_avg_num_neighbors(per_head), compute_avg_num_neighbors(loader))
    assert joint_avg_num_neighbors(per_head) != pytest.approx(
        np.mean([s.avg_num_neighbors for s in per_head]), abs=1e-6
    ), "averaging the heads' averages would also be plausible; it is not legacy"
