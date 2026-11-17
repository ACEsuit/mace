"""Dataset statistics: which structures each quantity reads, and the counts.

The values against the frozen legacy functions are pinned live in
``tests/parity/test_statistics_parity.py``. The packages job installs no legacy
stack, so the formulas are also pinned here against hand-computed values,
beside the rules legacy has no counterpart for.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from mace_core.data import (
    AtomicNumberTable,
    Configuration,
    DataBackendError,
    DatasetStatistics,
    InMemoryBackend,
    compute_statistics,
    fit_atomic_energies,
    joint_avg_num_neighbors,
)
from mace_core.neighbors import (
    AllPairsBackend,
    available_neighbor_backends,
    get_neighbor_backend,
)
from pydantic import ValidationError

E0 = {1: -0.5, 8: -2.0}
CUTOFF = 1.5


def _structure(
    energy: float | None,
    *,
    weight: float = 1.0,
    head: str = "Default",
    forces: float | None = 0.3,
) -> Configuration:
    """Water: three atoms, two O-H edges each way within the cutoff."""
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        properties={
            "energy": energy,
            "forces": None if forces is None else np.full((3, 3), forces),
            "dipole": None,
        },
        property_weights={"energy": weight, "forces": 1.0, "dipole": 0.0},
        head=head,
    )


def _molecule(numbers: list[int], energy: float | None) -> Configuration:
    """Atoms 3 A apart along x, so no two are neighbours: for the E0 fit."""
    return Configuration(
        atomic_numbers=np.array(numbers),
        positions=np.array([[3.0 * i, 0.0, 0.0] for i in range(len(numbers))]),
        properties={"energy": energy},
    )


def _statistics(structures, indices=None, **options) -> DatasetStatistics:
    options = {"neighbor_backend": AllPairsBackend(), **options}
    return compute_statistics(
        InMemoryBackend(structures),
        indices,
        cutoff=CUTOFF,
        atomic_energies=E0,
        **options,
    )


def _interaction_per_atom(energy: float) -> float:
    return (energy - (E0[8] + 2 * E0[1])) / 3


# The quantities, and which structures they read.


def test_the_energy_is_measured_per_atom_against_the_given_references():
    statistics = _statistics([_structure(-3.5), _structure(-3.8)])
    expected = [_interaction_per_atom(-3.5), _interaction_per_atom(-3.8)]
    assert statistics.energy_mean == pytest.approx(np.mean(expected))
    assert statistics.energy_std == pytest.approx(np.std(expected, ddof=1))
    assert statistics.rms_forces == pytest.approx(0.3)
    assert statistics.rms_dipoles is None


def test_a_structure_without_a_label_is_left_out_of_that_quantity_only():
    labelled = [_structure(-3.5), _structure(-3.8)]
    with_gap = [*labelled, _structure(None, forces=0.6)]
    assert _statistics(with_gap).energy_mean == _statistics(labelled).energy_mean
    # The unlabelled energy leaves the forces of that structure in.
    assert _statistics(with_gap).rms_forces == pytest.approx(
        np.sqrt((2 * 0.3**2 + 0.6**2) / 3)
    )


def test_a_present_label_weighted_to_zero_still_counts():
    weighted = [_structure(-3.5), _structure(-3.8, weight=0.0)]
    assert _statistics(weighted).energy_mean == pytest.approx(
        np.mean([_interaction_per_atom(-3.5), _interaction_per_atom(-3.8)])
    )


def test_the_spread_needs_two_structures():
    assert _statistics([_structure(-3.5)]).energy_std is None


@pytest.mark.parametrize("label", ["energy", "forces"])
def test_a_label_that_is_not_finite_is_refused_naming_the_structure(label):
    broken = _structure(-3.8)
    broken.properties[label] = np.nan * np.asarray(broken.properties[label])
    with pytest.raises(DataBackendError, match=f"structure 1 .*its {label} is not"):
        _statistics([_structure(-3.5), broken])


def test_an_element_without_a_reference_energy_is_refused_naming_the_structure():
    with pytest.raises(DataBackendError, match=r"structure 0 .*elements \[1\]"):
        compute_statistics(
            InMemoryBackend([_structure(-3.5)]),
            cutoff=CUTOFF,
            atomic_energies={8: -2.0},
            neighbor_backend=AllPairsBackend(),
        )


def test_an_energy_on_an_empty_structure_is_refused():
    empty = Configuration(
        atomic_numbers=np.zeros(0, dtype=int),
        positions=np.zeros((0, 3)),
        properties={"energy": 0.0},
    )
    with pytest.raises(DataBackendError, match="an energy and no atoms"):
        _statistics([empty])
    with pytest.raises(DataBackendError, match="an energy and no atoms"):
        fit_atomic_energies(
            InMemoryBackend([_molecule([8, 1], -2.9), empty]),
            atomic_numbers=AtomicNumberTable([1, 8]),
        )


def test_an_isolated_atom_energy_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match=r"elements \[1\] are not finite"):
        compute_statistics(
            InMemoryBackend([_structure(-3.5)]),
            cutoff=CUTOFF,
            atomic_energies={1: np.nan, 8: -2.0},
            neighbor_backend=AllPairsBackend(),
        )


def test_one_call_reads_one_head():
    mixed = InMemoryBackend([_structure(-3.5, head="a"), _structure(-3.8, head="b")])
    with pytest.raises(DataBackendError, match=r"structure 1 .*head is 'b'"):
        compute_statistics(
            mixed, cutoff=CUTOFF, atomic_energies=E0, neighbor_backend=AllPairsBackend()
        )
    with pytest.raises(DataBackendError, match="head is 'b'"):
        fit_atomic_energies(mixed, atomic_numbers=AtomicNumberTable([1, 8]))


def test_only_the_given_indices_are_read():
    class Counting(InMemoryBackend):
        read: ClassVar[list[int]] = []

        def __getitem__(self, index):
            Counting.read.append(index)
            return super().__getitem__(index)

        def iter_range(self, start=0, stop=None, step=1):
            raise AssertionError("statistics over indices iterate the backend")

    structures = [
        _molecule([8, 1], -2.9),
        _molecule([8, 8], -99.0),
        _molecule([8, 1, 1], -3.4),
    ]
    backend = Counting(structures)
    statistics = compute_statistics(
        backend,
        [0, 2],
        cutoff=CUTOFF,
        atomic_energies=E0,
        neighbor_backend=AllPairsBackend(),
    )
    assert Counting.read == [0, 2]
    assert statistics == _statistics([structures[0], structures[2]])

    Counting.read.clear()
    fitted = fit_atomic_energies(
        backend, [0, 2], atomic_numbers=AtomicNumberTable([1, 8])
    )
    assert Counting.read == [0, 2]
    assert fitted == fit_atomic_energies(
        InMemoryBackend([structures[0], structures[2]]),
        atomic_numbers=AtomicNumberTable([1, 8]),
    )


# Neighbour counts.


def test_the_neighbour_count_skips_atoms_without_a_neighbour():
    lonely = Configuration(
        atomic_numbers=np.array([1]), positions=np.zeros((1, 3)), properties={}
    )
    statistics = _statistics([_structure(-3.5), lonely])
    assert (statistics.n_edges, statistics.n_receivers) == (4, 3)
    assert statistics.avg_num_neighbors == pytest.approx(4 / 3)


def test_neighbours_are_searched_with_the_given_backend_and_no_other(monkeypatch):
    import mace_core.data.statistics as statistics_module

    searched_with = []
    search = statistics_module.get_neighborhood

    def recording(*arguments, backend, **options):
        searched_with.append(backend)
        return search(*arguments, backend=backend, **options)

    monkeypatch.setattr(statistics_module, "get_neighborhood", recording)
    given = AllPairsBackend()
    _statistics([_structure(-3.5), _structure(-3.8)], neighbor_backend=given)
    assert searched_with == [given, given]


@pytest.mark.parametrize(
    "name", [b.name for b in available_neighbor_backends() if b.loaded]
)
def test_every_neighbour_backend_gives_the_same_counts(name):
    """On a periodic crystal too, where the backends' image searches differ."""
    crystal = bulk("Cu", "fcc", a=3.6)
    periodic = Configuration(
        atomic_numbers=crystal.numbers,
        positions=crystal.positions,
        cell=np.array(crystal.cell),
        pbc=(True, True, True),
    )
    structures = [_structure(-3.5), periodic]

    def counts(backend):
        statistics = compute_statistics(
            InMemoryBackend(structures),
            cutoff=3.0,
            atomic_energies=E0,
            neighbor_backend=backend,
        )
        return statistics.n_edges, statistics.n_receivers

    assert counts(get_neighbor_backend(name)) == counts(AllPairsBackend())


def test_heads_combine_by_their_counts_not_their_averages():
    first = DatasetStatistics(cutoff=5.0, atomic_energies={}, n_edges=10, n_receivers=2)
    second = DatasetStatistics(
        cutoff=5.0, atomic_energies={}, n_edges=30, n_receivers=10
    )
    assert joint_avg_num_neighbors([first, second]) == pytest.approx(40 / 12)
    with pytest.raises(ValueError, match="cannot be combined"):
        joint_avg_num_neighbors([first, first.model_copy(update={"cutoff": 4.0})])
    with pytest.raises(ValueError, match="no statistics to combine"):
        joint_avg_num_neighbors([])


def test_no_neighbour_at_all_has_no_average():
    empty = DatasetStatistics(cutoff=5.0, atomic_energies={}, n_edges=0, n_receivers=0)
    with pytest.raises(ValueError, match="no atom has a neighbour"):
        _ = empty.avg_num_neighbors


# The record.


def test_the_statistics_round_trip_through_json_with_atomic_numbers_as_ints():
    statistics = _statistics([_structure(-3.5), _structure(-3.8)])
    restored = DatasetStatistics.model_validate_json(statistics.model_dump_json())
    assert restored == statistics
    assert set(restored.atomic_energies) == {1, 8}


@pytest.mark.parametrize(
    "record",
    [
        '{"cutoff": 5, "atomic_energies": {}, "n_edges": 3, "n_receivers": 9}',
        '{"cutoff": 5, "atomic_energies": {}, "n_edges": 0, "n_receivers": 0,'
        ' "rms_forces": -1}',
        '{"cutoff": 5, "atomic_energies": {"0": 1.0}, "n_edges": 0, "n_receivers": 0}',
        '{"cutoff": 5, "atomic_energies": {"8": NaN}, "n_edges": 0, "n_receivers": 0}',
        '{"cutoff": Infinity, "atomic_energies": {}, "n_edges": 0, "n_receivers": 0}',
    ],
    ids=[
        "more-receivers-than-edges",
        "negative-rms",
        "atomic-number-zero",
        "nan-energy",
        "infinite-cutoff",
    ],
)
def test_a_record_that_could_not_have_been_measured_is_refused(record):
    with pytest.raises(ValidationError):
        DatasetStatistics.model_validate_json(record)


# The least-squares E0 fit.


def test_the_fit_recovers_the_energies_that_made_the_data():
    references = {1: -0.4, 8: -2.2}
    structures = [
        _molecule(numbers, sum(references[z] for z in numbers))
        for numbers in ([8, 1, 1], [8, 8, 1, 1], [8, 1])
    ]
    unlabelled = _molecule([1, 1], None)
    fit = fit_atomic_energies(
        InMemoryBackend([*structures, unlabelled]),
        atomic_numbers=AtomicNumberTable([1, 8]),
    )
    assert fit == pytest.approx(references)


@pytest.mark.parametrize(
    ("structures", "table", "message"),
    [
        (
            [_molecule([8, 1], -2.9)],
            [8],
            r"structure 0 .*elements \[1\] outside the table",
        ),
        ([_molecule([8], None)], [1, 8], r"holds elements \[1, 8\]"),
    ],
    ids=["element-outside-the-table", "nothing-labelled"],
)
def test_the_fit_refuses_what_it_cannot_fit(structures, table, message):
    with pytest.raises(ValueError, match=message):
        fit_atomic_energies(
            InMemoryBackend(structures), atomic_numbers=AtomicNumberTable(table)
        )


def test_one_composition_gives_the_minimum_norm_fit_and_says_so(caplog):
    """As the frozen tree: right for that composition, arbitrary per element."""
    structures = [_molecule([8, 1], -2.9), _molecule([8, 8, 1, 1], -5.8)]
    with caplog.at_level("WARNING"):
        fit = fit_atomic_energies(
            InMemoryBackend(structures), atomic_numbers=AtomicNumberTable([1, 8])
        )
    assert fit[1] + fit[8] == pytest.approx(-2.9)
    assert fit[1] == pytest.approx(fit[8])
    assert "determine only 1 of the 2" in caplog.text
