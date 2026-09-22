"""The dataset API: the Protocol, the registry, and the one statistics pass.

Two of these tests are about what the frozen tree does and v1 must not. Format
selection there is four successive guesses at the path name followed by
"attempting to load as LMDB", so a typo is not an error; and a bad item is
printed and returned as `None` into the batch. Both are asserted here to be
structurally impossible rather than merely unlikely.
"""

import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from mace_core.data import (
    AmbiguousFormatError,
    DataBackendError,
    DatasetManifest,
    DatasetStatistics,
    KeySpecification,
    UnknownBackendError,
    XYZBackend,
    compute_statistics,
    count_neighbours,
    data_backend_conformance,
    least_squares_atomic_energies,
    open_dataset,
)
from mace_core.data import registry as registry_module


@pytest.fixture
def key_spec():
    return KeySpecification.from_defaults()


@pytest.fixture
def dataset(tmp_path):
    """Three structures, two of them carrying an energy and forces."""
    path = tmp_path / "train.xyz"
    frames = []
    for index, count in enumerate((2, 3, 4)):
        atoms = Atoms(
            numbers=[1] * count,
            positions=np.arange(count * 3).reshape(count, 3) * 0.7,
            cell=np.eye(3) * 10.0,
            pbc=True,
        )
        atoms.info["REF_energy"] = -1.5 * count + 0.1 * index
        atoms.arrays["REF_forces"] = np.full((count, 3), 0.25 * (index + 1))
        frames.append(atoms)
    write(path, frames, format="extxyz")
    return path


# ---------------------------------------------------------------------------
# The backend, through the shared harness
# ---------------------------------------------------------------------------


def test_the_xyz_backend_passes_the_conformance_harness(dataset, key_spec):
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    data_backend_conformance(backend, expected_length=3)


def test_reading_past_the_end_raises_and_names_the_source(dataset, key_spec):
    """The frozen tree prints and returns None into the batch here."""
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    with pytest.raises(DataBackendError) as caught:
        backend[99]
    message = str(caught.value)
    assert "index 99" in message
    assert "3 configurations" in message


def test_opening_a_path_that_is_not_there_raises_rather_than_guessing(
    tmp_path, key_spec
):
    with pytest.raises(DataBackendError, match="no such file"):
        XYZBackend.open(tmp_path / "typo.xyz", key_spec=key_spec)


def test_sniffing_does_not_read_the_file():
    """Cheap by construction: a suffix check. A sniff that parsed would make
    `format='auto'` cost a full read of every candidate."""
    assert XYZBackend.sniff("/nowhere/at/all/train.xyz")
    assert XYZBackend.sniff("TRAIN.EXTXYZ")
    assert not XYZBackend.sniff("/nowhere/train.h5")


def test_the_backend_survives_being_sent_to_a_worker(dataset, key_spec):
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    revived = pickle.loads(pickle.dumps(backend))
    assert len(revived) == len(backend)


# ---------------------------------------------------------------------------
# Resolution, and the absence of any fall-through
# ---------------------------------------------------------------------------


@dataclass
class PretendEntry:
    name: str
    factory: object = None
    failure: BaseException | None = None

    def load(self):
        if self.failure is not None:
            raise self.failure
        return self.factory


class ClaimsEverything:
    name = "greedy"

    @classmethod
    def sniff(cls, source):
        return True

    @classmethod
    def open(cls, source, *, key_spec, head="Default"):
        raise AssertionError("should not have been reached")


@pytest.fixture
def registered(monkeypatch):
    def install(*entries):
        monkeypatch.setattr(
            registry_module, "entry_points", lambda group: list(entries)
        )

    return install


def test_an_explicit_format_nobody_registered_lists_what_there_is(
    registered, dataset, key_spec
):
    registered(PretendEntry("xyz", XYZBackend))
    with pytest.raises(UnknownBackendError) as caught:
        open_dataset(dataset, format="parquet", key_spec=key_spec)
    assert "['xyz']" in str(caught.value)


def test_auto_resolves_when_exactly_one_backend_claims_it(
    registered, dataset, key_spec
):
    registered(PretendEntry("xyz", XYZBackend))
    backend = open_dataset(dataset, key_spec=key_spec)
    assert len(backend) == 3


def test_auto_refuses_when_nothing_claims_it_instead_of_falling_through(
    registered, tmp_path, key_spec
):
    """The defect this replaces: the frozen tree ends its guesses with
    'attempting to load as LMDB', so a typo'd path is opened as a database."""
    registered(PretendEntry("xyz", XYZBackend))
    mystery = tmp_path / "train.parquet"
    mystery.write_text("")
    with pytest.raises(AmbiguousFormatError) as caught:
        open_dataset(mystery, key_spec=key_spec)
    message = str(caught.value)
    assert "no registered data backend claims" in message
    assert "no last resort" in message


def test_auto_refuses_when_two_backends_claim_it(registered, dataset, key_spec):
    """Picking the first would hide a bug in one of their sniff methods."""
    registered(
        PretendEntry("xyz", XYZBackend), PretendEntry("greedy", ClaimsEverything)
    )
    with pytest.raises(AmbiguousFormatError) as caught:
        open_dataset(dataset, key_spec=key_spec)
    assert "all claim this source" in str(caught.value)


def test_a_backend_that_cannot_import_is_recorded_and_import_still_works(registered):
    registered(PretendEntry("lmdb", failure=ImportError("no lmdb")))
    found = registry_module.available_backends()
    assert not found["lmdb"].loaded
    assert "no lmdb" in found["lmdb"].reason


def test_the_priority_order_is_declared_and_not_installation_order():
    """Otherwise the same path opens as different formats on two machines."""
    assert registry_module.SNIFF_PRIORITY[0] == "xyz"
    assert registry_module._ordered(["lmdb", "xyz", "hdf5"]) == [
        "xyz",
        "hdf5",
        "lmdb",
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_the_neighbour_count_of_a_simple_cubic_lattice_is_six_per_atom():
    """Known without any library: in a cubic lattice of spacing a, every atom
    has six neighbours at a and none closer."""
    cell = np.eye(3) * 2.0
    positions = np.zeros((1, 3))
    assert count_neighbours(positions, cell, (True, True, True), 2.1) == 6
    assert count_neighbours(positions, cell, (True, True, True), 2.9) == 18
    assert count_neighbours(positions, cell, (True, True, True), 1.9) == 0


def test_an_aperiodic_structure_counts_only_real_pairs():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert count_neighbours(positions, None, None, 1.5) == 2


def test_a_thin_cell_still_finds_its_images():
    """The image search range is derived from the reciprocal vectors rather
    than fixed, so a cell much thinner than the cutoff still works. A constant
    number of images would silently miss neighbours here."""
    cell = np.diag([0.6, 10.0, 10.0])
    positions = np.zeros((1, 3))
    assert count_neighbours(positions, cell, (True, False, False), 1.9) == 6


def test_the_statistics_are_computed_in_one_pass_over_any_backend(dataset, key_spec):
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    statistics = compute_statistics(
        backend, atomic_numbers=[1], r_max=3.0, atomic_energies={1: -1.0}
    )
    assert statistics.atomic_numbers == [1]
    assert statistics.r_max == 3.0
    assert statistics.avg_num_neighbors >= 0.0
    # Two atoms per structure at 0.25, 0.5 and 0.75 in every component.
    assert statistics.std == pytest.approx(
        float(
            np.sqrt(np.mean(np.concatenate([[0.25] * 6, [0.5] * 9, [0.75] * 12]) ** 2))
        )
    )


def test_the_energy_mean_is_taken_against_the_energies_that_were_passed_in(
    dataset, key_spec
):
    """The point of resolving them outside: the mean and the model buffer
    cannot disagree, whatever the energies' origin."""
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    first = compute_statistics(backend, [1], 3.0, {1: 0.0}, scaling="std")
    second = compute_statistics(backend, [1], 3.0, {1: -1.5}, scaling="std")
    assert first.mean != pytest.approx(second.mean)
    assert second.mean == pytest.approx(first.mean + 1.5)


def test_a_scaling_with_no_data_says_so_rather_than_returning_one(tmp_path, key_spec):
    """A spread of 1.0 returned quietly would divide the model's outputs by a
    number that means nothing."""
    path = tmp_path / "energies_only.xyz"
    atoms = Atoms(
        numbers=[1, 1], positions=[[0, 0, 0], [0.8, 0, 0]], cell=np.eye(3) * 9, pbc=True
    )
    atoms.info["REF_energy"] = -2.0
    write(path, [atoms], format="extxyz")
    backend = XYZBackend.open(path, key_spec=key_spec)
    with pytest.raises(ValueError, match="force RMS cannot be computed"):
        compute_statistics(backend, [1], 3.0, {1: -1.0}, scaling="rms_forces")


def test_an_unknown_scaling_names_the_ones_there_are(dataset, key_spec):
    backend = XYZBackend.open(dataset, key_spec=key_spec)
    # Through a variable, because an unknown scaling is the thing being tested
    # and a checker is right to reject the literal against the enumeration.
    unknown: Any = "minmax"
    with pytest.raises(ValueError, match="'rms_forces'"):
        compute_statistics(backend, [1], 3.0, {1: -1.0}, scaling=unknown)


def test_the_least_squares_fit_recovers_an_exact_reference(key_spec):
    """Built so the answer is known: every structure's energy is exactly the
    sum of its per-element references."""
    from mace_core.data.configuration import Configuration

    truth = {1: -0.5, 8: -4.0}
    configurations = []
    for hydrogens, oxygens in ((2, 1), (4, 2), (1, 3)):
        numbers = np.array([1] * hydrogens + [8] * oxygens)
        energy = hydrogens * truth[1] + oxygens * truth[8]
        configurations.append(
            Configuration(
                atomic_numbers=numbers,
                positions=np.zeros((numbers.size, 3)),
                properties={"energy": energy},
            )
        )
    fitted = least_squares_atomic_energies(configurations, [1, 8])
    assert fitted[1] == pytest.approx(truth[1])
    assert fitted[8] == pytest.approx(truth[8])


def test_the_fit_refuses_a_dataset_with_no_energies():
    from mace_core.data.configuration import Configuration

    bare = [Configuration(atomic_numbers=np.array([1]), positions=np.zeros((1, 3)))]
    with pytest.raises(ValueError, match="cannot be fitted"):
        least_squares_atomic_energies(bare, [1])


# ---------------------------------------------------------------------------
# The persisted records
# ---------------------------------------------------------------------------


def test_statistics_round_trip_as_real_json_with_real_types():
    """The frozen tree writes these as stringified Python dicts and reads them
    back with ast.literal_eval, which makes the file unreadable by anything
    that is not Python."""
    original = DatasetStatistics(
        atomic_energies={1: -0.5, 8: -4.0},
        avg_num_neighbors=12.5,
        mean=-0.25,
        std=1.75,
        atomic_numbers=[1, 8],
        r_max=5.0,
    )
    text = original.to_json()
    assert json.loads(text)["atomic_energies"] == {"1": -0.5, "8": -4.0}
    revived = DatasetStatistics.from_json(text)
    assert revived == original
    assert all(isinstance(z, int) for z in revived.atomic_energies)


def test_the_manifest_counts_its_shards_rather_than_assuming_a_group_size():
    """An exact length has to survive a final shard that is shorter, which is
    why the counts are stored per shard."""
    manifest = DatasetManifest(shard_counts=[100, 100, 37], r_max=5.0)
    assert manifest.length == 237
    revived = DatasetManifest.from_json(manifest.to_json())
    assert revived.length == 237
    assert revived.schema_version == 1


def test_the_manifest_carries_its_statistics_through_json():
    manifest = DatasetManifest(
        shard_counts=[2],
        statistics=DatasetStatistics(atomic_energies={1: -0.5}, r_max=4.0),
    )
    revived = DatasetManifest.from_json(manifest.to_json())
    assert revived.statistics is not None
    assert revived.statistics.atomic_energies == {1: -0.5}


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_the_data_layer_imports_no_framework(framework):
    probe = (
        "import sys, mace_core.data, mace_core.data.statistics\n"
        f"assert {framework!r} not in sys.modules, "
        f"'mace_core.data pulled in {framework}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
