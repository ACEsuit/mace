"""The in-memory backend and the contract check every backend runs."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from mace_core.data import (
    Configuration,
    DataBackend,
    DataBackendError,
    InMemoryBackend,
    KeySpecification,
)
from mace_core.data.conformance import check_data_backend


def _structure(z: int, x: float, head: str = "Default") -> Configuration:
    return Configuration(
        atomic_numbers=np.array([z, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        head=head,
    )


def _structures(n: int = 4) -> list[Configuration]:
    return [_structure(8, 0.9 + 0.1 * i) for i in range(n)]


class _CountingBackend(InMemoryBackend):
    """Counts every structure read; iteration reads through indexing."""

    reads = 0

    def __getitem__(self, index):
        _CountingBackend.reads += 1
        return super().__getitem__(index)


def test_the_in_memory_backend_keeps_the_contract():
    check_data_backend(InMemoryBackend(_structures()), expected_length=4)


def test_the_contract_check_catches_a_backend_that_returns_none():
    class Lenient(InMemoryBackend):
        def __getitem__(self, index):
            if not 0 <= index < len(self):
                return None
            return super().__getitem__(index)

    with pytest.raises(AssertionError, match="did not raise DataBackendError"):
        check_data_backend(Lenient(_structures()), expected_length=4)


class _DropsLabelsWhenIterating(InMemoryBackend):
    def iter_range(self, start=0, stop=None, step=1):
        for structure in super().iter_range(start, stop, step):
            yield dataclasses.replace(structure, properties={})


class _IgnoresStep(InMemoryBackend):
    def iter_range(self, start=0, stop=None, step=1):
        return super().iter_range(start, stop)


class _ForgetsOnPickle(InMemoryBackend):
    def __getstate__(self):
        return {**self.__dict__, "_configurations": self._configurations[:1]}


class _StopsIteratingEarly(InMemoryBackend):
    def __iter__(self):
        return self.iter_range(stop=1)


class _MiscountsLength(InMemoryBackend):
    def __len__(self):
        return super().__len__() + 1


class _NamesTheWrongIndex(InMemoryBackend):
    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise DataBackendError(self.source, 0, "out of range")
        return super().__getitem__(index)


class _NumpyEnergies(InMemoryBackend):
    def isolated_atom_energies(self):
        return {np.int64(8): -2.0}


class _NoIsolatedAtomEnergies(InMemoryBackend):
    isolated_atom_energies = None


@pytest.mark.parametrize(
    ("broken", "length", "message"),
    [
        (InMemoryBackend, 5, "yields 4 structures, not 5"),
        (_DropsLabelsWhenIterating, 4, r"backend\[0\] is not the structure"),
        (_IgnoresStep, 4, r"iter_range\(0, 4, 2\) is not that slice"),
        (_ForgetsOnPickle, 4, "after a pickle round trip"),
        (_MiscountsLength, 4, "len\\(\\) is 5, not 4"),
        (_NamesTheWrongIndex, 4, "names index 0"),
        (_NumpyEnergies, 4, "is not \\{int: float\\}"),
        (_NoIsolatedAtomEnergies, 4, "lacks part of the DataBackend protocol"),
        (InMemoryBackend, 1, "needs at least two structures"),
        (_StopsIteratingEarly, 4, "iterating the backend is not iter_range"),
    ],
)
def test_the_contract_check_catches_each_broken_part(broken, length, message):
    structures = [
        dataclasses.replace(s, properties={"energy": -1.0}) for s in _structures()
    ]
    with pytest.raises(AssertionError, match=message):
        check_data_backend(broken(structures), expected_length=length)


def test_iter_range_refuses_what_indexing_refuses():
    backend = InMemoryBackend(_structures(), source="train.xyz")
    with pytest.raises(DataBackendError, match="structure -1"):
        list(backend.iter_range(-1))
    with pytest.raises(DataBackendError, match="structure 4"):
        list(backend.iter_range(0, 6))
    with pytest.raises(DataBackendError, match="positive step, not 0"):
        list(backend.iter_range(0, None, 0))


def test_an_index_out_of_range_names_the_source_and_the_index():
    backend = InMemoryBackend(_structures(2), source="train.xyz")
    with pytest.raises(DataBackendError, match=r"structure 5 of 'train.xyz'") as error:
        backend[5]
    assert error.value.index == 5


def test_load_reads_once_and_then_answers_as_the_backend_does():
    source = _CountingBackend(
        _structures(), source="train.xyz", isolated_atom_energies={8: -2}
    )
    _CountingBackend.reads = 0
    loaded = InMemoryBackend.load(source)
    assert _CountingBackend.reads == len(source)

    assert isinstance(loaded, DataBackend)
    assert loaded.source == "train.xyz"
    check_data_backend(loaded, expected_length=4)
    assert loaded.isolated_atom_energies() == {8: -2.0}
    assert [s.positions[1, 0] for s in loaded.iter_range()] == [
        s.positions[1, 0] for s in source.iter_range(stop=len(source))
    ]
    _CountingBackend.reads = 0
    loaded[2]
    list(loaded.iter_range())
    assert _CountingBackend.reads == 0


def test_the_list_is_copied_and_the_configurations_are_not():
    structures = _structures(2)
    backend = InMemoryBackend(structures)
    structures.append(_structure(6, 1.0))
    assert len(backend) == 2
    assert backend[0] is structures[0]


def test_it_is_not_a_format():
    assert not InMemoryBackend.sniff("train.xyz")
    with pytest.raises(DataBackendError, match=r"InMemoryBackend\.load"):
        InMemoryBackend.open("train.xyz", key_spec=KeySpecification.from_defaults())
