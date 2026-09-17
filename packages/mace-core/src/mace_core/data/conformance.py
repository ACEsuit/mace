"""The checks every data backend has to pass, written once.

Point it at a backend and it exercises the parts of the Protocol that are easy
to get subtly wrong: the length, random access, that reading past the end
raises instead of returning something, that a sequential run agrees with the
random access, and that the whole thing survives being pickled.

The pickle check is not ceremony. A backend is sent to dataloader workers and to
other ranks, and one holding an open file handle has to drop it. That failure
shows up as a process that will not start, far from the backend.

DATA-2's shard backends and any third-party plugin get this by calling it.
"""

from __future__ import annotations

import pickle

from mace_core.data.backend import DataBackend, DataBackendError

__all__ = ["data_backend_conformance"]


def data_backend_conformance(backend: DataBackend, *, expected_length: int) -> None:
    """Run every conformance check against an opened backend.

    Args:
        backend: The opened backend.
        expected_length: How many configurations the source holds, known
            independently. Passed in rather than read from the backend, since
            checking a length against itself checks nothing.

    Raises:
        AssertionError: Naming which property failed.
    """
    assert isinstance(backend, DataBackend), (
        f"{type(backend).__name__} does not satisfy the DataBackend Protocol"
    )
    assert len(backend) == expected_length, (
        f"the backend reports {len(backend)} configurations and the source "
        f"holds {expected_length}"
    )

    first = backend[0]
    assert first.atomic_numbers.size == first.positions.shape[0], (
        "the atomic numbers and the positions disagree on how many atoms there are"
    )
    assert backend[0] is not None

    sequential = list(backend.iter_range())
    assert len(sequential) == expected_length, (
        f"iter_range yielded {len(sequential)} configurations and __len__ says "
        f"{expected_length}. A sequential fast path that disagrees with random "
        f"access is how a subset of a dataset trains silently."
    )
    for index, configuration in enumerate(sequential):
        assert configuration.atomic_numbers.tolist() == (
            backend[index].atomic_numbers.tolist()
        ), f"iter_range and __getitem__ disagree at index {index}"

    stepped = list(backend.iter_range(0, expected_length, 2))
    assert len(stepped) == len(range(0, expected_length, 2))

    try:
        backend[expected_length]
    except DataBackendError:
        pass
    else:
        raise AssertionError(
            "reading past the end returned something instead of raising. The "
            "frozen tree returns None into the batch here, which is the defect "
            "this contract exists to remove."
        )

    revived = pickle.loads(pickle.dumps(backend))
    assert len(revived) == expected_length, (
        "the backend did not survive a pickle round trip, so it cannot be sent "
        "to a dataloader worker or to another rank"
    )
    assert revived[0].atomic_numbers.tolist() == first.atomic_numbers.tolist()

    statistics = backend.statistics()
    assert statistics is None or statistics.r_max >= 0.0
    manifest = backend.metadata()
    assert manifest is None or manifest.schema_version >= 1
