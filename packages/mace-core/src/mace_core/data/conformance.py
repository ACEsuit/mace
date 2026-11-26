"""The checks every dataset backend has to pass, for any backend to call.

A format added in another distribution proves itself by calling
:func:`check_data_backend` on a small dataset it opened. What a backend does
with a corrupt source or a missing label depends on the format, so each
backend tests those on fixtures of its own; this checks the contract they all
share.
"""

from __future__ import annotations

import pickle
from dataclasses import fields
from typing import Any

import numpy as np

from mace_core.data.backend import DataBackend, DataBackendError
from mace_core.data.configuration import Configuration

__all__ = ["check_data_backend"]


def _equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left, right))


def _same(left: Configuration, right: Configuration) -> bool:
    """Every field equal, the labels and their weights included."""
    return (
        all(
            _equal(getattr(left, f.name), getattr(right, f.name))
            for f in fields(Configuration)
            if f.name not in ("properties", "property_weights")
        )
        and left.property_weights == right.property_weights
        and left.properties.keys() == right.properties.keys()
        and all(
            _equal(left.properties[k], right.properties[k]) for k in left.properties
        )
    )


def _same_sequence(left: list[Configuration], right: list[Configuration]) -> bool:
    return len(left) == len(right) and all(
        _same(a, b) for a, b in zip(left, right, strict=True)
    )


def check_data_backend(backend: DataBackend, expected_length: int) -> None:
    """Assert that ``backend`` keeps the :class:`DataBackend` contract.

    Args:
        backend: An opened backend over a small dataset of at least two
            structures, so that order and slicing are observable.
        expected_length: How many structures the source holds, known to the
            caller independently of the backend, so a structure the backend
            drops on every path is still caught.

    Raises:
        AssertionError: naming the first part of the contract it breaks.
    """
    assert isinstance(backend, DataBackend), (
        f"{type(backend).__name__} lacks part of the DataBackend protocol"
    )
    structures = list(backend.iter_range())
    assert expected_length >= 2, "check_data_backend needs at least two structures"
    assert len(structures) == expected_length, (
        f"iter_range() yields {len(structures)} structures, not {expected_length}"
    )
    assert len(backend) == expected_length, (
        f"len() is {len(backend)}, not {expected_length}"
    )
    for index, structure in enumerate(structures):
        assert _same(backend[index], structure), (
            f"backend[{index}] is not the structure iter_range() yields there"
        )
    for start, stop, step in ((1, None, 1), (0, len(structures), 2), (1, 2, 1)):
        assert _same_sequence(
            list(backend.iter_range(start, stop, step)), structures[start:stop:step]
        ), f"iter_range({start}, {stop}, {step}) is not that slice"
    for index in (len(structures), -1):
        try:
            backend[index]
        except DataBackendError as error:
            assert error.index == index, (
                f"the error for index {index} names index {error.index}"
            )
        else:
            raise AssertionError(f"backend[{index}] did not raise DataBackendError")
    energies = backend.isolated_atom_energies()
    assert all(
        isinstance(z, int) and isinstance(e, float) for z, e in energies.items()
    ), f"isolated_atom_energies() is not {{int: float}}: {energies!r}"
    assert _same_sequence(list(backend), structures), (
        "iterating the backend is not iter_range()"
    )
    restored = pickle.loads(pickle.dumps(backend))
    assert _same_sequence(list(restored.iter_range()), structures), (
        "the backend reads differently after a pickle round trip"
    )
