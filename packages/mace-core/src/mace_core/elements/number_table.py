"""The ordered set of chemical elements a model was fitted for.

A model's one-hot element embedding is indexed by position in this table, not
by atomic number, so the table *is* part of the model: reorder it and every
weight in the embedding refers to a different element.

Ordering and de-duplication live in :func:`atomic_number_table_from_zs`, not in
the class. The split is deliberate and load-bearing. Building the table from a
dataset means collecting whatever elements appear and sorting them, while
loading a trained model means adopting the table saved with its weights, which
must be taken exactly as it is. A class that sorted in its constructor would
silently move the columns under those weights. A model's table is ascending,
and :func:`model_atomic_number_table` refuses one that is not rather than
sorting it.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Iterable, Mapping, Set
from typing import Final, SupportsIndex

__all__ = [
    "MAX_ATOMIC_NUMBER",
    "AtomicNumberTable",
    "atomic_number_table_from_zs",
    "model_atomic_number_table",
]

#: The largest atomic number an element table may hold.
MAX_ATOMIC_NUMBER: Final = 118


class AtomicNumberTable:
    """An ordered sequence of atomic numbers, and the index each one maps to.

    Args:
        zs: The atomic numbers (Z), in the order the model's element embedding
            expects them. Taken as given: not sorted. Any iterable of integers
            is accepted, a numpy array read from a checkpoint included, and it
            is copied into a tuple of plain ``int``, so the table hashes the
            same for as long as it lives whatever the caller does with its own
            sequence afterwards.

    Raises:
        ValueError: if an atomic number appears twice. Two embedding indices
            would then name one element, and :meth:`z_to_index` would only
            ever return the first. Dropping the repeat would shift every later
            index, so the order is refused rather than repaired; build the
            table with :func:`atomic_number_table_from_zs` to de-duplicate.
    """

    def __init__(self, zs: Iterable[SupportsIndex]) -> None:
        # operator.index accepts numpy integers and refuses a float, which
        # int() would truncate into a different element without a word.
        atomic_numbers = tuple(operator.index(z) for z in zs)
        repeated = sorted(
            z for z, count in Counter(atomic_numbers).items() if count > 1
        )
        if repeated:
            raise ValueError(
                f"atomic numbers {repeated} appear more than once in "
                f"{list(atomic_numbers)!r}. Each element takes exactly one "
                f"embedding index; remove the repeats, or build the table with "
                f"atomic_number_table_from_zs to sort and de-duplicate."
            )
        self.zs: tuple[int, ...] = atomic_numbers

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self) -> str:
        return f"AtomicNumberTable: {self.zs}"

    def __repr__(self) -> str:
        return f"AtomicNumberTable({list(self.zs)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicNumberTable):
            return NotImplemented
        return self.zs == other.zs

    def __hash__(self) -> int:
        return hash(self.zs)

    def index_to_z(self, index: int) -> int:
        """The atomic number at ``index`` in the embedding."""
        return self.zs[index]

    def z_to_index(self, atomic_number: SupportsIndex) -> int:
        """The embedding index of ``atomic_number``.

        Raises:
            ValueError: if the element is absent from the table, which means
                the structure contains an element the model was not fitted for.
        """
        atomic_number = operator.index(atomic_number)
        try:
            return self.zs.index(atomic_number)
        except ValueError:
            raise ValueError(
                f"element {atomic_number} is not in the table {list(self.zs)!r}: "
                f"the model was not fitted for it."
            ) from None


def atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    """Build a table from any iterable of atomic numbers, sorted and unique.

    This is the dataset path: the elements arrive in whatever order the
    structures happen to list them, and the table has to be a deterministic
    function of the *set* of elements present.
    """
    return AtomicNumberTable(sorted(set(zs)))


def model_atomic_number_table(
    atomic_numbers: Iterable[SupportsIndex],
) -> AtomicNumberTable:
    """Read the element table of a model, which is strictly ascending.

    A model's table names at least one element, and every entry is an atomic
    number from 1 to :data:`MAX_ATOMIC_NUMBER`. Zero is not an element: it
    marks a padding atom, which sorts below every element and so takes the
    first column. The table is ascending so that an element set has one table
    and one column order, and a model finds an atom's column by searching the
    table rather than through a lookup built from it.

    Raises:
        ValueError: if the table is not a sequence, is empty, holds an entry
            that is not an integer from 1 to :data:`MAX_ATOMIC_NUMBER`, or is
            not strictly ascending, which includes repeating an element. A
            boolean is not an integer here, and a mapping, a set or a string
            is not a sequence of them.
    """
    if isinstance(atomic_numbers, (Mapping, Set, str, bytes)):
        raise ValueError(
            f"the element table is a sequence of atomic numbers in column "
            f"order, not a {type(atomic_numbers).__name__}."
        )
    try:
        entries = list(atomic_numbers)
    except TypeError:
        raise ValueError(
            f"the element table is a sequence of atomic numbers, not "
            f"{atomic_numbers!r}."
        ) from None
    invalid = [entry for entry in entries if not _is_index(entry)]
    if invalid:
        raise ValueError(
            f"the element table {entries!r} holds {invalid!r}, which are not "
            f"atomic numbers."
        )
    numbers = [operator.index(entry) for entry in entries]
    if not numbers:
        raise ValueError(
            "the element table is empty; a model names at least one element."
        )
    outside = [z for z in numbers if not 1 <= z <= MAX_ATOMIC_NUMBER]
    if outside:
        raise ValueError(
            f"the element table {numbers} holds {outside}, outside the atomic "
            f"numbers 1 to {MAX_ATOMIC_NUMBER}. Padding atoms carry 0, which is "
            f"never an element."
        )
    if numbers != sorted(set(numbers)):
        raise ValueError(
            f"the element table {numbers} is not strictly ascending. A model's "
            f"columns are its elements in ascending atomic number, each once: "
            f"{sorted(set(numbers))}."
        )
    return AtomicNumberTable(numbers)


def _is_index(entry: object) -> bool:
    """Whether `entry` has an exact integer value through `__index__`, as a
    numpy integer or a 0-d integer tensor read from a checkpoint does, and is
    not a boolean."""
    if isinstance(entry, bool):
        return False
    try:
        operator.index(entry)  # ty: ignore[invalid-argument-type]
    except TypeError:
        return False
    return True
