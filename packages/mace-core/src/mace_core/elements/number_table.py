"""The ordered set of chemical elements a model was fitted for.

A model's one-hot element embedding is indexed by position in this table, not
by atomic number, so the table *is* part of the model: reorder it and every
weight in the embedding refers to a different element.

Ordering and de-duplication live in :func:`atomic_number_table_from_zs`, not in
the class. The split is deliberate and load-bearing. Building the table from a
dataset means collecting whatever elements appear and sorting them, while
loading a trained model means adopting the order already recorded in the
checkpoint, which must be taken exactly as it is. A class that sorted in its
constructor would silently reorder the second case.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = ["AtomicNumberTable", "atomic_number_table_from_zs"]


class AtomicNumberTable:
    """An ordered sequence of atomic numbers, and the index each one maps to.

    Args:
        zs: The atomic numbers (Z), in the order the model's element embedding
            expects them. Taken as given: not sorted.

    Raises:
        ValueError: if an atomic number appears twice. Two embedding indices
            would then name one element, and :meth:`z_to_index` would only
            ever return the first. Dropping the repeat would shift every later
            index, so the order is refused rather than repaired; build the
            table with :func:`atomic_number_table_from_zs` to de-duplicate.
    """

    def __init__(self, zs: Sequence[int]) -> None:
        repeated = sorted(z for z, count in Counter(zs).items() if count > 1)
        if repeated:
            raise ValueError(
                f"atomic numbers {repeated} appear more than once in "
                f"{list(zs)!r}. Each element takes exactly one embedding "
                f"index; remove the repeats, or build the table with "
                f"atomic_number_table_from_zs to sort and de-duplicate."
            )
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self) -> str:
        return f"AtomicNumberTable: {tuple(self.zs)}"

    def __repr__(self) -> str:
        return f"AtomicNumberTable({list(self.zs)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicNumberTable):
            return NotImplemented
        return list(self.zs) == list(other.zs)

    def __hash__(self) -> int:
        return hash(tuple(self.zs))

    def index_to_z(self, index: int) -> int:
        """The atomic number at ``index`` in the embedding."""
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        """The embedding index of ``atomic_number``.

        Raises:
            ValueError: if the element is absent from the table, which means
                the structure contains an element the model was not fitted for.
        """
        return self.zs.index(atomic_number)


def atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    """Build a table from any iterable of atomic numbers, sorted and unique.

    This is the dataset path: the elements arrive in whatever order the
    structures happen to list them, and the table has to be a deterministic
    function of the *set* of elements present.
    """
    return AtomicNumberTable(sorted(set(zs)))
