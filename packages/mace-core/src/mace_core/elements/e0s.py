"""Isolated-atom energies, once resolved.

This module holds the *result* of resolution, not the resolution. Turning the
five-variant declaration (a literal table, a file, ``average``, ``estimated``,
a foundation model) into numbers is the configuration layer's job, and the
validators and the sidecar schema go with it. What the output layer needs is
the answer, in one shape, already in fp64.

Keeping the two apart is what stops the model from parsing E0 strings or
reading a foundation model's buffers by reflection, which is how the value
gets read in three different ways today.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["ResolvedE0s"]


@dataclass(frozen=True)
class ResolvedE0s:
    """One isolated-atom energy per head and element, in eV.

    Attributes:
        values: ``head -> atomic number -> energy``. A plain mapping of plain
            floats: fp64 is the only precision these are ever held at, and a
            float is already that. The model's compute dtype does not reach
            here.
    """

    values: Mapping[str, Mapping[int, float]]

    def to_array(
        self, heads: Sequence[str], atomic_numbers: Sequence[int]
    ) -> list[list[float]]:
        """The ``[n_heads, n_elements]`` table, in the given orders.

        A nested list rather than an array, because this package holds no
        framework and numpy is not one of its dependencies either. The caller
        materializes it as a buffer.

        Raises:
            KeyError: If a head or an element was never resolved. Falling back
                to zero here would train a model whose energies are wrong by a
                constant per atom of that element, which reads as a bad fit
                rather than as a missing number.
        """
        table = []
        for head in heads:
            if head not in self.values:
                raise KeyError(
                    f"no isolated-atom energies were resolved for head "
                    f"{head!r}. The heads that have them are "
                    f"{sorted(self.values)}."
                )
            row = self.values[head]
            missing = [z for z in atomic_numbers if z not in row]
            if missing:
                raise KeyError(
                    f"head {head!r} has no isolated-atom energy for element(s) "
                    f"{missing}. Its table covers {sorted(row)}."
                )
            table.append([float(row[z]) for z in atomic_numbers])
        return table
