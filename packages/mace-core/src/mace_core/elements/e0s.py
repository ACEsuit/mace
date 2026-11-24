"""Isolated-atom energies, once resolved to numbers.

This module holds the result of resolving a head's E0 declaration, not the
resolution. Turning a declaration (a table, the isolated atoms of a training
file, a least-squares fit, a foundation model's tensor) into numbers happens
once, before the model is built. The model reads this and parses nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PositiveInt,
    model_validator,
)

from mace_core.heads import HeadName

__all__ = ["E0Source", "ResolvedE0s"]

#: Where a head's values came from: a table written in the configuration, the
#: isolated-atom structures of its training data, a least-squares fit to its
#: training energies, a foundation model's own table, or a foundation model's
#: table corrected against the head's data.
E0Source = Literal["table", "isolated_atoms", "average", "foundation", "estimated"]


class ResolvedE0s(BaseModel):
    """One isolated-atom energy per head and element, in eV.

    Attributes:
        values: ``head -> atomic number -> energy``. Plain floats, which are
            float64 whatever the model computes in.
        provenance: ``head -> where its values came from``, one entry per
            head of ``values``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: dict[HeadName, dict[PositiveInt, FiniteFloat]] = Field(min_length=1)
    provenance: dict[HeadName, E0Source]

    @model_validator(mode="after")
    def _one_provenance_per_head(self) -> ResolvedE0s:
        if set(self.provenance) != set(self.values):
            raise ValueError(
                f"provenance covers the heads {sorted(self.provenance)} and values "
                f"covers {sorted(self.values)}; every head has exactly one."
            )
        return self

    def to_array(
        self, heads: Sequence[str], atomic_numbers: Sequence[int]
    ) -> list[list[float]]:
        """The ``[n_heads, n_elements]`` table, rows and columns in the given
        orders.

        A nested list, since this package holds no array library; the caller
        makes it a tensor.

        Raises:
            KeyError: If a head or an element has no energy. A zero in its
                place would shift every energy by a constant per atom of that
                element, which trains as a bad fit rather than failing.
        """
        table = []
        for head in heads:
            if head not in self.values:
                raise KeyError(
                    f"head {head!r} has no isolated-atom energies; the heads "
                    f"that have them are {sorted(self.values)}."
                )
            row = self.values[head]
            missing = [z for z in atomic_numbers if z not in row]
            if missing:
                raise KeyError(
                    f"head {head!r} has no isolated-atom energy for the atomic "
                    f"number(s) {missing}; it covers {sorted(row)}."
                )
            table.append([row[z] for z in atomic_numbers])
        return table
