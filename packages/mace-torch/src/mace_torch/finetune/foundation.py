"""What a run reads from the foundation model it starts from.

The data stage needs three things from a foundation model and no more: its
element table, which the fine-tune's model is built over; its isolated-atom
energies per head, which a head declaring ``foundation`` E0s copies; and, for
farthest-point sampling, a descriptor per structure. They are gathered here
into one object so the data stage takes one argument and never a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable

__all__ = ["Describe", "FoundationContext", "FoundationError"]

#: One descriptor row per structure, in the order given.
Describe = Callable[[Sequence[Configuration]], np.ndarray]


class FoundationError(ValueError):
    """A request the foundation model cannot answer."""


@dataclass(frozen=True)
class FoundationContext:
    """A foundation model, as the data stage sees it.

    Attributes:
        z_table: Its element table. A fine-tune's model is built over it, so a
            structure with an element outside it is refused rather than given
            an embedding row nobody trained.
        heads: Its heads, in order.
        e0s: Its isolated-atom energies, ``head -> atomic number -> eV``, as
            its checkpoint records them.
        describe: Descriptors for farthest-point sampling, or ``None`` when the
            run does not sample that way.
    """

    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    describe: Describe | None = None

    def e0_table(self, head: str | None) -> Mapping[int, float]:
        """One head's energies.

        Raises:
            FoundationError: If ``head`` names none of its heads, or is
                ``None`` while it has several. The frozen tree takes the first
                in that case and only logs which it took.
        """
        if head is None:
            if len(self.heads) != 1:
                raise FoundationError(
                    f"the foundation model has heads {list(self.heads)}, and "
                    f"copying its energies needs one named. Set `head` on the "
                    f"E0 declaration."
                )
            head = self.heads[0]
        if head not in self.e0s:
            raise FoundationError(
                f"the foundation model has no head {head!r}. Its heads are "
                f"{list(self.heads)}."
            )
        return self.e0s[head]
