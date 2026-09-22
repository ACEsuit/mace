"""The raw parsed structure: one labelled atomic configuration.

This is the boundary object of the whole data layer. A format backend produces
these and graph construction consumes them, so it is the one place both sides
have to agree, and it is deliberately dumb: numpy arrays and plain Python, no
tensors, no neighbour list, no framework.

**Property keys are convention names, never file keys.** ``properties`` is
keyed by ``energy``, ``forces``, ``magmom`` and so on, including multi-level-of
-theory names such as ``pbe_energy`` or ``r2scan_forces``. The mapping from a
convention name to the string the value was stored under in the file is
resolved during parsing and lives only in
:class:`~mace_core.data.keys.KeySpecification`. Nothing downstream of the
parser ever sees a ``REF_energy``-style file key, so nothing downstream has to
be told which key spec produced it.

**The cell is the physical cell as parsed, and nothing else.** A neighbour list
over an aperiodic system needs an artificial box to bin into, and that box is
the neighbour builder's business: it is constructed there, used there, and
never written back. A configuration that carried a synthetic cell would report
a stress divided by an invented volume, with nothing in the object to say the
volume was invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["DEFAULT_CONFIG_TYPE", "DEFAULT_HEAD", "Configuration"]

#: The config type a structure gets when the file does not name one.
DEFAULT_CONFIG_TYPE = "Default"

#: The head a structure is attributed to when the caller does not name one.
DEFAULT_HEAD = "Default"


@dataclass
class Configuration:
    """One labelled structure, as parsed.

    Args:
        atomic_numbers: ``[n_atoms]`` integer atomic numbers (Z).
        positions: ``[n_atoms, 3]`` Cartesian positions, in Angstrom.
        properties: Labels and graph-level inputs, keyed by convention name.
            A declared property whose key was absent from the file is present
            here as ``None`` rather than missing, so ``None`` is what says the
            label is unavailable. See :meth:`is_labelled`.
        property_weights: Per-property weight in the loss, one entry per key in
            ``properties``. An absent label is also zeroed here, as a safety
            net for a consumer that reads only the weight, but the zero does
            **not** mean absence: a file is free to write
            ``config_forces_weight=0.0`` for a structure whose forces are
            perfectly present, and the two are the same number. Ask
            :meth:`is_labelled`.
        cell: ``[3, 3]`` lattice vectors as rows, in Angstrom, exactly as the
            file gave them. ``None`` when the format carries no cell at all;
            an all-zero matrix is what an aperiodic structure normally gets.
        pbc: ``(3,)`` of bools, one per lattice vector.
        weight: Weight of the whole structure in the loss.
        config_type: Free-form label used to group error tables, and to mark
            the isolated atoms an E0 is read from.
        head: Which head this structure trains, for multi-head fits.
    """

    atomic_numbers: np.ndarray
    positions: np.ndarray
    properties: dict[str, Any] = field(default_factory=dict)
    property_weights: dict[str, float] = field(default_factory=dict)
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    weight: float = 1.0
    config_type: str = DEFAULT_CONFIG_TYPE
    head: str = DEFAULT_HEAD

    def __len__(self) -> int:
        return len(self.atomic_numbers)

    def is_labelled(self, name: str) -> bool:
        """Whether this structure carries a value for ``name``.

        The one unambiguous answer. A zero in ``property_weights`` cannot give
        it: an absent label is zeroed, and so is a label the file deliberately
        weighted to zero, and those are different facts about the structure.

        Args:
            name: A convention name, as ``properties`` is keyed.

        Returns:
            ``False`` for a property that was never declared as well as for one
            declared and absent, since neither gives a value to train on.
        """
        return self.properties.get(name) is not None
