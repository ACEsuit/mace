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

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress

from mace_core.elements.default_keys import DefaultKeys

__all__ = [
    "DEFAULT_CONFIG_TYPE",
    "DEFAULT_HEAD",
    "RESERVED_PROPERTY_NAMES",
    "Configuration",
]

#: The config type a structure gets when the file does not name one.
DEFAULT_CONFIG_TYPE = "Default"

#: The head a structure is attributed to when the caller does not name one.
DEFAULT_HEAD = "Default"

#: The default convention names whose shape is known, by name.
_DEFAULT_PROPERTIES: dict[str, DefaultKeys] = {
    member.convention_name: member for member in DefaultKeys
}

#: The two structure-level tensors that may arrive as the six Voigt
#: components in ase's order (xx, yy, zz, yz, xz, xy), which is what
#: ``Atoms.get_stress`` returns. Both are symmetric, so the six determine the
#: matrix. A polarizability need not be symmetric and gets no such reading.
_VOIGT_PROPERTIES = frozenset(
    {DefaultKeys.STRESS.convention_name, DefaultKeys.VIRIALS.convention_name}
)


@dataclass(eq=False)
class Configuration:
    """One labelled structure, as parsed.

    Two configurations are equal only when they are the same object. A
    field-by-field ``==`` would compare numpy arrays, whose ``==`` is
    elementwise and cannot be read as one boolean.

    Args:
        atomic_numbers: ``[n_atoms]`` integer atomic numbers (Z).
        positions: ``[n_atoms, 3]`` Cartesian positions, in Angstrom.
        properties: Labels and graph-level inputs, keyed by convention name.
            A declared property whose key was absent from the file is present
            here as ``None`` rather than missing, so ``None`` is what says the
            label is unavailable. See :meth:`is_labelled`. A default convention
            name holds the shape its row of
            :class:`~mace_core.elements.DefaultKeys` gives, checked on
            construction, and nothing is reshaped to fit it. The one exception
            is a stress or a virial given as six Voigt components, which is
            expanded to the ``[3, 3]`` matrix. Any other name (a
            multi-level-of-theory label, an embedding feature) has no row and
            carries whatever shape it was given. No name may be a field of this
            class, :data:`RESERVED_PROPERTY_NAMES`: the head, for one, is
            :attr:`head` and nowhere else.
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
        head: Which head this structure trains, for multi-head fits. The only
            place it is stored.

    Raises:
        ValueError: if the atomic numbers, positions, cell, pbc or a default
            property has a shape other than the one stated above, or if a
            property is named like a field of this class. The message names
            the value and the shape it should have.
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

    def __post_init__(self) -> None:
        if np.ndim(self.atomic_numbers) != 1:
            raise ValueError(
                f"atomic_numbers has shape {np.shape(self.atomic_numbers)}. "
                f"Expected [n_atoms], one atomic number per atom."
            )
        n_atoms = len(self.atomic_numbers)
        _require_shape("positions", self.positions, (n_atoms, 3))
        if self.cell is not None:
            _require_shape("cell", self.cell, (3, 3))
        if self.pbc is not None and len(self.pbc) != 3:
            raise ValueError(
                f"pbc is {self.pbc!r}. Expected three flags, one per lattice vector."
            )
        reserved = sorted(RESERVED_PROPERTY_NAMES.intersection(self.properties))
        if reserved:
            raise ValueError(
                f"{reserved} cannot be property names: each is a field of "
                f"Configuration and is stored there. Rename the property."
            )
        self.properties = {
            name: _checked_property(name, value, n_atoms)
            for name, value in self.properties.items()
        }

    def __len__(self) -> int:
        return len(self.atomic_numbers)

    def is_labelled(self, name: str) -> bool:
        """Whether this structure carries a value for ``name``.

        The one unambiguous answer. A zero in ``property_weights`` cannot give
        it: an absent label is zeroed, and so is a label the file deliberately
        weighted to zero, and those are different facts about the structure.

        Args:
            name: A convention name, as ``properties`` is keyed.

        Raises:
            ValueError: if ``name`` was never declared. Every declared property
                is present in ``properties``, as ``None`` when the file lacked
                it, so a name that is missing is a misspelling or a property
                the key specification does not resolve, and answering
                ``False`` would read as "this structure is unlabelled".
        """
        if name not in self.properties:
            raise ValueError(
                f"{name!r} is not a declared property of this configuration. "
                f"The declared ones are {sorted(self.properties)}."
            )
        return self.properties[name] is not None


#: Names that cannot key ``properties``, because each is a field of
#: :class:`Configuration` and is stored there.
RESERVED_PROPERTY_NAMES: frozenset[str] = frozenset(
    configuration_field.name for configuration_field in fields(Configuration)
)


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    shape = np.shape(value)
    if shape != expected:
        raise ValueError(
            f"{name} has shape {shape}. Expected {expected}; values are not "
            f"reshaped, so write the array in that shape."
        )


def _checked_property(name: str, value: Any, n_atoms: int) -> Any:
    """``value`` if its shape is the one its convention name fixes.

    A stress or virial given as six Voigt components comes back as the
    ``[3, 3]`` matrix. A name outside the default table comes back unchecked.
    """
    member = _DEFAULT_PROPERTIES.get(name)
    if value is None or member is None:
        return value
    shape = np.shape(value)
    if name in _VOIGT_PROPERTIES:
        if shape == (6,):
            return voigt_6_to_full_3x3_stress(np.asarray(value, dtype=float))
        if shape != (3, 3):
            raise ValueError(
                f"{name} has shape {shape}. It is read as a 3x3 matrix, or as "
                f"the six Voigt components (xx, yy, zz, yz, xz, xy). A flat "
                f"list of nine is refused rather than reshaped, since nothing "
                f"says whether it was written row by row; write the matrix "
                f"itself."
            )
        return value
    expected = member.expected_shape(n_atoms)
    if shape != expected:
        per_atom = (
            f" for a structure of {n_atoms} atoms" if member.storage == "atom" else ""
        )
        raise ValueError(
            f"{name} has shape {shape}. Expected {expected}{per_atom}. A value "
            f"of any other layout is refused rather than reshaped; write the "
            f"array in that shape."
        )
    return value
