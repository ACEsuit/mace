"""The default file keys a labelled structure file is read with.

These twelve names are a **data contract**, not a default anyone is free to
adjust: every labelled dataset on disk was written against them, so renaming
one does not break a build, it silently stops reading somebody's forces.

Four things are fixed per name, and all four live here so none of them can be
stated twice:

*convention name*
    what the rest of the stack calls a property -- ``energy``, ``forces``,
    ``magmom``. Nothing downstream of parsing ever sees anything else.

*file key*
    the string the value is stored under in the file -- ``REF_energy``,
    ``REF_forces``. Confined to the key specification and to the parser.

*where it lives*
    ``"graph"`` for one value per structure, ``"atom"`` for one per atom. This
    is the same word :class:`~mace_core.data.keys.EmbeddingFeatureSpec` takes
    from a user, on purpose: a declared feature and a default key are the same
    kind of thing and were spelled two different ways.

*shape*
    the shape of one value: of the whole value for a per-structure property,
    of one atom's entry for a per-atom one, so ``forces`` is ``(3,)`` here and
    ``(n_atoms, 3)`` on a configuration. A
    :class:`~mace_core.data.configuration.Configuration` refuses a value of
    any other shape.

The member names below are the convention names, uppercased;
:meth:`DefaultKeys.keydict` derives the ``<name>_key`` spelling that the
command line exposes (``--energy_key``, ``--magforces_key``), so the CLI
surface and this table cannot drift apart.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, get_args

__all__ = ["STORAGES", "DefaultKeys", "Storage"]

#: Where a value is read from. Two places, because a structure file stores a
#: per-structure value and a per-atom array in different ones and there is no
#: third.
Storage = Literal["graph", "atom"]

#: The values of :data:`Storage`, for the checks a type checker cannot make
#: on a string that arrives at run time.
STORAGES: tuple[Storage, ...] = get_args(Storage)


class DefaultKeys(Enum):
    """Convention name (the member), default file key (its value), storage and
    the shape of one value.

    ``member.value`` is the file key, so the enum reads as the table it is.
    ``member.storage`` is ``"graph"`` or ``"atom"``. ``member.value_shape`` is
    the shape of one structure's value, or of one atom's entry for a per-atom
    property; :meth:`expected_shape` gives the shape a configuration holds.
    """

    ENERGY = ("REF_energy", "graph", ())
    FORCES = ("REF_forces", "atom", (3,))
    STRESS = ("REF_stress", "graph", (3, 3))
    VIRIALS = ("REF_virials", "graph", (3, 3))
    DIPOLE = ("dipole", "graph", (3,))
    POLARIZABILITY = ("polarizability", "graph", (3, 3))
    CHARGES = ("REF_charges", "atom", ())
    TOTAL_CHARGE = ("total_charge", "graph", ())
    TOTAL_SPIN = ("total_spin", "graph", ())
    ELEC_TEMP = ("elec_temp", "graph", ())
    MAGMOM = ("REF_magmom", "atom", (3,))
    MAGFORCES = ("REF_magforces", "atom", (3,))

    def __new__(
        cls, file_key: str, storage: Storage, value_shape: tuple[int, ...]
    ) -> DefaultKeys:
        member = object.__new__(cls)
        member._value_ = file_key
        member.storage = storage
        member.value_shape = value_shape
        return member

    storage: Storage
    value_shape: tuple[int, ...]

    @property
    def convention_name(self) -> str:
        """What the rest of the stack calls this property."""
        return self.name.lower()

    def expected_shape(self, n_atoms: int) -> tuple[int, ...]:
        """The shape this property has on a structure of ``n_atoms`` atoms."""
        if self.storage == "atom":
            return (n_atoms, *self.value_shape)
        return self.value_shape

    @classmethod
    def keydict(cls) -> dict[str, str]:
        """``{"<convention name>_key": "<default file key>"}`` for all twelve.

        The ``_key`` suffix is the command-line spelling, so this is also the
        set of overrides a key specification accepts.
        """
        return {f"{member.convention_name}_key": member.value for member in cls}

    @classmethod
    def convention_names(cls) -> tuple[str, ...]:
        """The twelve convention names, in declaration order."""
        return tuple(member.convention_name for member in cls)

    @classmethod
    def names_stored_per(cls, storage: Storage) -> frozenset[str]:
        """The convention names read from ``"graph"`` or from ``"atom"``.

        Raises:
            ValueError: for any other storage. A misspelling such as
                ``"atoms"`` would otherwise match no member and answer with an
                empty set, which reads as "nothing is stored there".
        """
        if storage not in STORAGES:
            raise ValueError(
                f"{storage!r} is not a storage. A value is stored per "
                f"{' or per '.join(repr(s) for s in STORAGES)}."
            )
        return frozenset(
            member.convention_name for member in cls if member.storage == storage
        )
