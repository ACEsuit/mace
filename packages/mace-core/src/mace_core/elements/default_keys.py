"""The default file keys a labelled structure file is read with.

These thirteen names are a **data contract**, not a default anyone is free to
adjust: every labelled dataset on disk was written against them, so renaming
one does not break a build, it silently stops reading somebody's forces.

Three things are fixed per name, and all three live here so none of them can be
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

The member names below are the convention names, uppercased;
:meth:`DefaultKeys.keydict` derives the ``<name>_key`` spelling that the
command line exposes (``--energy_key``, ``--magforces_key``), so the CLI
surface and this table cannot drift apart.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

__all__ = ["DefaultKeys", "Storage"]

#: Where a value is read from. Two places, because a structure file stores a
#: per-structure value and a per-atom array in different ones and there is no
#: third.
Storage = Literal["graph", "atom"]


class DefaultKeys(Enum):
    """Convention name (the member), default file key (its value), and storage.

    ``member.value`` stays the file key, so the enum still reads as the table it
    was. ``member.storage`` is the half that used to be written out again as two
    frozensets in :mod:`mace_core.data.keys`.
    """

    ENERGY = ("REF_energy", "graph")
    FORCES = ("REF_forces", "atom")
    STRESS = ("REF_stress", "graph")
    VIRIALS = ("REF_virials", "graph")
    DIPOLE = ("dipole", "graph")
    POLARIZABILITY = ("polarizability", "graph")
    HEAD = ("head", "graph")
    CHARGES = ("REF_charges", "atom")
    TOTAL_CHARGE = ("total_charge", "graph")
    TOTAL_SPIN = ("total_spin", "graph")
    ELEC_TEMP = ("elec_temp", "graph")
    MAGMOM = ("REF_magmom", "atom")
    MAGFORCES = ("REF_magforces", "atom")

    def __new__(cls, file_key: str, storage: Storage) -> DefaultKeys:
        member = object.__new__(cls)
        member._value_ = file_key
        member.storage = storage
        return member

    storage: Storage

    @property
    def convention_name(self) -> str:
        """What the rest of the stack calls this property."""
        return self.name.lower()

    @classmethod
    def keydict(cls) -> dict[str, str]:
        """``{"<convention name>_key": "<default file key>"}`` for all thirteen.

        The ``_key`` suffix is the command-line spelling, so this is also the
        set of overrides a key specification accepts.
        """
        return {f"{member.convention_name}_key": member.value for member in cls}

    @classmethod
    def convention_names(cls) -> tuple[str, ...]:
        """The thirteen convention names, in declaration order."""
        return tuple(member.convention_name for member in cls)

    @classmethod
    def names_stored_per(cls, storage: Storage) -> frozenset[str]:
        """The convention names read from ``"graph"`` or from ``"atom"``."""
        return frozenset(
            member.convention_name for member in cls if member.storage == storage
        )
