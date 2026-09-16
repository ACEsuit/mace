"""The default file keys a labelled structure file is read with.

These thirteen names are a **data contract**, not a default anyone is free to
adjust: every labelled dataset on disk was written against them, so renaming
one does not break a build, it silently stops reading somebody's forces.

The two halves are different things and the distinction is what
:class:`~mace_core.data.keys.KeySpecification` is built on:

*convention name*
    what the rest of the stack calls a property -- ``energy``, ``forces``,
    ``magmom``. Nothing downstream of parsing ever sees anything else.

*file key*
    the string the value is stored under in the file -- ``REF_energy``,
    ``REF_forces``. Confined to the key specification and to the parser.

The member names below are the convention names, uppercased;
:meth:`DefaultKeys.keydict` derives the ``<name>_key`` spelling that the
command line exposes (``--energy_key``, ``--magforces_key``), so the CLI
surface and this table cannot drift apart.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["DefaultKeys"]


class DefaultKeys(Enum):
    """Convention name (the member) to default file key (its value)."""

    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    POLARIZABILITY = "polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    ELEC_TEMP = "elec_temp"
    MAGMOM = "REF_magmom"
    MAGFORCES = "REF_magforces"

    @classmethod
    def keydict(cls) -> dict[str, str]:
        """``{"<convention name>_key": "<default file key>"}`` for all thirteen.

        The ``_key`` suffix is the command-line spelling, so this is also the
        set of overrides a key specification accepts.
        """
        return {f"{member.name.lower()}_key": member.value for member in cls}

    @classmethod
    def convention_names(cls) -> tuple[str, ...]:
        """The thirteen convention names, in declaration order."""
        return tuple(member.name.lower() for member in cls)
