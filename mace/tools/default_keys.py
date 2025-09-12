from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    ELECTRIC_FIELD = "REF_electric_field"
    POLARIZATION = "REF_polarization"
    BECS = "REF_becs"
    POLARIZABILITY = "REF_polarizability"
    HEAD = "head"
    CHARGES = "REF_charges"
    TOTAL_CHARGE = "total_charge"
    TOTAL_SPIN = "total_spin"
    ELEC_TEMP = "elec_temp"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict
