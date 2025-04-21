from __future__ import annotations

from enum import Enum


class DefaultKeys(Enum):
    ENERGY = "REF_energy"
    FORCES = "REF_forces"
    STRESS = "REF_stress"
    VIRIALS = "REF_virials"
    DIPOLE = "dipole"
    HEAD = "head"
    CHARGES = "REF_charges"
    elec_temp = "elec_temp"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            if member is DefaultKeys.HEAD:
                key_dict["head"] = member.value
            else:
                key_name = f"{member.name.lower()}_key"
                key_dict[key_name] = member.value
        return key_dict
