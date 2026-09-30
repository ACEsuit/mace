"""The published foundation models, and what each one becomes in v1.

Every artifact the frozen tree's loaders reach is listed here once, by the
loader and the name that reach it and the URL it is published at, with the
family it converts onto. One is not converted: MACE-ANI-CC, superseded by
MACE-OFF23 for organic chemistry, is listed as dropped so that a lookup of it
finds a reason and a replacement rather than nothing.

This is the record the registry of loadable models is built from; the names
users type and the aliases that keep old ones working are the registry's.
Nothing here imports a framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["ROSTER", "RosterEntry", "roster_entry"]

Family = Literal["scale_shift", "polar", "dielectric"]

_MP = "https://github.com/ACEsuit/mace-mp/releases/download"
_FOUNDATIONS = "https://github.com/ACEsuit/mace-foundations/releases/download"
_OFF = "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23"


@dataclass(frozen=True)
class RosterEntry:
    """One published artifact.

    Attributes:
        loader: The frozen tree's loader that reaches it.
        name: The name that loader takes for it.
        url: Where it is published.
        family: What it converts onto, or ``None`` when it is not converted.
        replaced_by: For an artifact that is not converted, what to use
            instead.
        reason: For an artifact that is not converted, why.
    """

    loader: str
    name: str
    url: str
    family: Family | None
    replaced_by: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        dropped = self.family is None
        if dropped != (self.replaced_by is not None and self.reason is not None):
            raise ValueError(
                f"{self.loader}({self.name!r}) is "
                f"{'dropped' if dropped else 'converted'}; a dropped artifact "
                f"names its replacement and the reason, and a converted one "
                f"neither."
            )


def _mp(name: str, release: str, file: str, base: str = _MP) -> RosterEntry:
    return RosterEntry("mace_mp", name, f"{base}/{release}/{file}", "scale_shift")


ROSTER: tuple[RosterEntry, ...] = (
    _mp("small", "mace_mp_0", "2023-12-10-mace-128-L0_energy_epoch-249.model"),
    _mp("medium", "mace_mp_0", "2023-12-03-mace-128-L1_epoch-199.model"),
    _mp("large", "mace_mp_0", "MACE_MPtrj_2022.9.model"),
    _mp("small-0b", "mace_mp_0b", "mace_agnesi_small.model"),
    _mp("medium-0b", "mace_mp_0b", "mace_agnesi_medium.model"),
    _mp("small-0b2", "mace_mp_0b2", "mace-small-density-agnesi-stress.model"),
    _mp("medium-0b2", "mace_mp_0b2", "mace-medium-density-agnesi-stress.model"),
    _mp("large-0b2", "mace_mp_0b2", "mace-large-density-agnesi-stress.model"),
    _mp("medium-0b3", "mace_mp_0b3", "mace-mp-0b3-medium.model"),
    _mp("medium-mpa-0", "mace_mpa_0", "mace-mpa-0-medium.model"),
    _mp("small-omat-0", "mace_omat_0", "mace-omat-0-small.model"),
    _mp("medium-omat-0", "mace_omat_0", "mace-omat-0-medium.model"),
    _mp(
        "mace-matpes-pbe-0",
        "mace_matpes_0",
        "MACE-matpes-pbe-omat-ft.model",
        _FOUNDATIONS,
    ),
    _mp(
        "mace-matpes-r2scan-0",
        "mace_matpes_0",
        "MACE-matpes-r2scan-omat-ft.model",
        _FOUNDATIONS,
    ),
    _mp("mh-0", "mace_mh_1", "mace-mh-0.model", _FOUNDATIONS),
    _mp("mh-1", "mace_mh_1", "mace-mh-1.model", _FOUNDATIONS),
    *(
        RosterEntry("mace_off", size, f"{_OFF}/MACE-OFF23_{size}.model", "scale_shift")
        for size in ("small", "medium", "large")
    ),
    RosterEntry(
        "mace_omol",
        "extra_large",
        f"{_FOUNDATIONS}/mace_omol_0/MACE-omol-0-extra-large-1024.model",
        "scale_shift",
    ),
    *(
        RosterEntry(
            "mace_polar",
            f"polar-1-{size}",
            f"{_FOUNDATIONS}/mace_polar_1/MACE-POLAR-1-{size.upper()}.model",
            "polar",
        )
        for size in ("s", "m", "l")
    ),
    RosterEntry(
        "mace_mdp",
        "default",
        "https://raw.githubusercontent.com/Nilsgoe/MACE-MDP/main/models/MACE-MDP.model",
        "dielectric",
    ),
    RosterEntry(
        "mace_anicc",
        "default",
        "https://github.com/ACEsuit/mace/raw/main/mace/calculators/"
        "foundations_models/ani500k_large_CC.model",
        None,
        replaced_by="MACE-OFF23",
        reason=(
            "a 2023 organic-chemistry model superseded by MACE-OFF23, and the "
            "only artifact that was bundled inside the package"
        ),
    ),
)


def roster_entry(loader: str, name: str) -> RosterEntry:
    """The entry a loader and a name reach.

    Raises:
        KeyError: Naming the loaders and names there are.
    """
    for entry in ROSTER:
        if (entry.loader, entry.name) == (loader, name):
            return entry
    known = ", ".join(sorted(f"{entry.loader} {entry.name}" for entry in ROSTER))
    raise KeyError(f"{loader} {name} is no published model; there are {known}.")
