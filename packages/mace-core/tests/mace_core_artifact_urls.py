"""One published URL of each foundation-model family, as the frozen tree has it."""

MP_SMALL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-10-mace-128-L0_energy_epoch-249.model"
)
POLAR_S = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/"
    "MACE-POLAR-1-S.model"
)
OFF_MEDIUM = (
    "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
    "MACE-OFF23_medium.model"
)
OMOL_EXTRA_LARGE = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/"
    "MACE-omol-0-extra-large-1024.model"
)
MDP = "https://raw.githubusercontent.com/Nilsgoe/MACE-MDP/main/models/MACE-MDP.model"

#: family -> URL, for the checks that must hold for every one of them.
FAMILIES = {
    "mp": MP_SMALL,
    "polar": POLAR_S,
    "off": OFF_MEDIUM,
    "omol": OMOL_EXTRA_LARGE,
    "mdp": MDP,
}
