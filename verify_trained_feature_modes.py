from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


DATA = (
    "/Users/rca/source_installs/anisoap/"
    "benchmarks/two_particle_gb/"
    "force_and_torque_benchmarks/"
    "random_rotations_gb.xyz"
)


def energy(atoms, calculator) -> float:
    trial = atoms.copy()
    calculator.reset()
    trial.calc = calculator
    return float(trial.get_potential_energy())


atoms = read(DATA, index=0)
rotated = atoms.copy()

quaternions = rotated.arrays["quaternions"].copy()
quaternions[0] = np.array(
    [
        np.cos(np.pi / 4.0),
        np.sin(np.pi / 4.0),
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
rotated.arrays["quaternions"] = quaternions

for mode in (
    "none",
    "isotropic",
    "traceless_moi",
    "moi",
):
    paths = sorted(
        Path("checkpoints").glob(
            f"smoke_{mode}*.model"
        )
    )

    if not paths:
        print(f"{mode:16s} model not found")
        continue

    model = torch.load(
        paths[-1],
        map_location="cpu",
        weights_only=False,
    )

    assert model.rigid_feature_mode == mode

    calculator = MACECalculator(
        models=model,
        device="cpu",
        default_dtype="float64",
    )

    original_energy = energy(
        atoms,
        calculator,
    )
    rotated_energy = energy(
        rotated,
        calculator,
    )

    difference = rotated_energy - original_energy

    print(
        f"{mode:16s} "
        f"E_original={original_energy:+.12e} "
        f"E_rotated={rotated_energy:+.12e} "
        f"delta={difference:+.12e}"
    )
