from pathlib import Path
import random

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

source = Path(
    "/Users/rca/source_installs/anisoap/benchmarks/"
    "two_particle_gb/force_and_torque_benchmarks/"
    "random_rotations_gb.xyz"
)

frames = read(source, index=":")

if len(frames) < 5:
    raise RuntimeError(
        f"Need at least five configurations; found {len(frames)}"
    )

prepared = []

for index, atoms in enumerate(frames):
    results = atoms.calc.results

    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)

    copied = atoms.copy()

    # Reattach a clean SinglePointCalculator so extxyz writes the
    # reference energy and forces as calculator results.
    copied.calc = SinglePointCalculator(
        copied,
        energy=energy,
        forces=forces,
    )

    prepared.append(copied)

rng = random.Random(17)
indices = list(range(len(prepared)))
rng.shuffle(indices)

n_test = max(1, int(round(0.2 * len(prepared))))
test_indices = indices[:n_test]
train_indices = indices[n_test:]

train_frames = [prepared[i] for i in train_indices]
test_frames = [prepared[i] for i in test_indices]

write(
    "train.xyz",
    train_frames,
    format="extxyz",
    write_results=True,
)

write(
    "test.xyz",
    test_frames,
    format="extxyz",
    write_results=True,
)

print(f"Total configurations: {len(prepared)}")
print(f"Training configurations: {len(train_frames)}")
print(f"Test configurations: {len(test_frames)}")

check = read("train.xyz", index=0)
check_i = train_indices[0]
check_frame = frames[check_i]

print("\nRe-read training frame:")
print("  calculator:", type(check.calc).__name__)
print("  results:", sorted(check.calc.results))
print("  energy:", check.get_potential_energy())
print("  forces:", check.get_forces())

print("\nOriginal training frame:")
print("  calculator:", type(check_frame.calc).__name__)
print("  results:", sorted(check_frame.calc.results))
print("  energy:", check_frame.get_potential_energy())
print("  forces:", check_frame.get_forces())
print("\nData split completed successfully")
