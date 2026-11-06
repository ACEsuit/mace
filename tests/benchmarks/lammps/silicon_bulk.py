"""Write a periodic silicon crystal for native_scaling.in."""

import argparse

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat", nargs=3, type=int, default=(8, 4, 4), metavar=("X", "Y", "Z")
    )
    repeats = parser.parse_args().repeat
    if min(repeats) < 1:
        parser.error("all repeat counts must be positive")
    silicon = bulk("Si", "diamond", a=5.431, cubic=True).repeat(repeats)
    silicon.rattle(stdev=0.025, seed=4811)
    silicon.wrap()
    MaxwellBoltzmannDistribution(
        silicon, temperature_K=300, rng=np.random.RandomState(9043)
    )
    Stationary(silicon)
    write(
        "silicon.data",
        silicon,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        velocities=True,
    )


if __name__ == "__main__":
    main()
