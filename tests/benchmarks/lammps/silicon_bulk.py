"""Write the periodic 1,024-atom silicon input for native_scaling.in."""

import numpy as np
from ase.build import bulk
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def main():
    silicon = bulk("Si", "diamond", a=5.431, cubic=True).repeat((8, 4, 4))
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
