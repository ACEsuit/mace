======================================
Foundation model to run NVT simulation
======================================


Here an example of using MACE-MP-0 foundation model to run NVT simulation is provided:

.. code-block:: python

    from mace.calculators import mace_mp 
    from ase import build
    from ase.md import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    macemp = mace_mp() # return the default medium ASE calculator equivalent to mace_mp(model="medium")
    #macemp = mace_mp(model="large") # return a larger model
    #macemp = mace_mp(model="https://tinyurl.com/y7uhwpje") # downlaod the model at the given url
    #macemp = mace_mp(dispersion=True) # return a model with D3 dispersion correction
    atoms = build.molecule('H2O')
    atoms.calc = macemp

    # Initialize velocities.
    T_init = 300  # Initial temperature in K
    MaxwellBoltzmannDistribution(atoms, T_init * units.kB)

    # Set up the Langevin dynamics engine for NVT ensemble.
    dyn = Langevin(atoms, 0.5 * units.fs, T_init * units.kB, 0.001)
    n_steps = 200 # Number of steps to run
    dyn.run(n_steps)
