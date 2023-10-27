=================
Foundation models
=================

###########################
Pretrained MPtrj Checkpoint
###########################

We have collaborated with the Materials Project (MP) who trained universal MACE checkpoints covering 89 elements on 1.6 M bulk crystals in the `MPTrj dataset <https://figshare.com/articles/dataset/23713842>`_. These pretrained models were used for materials stability prediction in `Matbench Discovery <https://matbench-discovery.materialsproject.org>`_ and the corresponding `preprint <https://arxiv.org/abs/2308.14920>`_. For easy reuse, these checkpoints were published on `Hugging Face <https://huggingface.co/cyrusyc/mace-universal>`_.

To access the pretrained checkpoints as an ASE calculator, you can use the following code snippets:

.. code-block:: python

    from mace.calculators import mace_mp
    from ase import build

    macemp = mace_mp() # return ASE calculator
    atoms = build.molecule('H2O')
    descriptors_mp = macemp.get_descriptors(atoms)

.. code-block:: python

    from mace.calculators import mace_mp
    from ase import build
    from ase.md import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    from ase.io.trajectory import Trajectory


    macemp = mace_mp() # return ASE calculator
    atoms = build.molecule('H2O')
    atoms.calc = macemp

    # Initialize velocities.
    T_init = 300  # Initial temperature in K
    MaxwellBoltzmannDistribution(atoms, T_init * units.kB)

    # Set up the Langevin dynamics engine for NVT ensemble.
    dyn = Langevin(atoms, 0.5 * units.fs, T_init * units.kB, 0.001)
    n_steps = 2000000  # Number of steps to run
    dyn.run(n_steps)

Please cite the relevent papers if you use these checkpoints (see `mace-universal <https://huggingface.co/cyrusyc/mace-universal>`).