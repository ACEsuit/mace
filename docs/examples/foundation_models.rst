=================
Foundation models
=================

Currently available pretrained MACE models:

1. MACE-MP: pretrained foundation models for materials chemistry, parameterised for 89 chemical elements. 
2. MACE-OFF23: transferable organic force fields, parameterised for neutral organic molecules made up of 10 different chemical elements. 
3. MACE-ANI-CC: MACE model trained on the coupled cluster accurate ANI training set of organic molecules, parameterised for H, C, N, O elements. 

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

Please cite the relevent papers if you use these checkpoints (see mace_mp docstrings for a list).

###########################
MACE-OFF23: Transferable Organic Force Fields
###########################

MACE-OFF23 are a series of three transferable organic force fields for organic chemistry. They were parameterised for 10 chemical elements: H, C, N, O, P, S, F, Cl, Br, I. It can be used to study systems of neutral molecules in gas phase liquid phase, or for organic crystals. If you use the model please cite the `preprint <https://arxiv.org/abs/2312.15211>`_. 

The models are published under the Academic Software License (`ASL <https://github.com/gabor1/ASL>`_) and can be downloaded from `here <https://github.com/ACEsuit/mace-off>`_.

The models can also be used simply as an ASE calculator:

.. code-block:: python

    from mace.calculators import mace_off
    from ase import build

    atoms = build.molecule('H2O')
    calc = mace_off(model="medium", device='cuda')
    atoms.set_calculator(calc)
    print(atoms.get_potential_energy())


###########################
MACE-ANI-CC: Coupled cluster Accurate Pretrained Model for H, C, N, O elements
###########################

If you use the model please cite the `paper <https://pubs.aip.org/aip/jcp/article/159/4/044118/2904837/Evaluation-of-the-MACE-force-field-architecture>`_. 

The model can also be used simply as an ASE calculator:

.. code-block:: python

    from mace.calculators import mace_anicc
    from ase import build

    atoms = build.molecule('H2O')
    calc = mace_anicc()
    atoms.set_calculator(calc)
    print(atoms.get_potential_energy())