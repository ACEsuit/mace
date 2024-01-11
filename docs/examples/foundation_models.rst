=================
Foundation models
=================

Currently available pretrained MACE models:

1. MACE-MP: pretrained foundation models for materials chemistry, parameterised for 89 chemical elements. 
2. MACE-OFF23: transferable organic force fields, parameterised for neutral organic molecules made up of 10 different chemical elements. 
3. MACE-ANI-CC: MACE model trained on the coupled cluster accurate ANI training set of organic molecules, parameterised for H, C, N, O elements. 

###########################
Pretrained MACE-MP-0 models
###########################

We have collaborated with the Materials Project (MP) to train a universal MACE checkpoints covering 89 elements on 1.6 M bulk crystals in the `MPTrj dataset <https://figshare.com/articles/dataset/23713842>`_.
The model are releaed on GitHub at https://github.com/ACEsuit/mace-mp.

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

A full benchmark of the MACE-MP-0 models across more than 30 applications can be found in the `paper <https://arxiv.org/abs/2103.01965>`_.

Please cite,

.. code-block:: latex

    @misc{batatia2023foundation,
        title={A foundation model for atomistic materials chemistry}, 
        author={Ilyes Batatia and Philipp Benner and Yuan Chiang and Alin M. Elena and Dávid P. Kovács and Janosh Riebesell and Xavier R. Advincula and Mark Asta and William J. Baldwin and Noam Bernstein and Arghya Bhowmik and Samuel M. Blau and Vlad Cărare and James P. Darby and Sandip De and Flaviano Della Pia and Volker L. Deringer and Rokas Elijošius and Zakariya El-Machachi and Edvin Fako and Andrea C. Ferrari and Annalena Genreith-Schriever and Janine George and Rhys E. A. Goodall and Clare P. Grey and Shuang Han and Will Handley and Hendrik H. Heenen and Kersti Hermansson and Christian Holm and Jad Jaafar and Stephan Hofmann and Konstantin S. Jakob and Hyunwook Jung and Venkat Kapil and Aaron D. Kaplan and Nima Karimitari and Namu Kroupa and Jolla Kullgren and Matthew C. Kuner and Domantas Kuryla and Guoda Liepuoniute and Johannes T. Margraf and Ioan-Bogdan Magdău and Angelos Michaelides and J. Harry Moore and Aakash A. Naik and Samuel P. Niblett and Sam Walton Norwood and Niamh O'Neill and Christoph Ortner and Kristin A. Persson and Karsten Reuter and Andrew S. Rosen and Lars L. Schaaf and Christoph Schran and Eric Sivonxay and Tamás K. Stenczel and Viktor Svahn and Christopher Sutton and Cas van der Oord and Eszter Varga-Umbrich and Tejs Vegge and Martin Vondrák and Yangshuai Wang and William C. Witt and Fabian Zills and Gábor Csányi},
        year={2023},
        eprint={2401.00096},
        archivePrefix={arXiv},
        primaryClass={physics.chem-ph}
    }

and the relevent papers if you use these checkpoints (see mace_mp docstrings for a list).

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