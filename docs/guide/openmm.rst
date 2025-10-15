.. _openmm:

=================
OpenMM Interface
=================

MACE models can be used to run molecular dynamics through OpenMM.  A wide variety of simulations can be run in this way, and it allows for execution of the full simulation on the GPU.

Installation
~~~~~~~~~~~~

The `mace-md` package provides a frontend for creating openMM systems and running simulations with MACE.

Clone the repository

.. code-block:: console

    git clone https://github.com/jharrymoore/mace-md.git


Create the conda environment from the provided `env.yaml` file. Note we use mamba as a drop in conda replacement.

``conda install mamba -c conda-forge``

If you are installing on a headnode, override the virtual cuda package to match the compute node CUDA version.

.. code-block:: console

    export CONDA_OVERRIDE_CUDA=12.4
    mamba env create -f env.yml


- To quickly test the installation, we will use the MACE-OFF23-S model available from https://github.com/ACEsuit/mace-off/blob/main/ and the example files provided in the mace-md repo.

.. code-block:: console

    mace-md -f ejm_31.sdf --model_path MACE-OFF23_small.model --output_dir md_test  --steps 1000 --unwrap --minimiser openmm

Testing your Installation
-------------------------

Run the unit tests for mace-md with the following:

.. code-block:: console

    pytest -s mace_md/tests/test_mace-md.py


Running MD simulations
----------------------

The following examples use files from the ``examples/example_data`` folder of the mace-md repository.

Pure MD simulations
~~~~~~~~~~~~~~~~~~~

The simplest use case is where the full system is simulated with the MACE potential.  The simulation can be started from a ``.xyz`` file as follows, which will run the simulation for 1000 steps, reporting structures and run information every 100 steps

.. code-block:: console
mace-md -f molecule.xyz \
    --model_path /path/to/my-mace.model \
    --steps 1000 \
    --timestep 1.0 \
    --interval 100 \
    --output_dir ./test_output


For a full set of command line argument options, run

``mace-md -h``


Hybrid ML/MM simulations
~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to run MD simulations where only a subset of the system is treated with a MACE potential, with the rest treated using a classical potential.  This is a 'mechanical embedding' regime, in that only the intramolecular components are described by the ML potential, whilst the long-range dispersion and coulomb interactions are still described clasically

To run these simulations, there are more stringent requirements on the filetypes, since a full MM topology must also be built, requiring explicit bonds and atomtypes.  This typically means the full system should be provided as a PDB file, whilst the small molecule (or the part to be evaluated with MACE) is provided as an sdf file.

Whilst it is possible to run a plain MD trajectory like this, this setup is particularly useful for computing free energy corrections from the full MM to the ML/MM hamiltonian.  By specifying ``--run_type repex``, a replica exchange simulation will be performed, in which each intermediate state has a fractional contribution of the MM and ML components for the small molecule.  The full command to run a replica exchange job looks like this

.. code-block:: console

  mace-md -f complex.pdb --ml_mol ligand.sdf --run_type repex --replicas 8 --output_dir ./repex_output --steps 1000 --model_path /path/to/my-mace.model

This will run 1 ns (1000 x 1 ps MCMC swap attempts), writing all information required to analyse the simulation and compute free energy corrections to the output dir.



Alchemical simulations with MACE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to perform alchemical free energy calculations via `mace-md` using a pure MACE simulations as described in https://arxiv.org/abs/2405.18171. The model must have been trained according to the softcore protocol introduced in the paper, for example the `MACE-OFF23-SC` model available under the ASL https://github.com/jharrymoore/MACE-OFF23-SC.

To perform an absolute hydration free energy calculation, use the following command, and the example input file from the `mace-md` repo .

.. code-block:: console

  mace-md -f methane_solv.pdb \
  --decouple \
  --resname UNK \ # this must correspond to the resid of the small molecule to be alchemically decoupled in the pdb file
  --output_dir repex_methane \
  --steps 1000 \ # 1 ns per replica
  --replicas 16 \ # by default use uniform spacing
  --steps_per_iter 1000 \ # 1ps between attempted exchanges
  --interval 1 \ # log every repex iteration
  --pressure 1.0 \ # NPT
  --restart \
  --model_path /path/to/mace.model \


Optionally, the following arguments will log the repex calculation progress to wandb

.. code-block:: console

  --wandb \
  --wandb_project test_repex_calculations \
  --wandb_name "methane"

Make sure you have run `wandb login` to authenticate with wandb first.
