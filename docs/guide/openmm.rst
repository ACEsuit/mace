.. _openmm:

=================
OpenMM Interface
=================

MACE models can be used to run molecular dynamics through OpenMM.  A wide variety of simulations can be run in this way, and it allows for execution of the full simulation on the GPU.

Manual Installation steps
-------------------------

Create the conda environment from the provided environment file. Note we use mamba as a drop in conda replacement.

``conda install mamba -c conda-forge``

If you are installing on a headnode, override the virtual cuda package to match the compute node CUDA version.

``export CONDA_OVERRIDE_CUDA=11.8``

``mamba env create -f mace-openmm.yml``

- Install additional MACE/openMM related packages from GitHub

.. code-block:: console

  pip install git+https://github.com/choderalab/mpiplus.git
  pip install git+https://github.com/ACEsuit/mace.git
  pip install git+https://github.com/openmm/openmm-ml.git
  pip install git+https://github.com/jharrymoore/openmmtools.git@development


- To quickly test the installation, run the following command to run a short MACE MD simulation

.. code-block:: console

  wget https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model
  mace-md -f ejm_31.sdf --ml_mol ejm_31.sdf --model_path MACE-OFF23_small.model --output_dir md_test --nl torch_nl --steps 1000 --minimiser ase --dtype float64 --remove_cmm --unwrap

Testing your Installation
-------------------------

Run the unit tests for mace-md with the following:

.. code-block:: console

  pytest -s openmmtools/tests/test_mace-md.py



Running MD simulations
----------------------


The following snippets use files from the ``examples/example_data`` folder of the openmmtools repository.

Run pure MACE MD simulation with Langevin dynamics:

.. code-block:: python

    from openmmtools.openmm_torch.hybrid_md import PureSystem
    import torch

    torch.set_default_dtype(torch.float64)

    file = "ejm_31.sdf"
    model_path = "MACE_SPICE_larger.model"
    temperature = 298

    system=PureSystem(
      	ml_mol=file
        model_path=model_path
        potential="mace"
        output_dir="output_md"
        temperature=298,
        nl="torch"
    )

    system.run_mixed_md(
        steps=5000, interval=25, output_file="output.pdb", restart=False,
    )
    
Example of a MACE NPT simulation with periodic boundary conditions:

.. code-block:: python
    
    from openmmtools.openmm_torch.hybrid_md import PureSystem
    import torch

    torch.set_default_dtype(torch.float64)

    file = "waterbox.xyz"
    model_path = "MACE_SPICE_larger.model"
    temperature = 298
    pressure = 1.0

    system=PureSystem(
        file=file,
        model_path=model_path,
        potential="mace",
        temperature=temperature,
        output_dir="output_md",
		pressure = pressure
    )

    system.run_mixed_md(
        steps=10000, interval=50, output_file="output_md_water.pdb", restart=False
    )

Example of a hybrid ML-MM simulation where the small molecule is parametrised by MACE, whilst the solvent and ions are modelled with a classical FF

.. code-block:: python
    
    from openmmtools.openmm_torch.hybrid_md import HybridSystem
    import torch

    torch.set_default_dtype(torch.float64)

    file = "ejm_31.sdf"
    model_path = "MACE_SPICE_larger.model"
    temperature = 298

    system = MixedSystem(
        file=file,
        ml_mol=file,
        model_path=model_path,
        potential="mace",
        output_dir="output_hybrid",
        temperature=298,
        nl="nnpops",
        nnpify_type="resname",
        resname="UNK",
    )

    system.run_mixed_md(
        steps=10000, interval=50, output_file="output_md_mlmm.pdb", restart=False
    )


Alternatively, simulations can also be run through the mace-md interface, which exposes exactly the same functionality.

Pure MD simulations
~~~~~~~~~~~~~~~~~~~

The simplest use case is where the full system is simulated with the MACE potential.  The simulation can be started from a ``.xyz`` file as follows, which will run the simulation for 1000 steps, reporting structures and run information every 100 steps

``mace-md -f molecule.xyz --ml_mol molecule.xyz --model_path /path/to/my-mace.model --steps 1000 --timestep 1.0 --integrator langevin --interval 100 --output_dir ./test_output``


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



