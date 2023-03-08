.. _lammps:

================
LAMMPS Interface
================

.. warning::
    The MACE LAMMPS interface is work in progress,
    with improvements in speed, stability, and generality ongoing.

    If you'd like to attempt using it, here are some rough instructions.

First steps if you're using CSD3
#####

This environment is specific to the Cascade Lake compute nodes. If you want to use Ice Lake, you'll need something slightly different - see the CSD3 documentation. Moreover, you may see MPI errors if you try running on a CSD3 head node; just use the compute nodes.::

    module purge
    module load rhel7/default-ccl
    module load gcc/9

First steps if you're using Archer2
#####

The default setup should be okay. But do everything in the work directory. After logging in, run the following, as recommended by their Quickstart for Developers.::

    export CC=cc export CXX=CC export FC=ftn export F77=ftn export F90=ftn

Download libtorch::

    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip
    unzip libtorch-shared-with-deps-1.13.0+cpu.zip
    rm libtorch-shared-with-deps-1.13.0+cpu.zip

If installing on Archer2 (or anywhere without MKL) run the following line.::

    sed -i 's/;caffe2::mkl//g' libtorch/share/cmake/Caffe2/Caffe2Targets.cmake

Install Lammps::

    git clone --branch mace --depth=1 https://github.com/ACEsuit/lammps
    cd lammps; mkdir build; cd build
    cmake -DCMAKE_INSTALL_PREFIX=$(pwd) \
          -DBUILD_MPI=ON \
          -DBUILD_OMP=ON \
          -DPKG_OPENMP=ON \
          -DPKG_ML-MACE=ON \
          -DCMAKE_PREFIX_PATH=$(pwd)/../../libtorch \
          ../cmake
    make -j 4
    make install

Preparing your model
#####

Train the model using the latest `main` branch. Ideally with pytorch 1.13, but I think 1.12.1 will also work. Afterwards, use a script like this to prepare a torchscript-compiled LAMMPS_MACE model::

    # serialize.py
    
    from e3nn.util import jit
    import sys
    import torch
    from mace.calculators import LAMMPS_MACE
    
    model_path = sys.argv[1]  # takes model name as command-line input
    model = torch.load(model_path)
    lammps_model = LAMMPS_MACE(model.to("cpu"))
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path+"-lammps.pt")

Lammps pair_style
#####

Something like this:::

    pair_style mace
    pair_coeff * * MACE_model.model-lammps.pt C H N O

If you are using a single MPI process with threading (recommended for small systems), use the no_domain_decomposition option for speedups:::

    # add this atom_modify command after your atom_style command
    atom_modify map yes

    # add the no_domain decomposition option to the pair_style declaration
    pair_stye mace no_domain_decomposition

With no_domain_decomposition, LAMMPS builds a periodic graph rather than treating ghost atoms as independent nodes.

Job submission
#####

Here is an example slurm script (for Cascade Lake). For now, I recommend relying mostly on threading for smaller systems. For larger systems, you'll need to experiment - multiple-node jobs will work, but I still recommend using a small number of MPI processes per node and threading for the rest. Definitely use the --exclusive option to get access to the full-node memory.::

    #!/bin/bash
    
    #SBATCH -J lammps-mace
    #SBATCH -A T2-CS125-CPU
    #SBATCH -p cclake
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --exclusive
    #SBATCH --time=08:00:00
    #SBATCH --mail-type=FAIL
    
    . /etc/profile.d/modules.sh
    module purge
    module load rhel7/default-ccl
    
    export OMP_NUM_THREADS=56
    export MKL_NUM_THREADS=56
    mpirun -np 1 ../../lammps/build/lmp -in in.lammps
