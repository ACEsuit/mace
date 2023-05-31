.. _lammps:

**************
MACE in LAMMPS
**************

.. warning::
    The LAMMPS interface for MACE is work in progress,
    with improvements in speed, stability, and generality ongoing.

    If you would still like to use it, here are
    some instructions. Please be extra cautious about
    benchmarking your LAMMPS model against, for example, the 
    equivalent ASE calculator.

Both CPU and GPU evaluation are possible, but GPU evaluation will give
MUCH better performance (at least at present).

Preparing your model
====================

Train the model using the `main` branch. Afterwards, use the `create_lammps_model.py` script to prepare a torchscript-compiled LAMMPS_MACE model::

    python [mace_dir]/scripts/create_lammps_model.py my_mace.model

Instructions for GPU
====================

Installation
------------

These instructions are for Cambridge-relevant machines and should be adapted as needed.

CSD3 Ampere Nodes
^^^^^^^^^^^^^^^^^

First steps::

    mkdir lammps-mace-gpu
    cd lammps-mace-gpu
    git clone --branch=mace --depth=1 https://github.com/ACEsuit/lammps
    wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
    mv libtorch libtorch-gpu

Request interactive node::

    sintr -t 1:0:0 -A MY-ACCOUNT-GPU -p ampere

Prepare the environment::

    module purge
    module load intel-mkl-2017.4-gcc-5.4.0-2tzpyn7
    module load rhel8/default-amp
    module load cudnn

Compile LAMMPS::

    cd lammps
    mkdir build-kokkos-cuda
    cd build-kokkos-cuda
    cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=$(pwd) \
        -D BUILD_MPI=yes \
        -D BUILD_OMP=yes \
        -D BUILD_SHARED_LIBS=yes \
        -D LAMMPS_EXCEPTIONS=yes \
        -D PKG_KOKKOS=yes \
        -D Kokkos_ARCH_AMDAVX=yes \
        -D Kokkos_ARCH_AMPERE100=yes \
        -D Kokkos_ENABLE_CUDA=yes \
        -D Kokkos_ENABLE_OPENMP=yes \
        -D Kokkos_ENABLE_DEBUG=no \
        -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=no \
        -D Kokkos_ENABLE_CUDA_UVM=no \
        -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
        -D PKG_ML-MACE=yes \
        -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch-gpu \
        ../cmake
     make -j 12

Using the model in LAMMPS
-------------------------

.. warning::
    At present, only single-GPU evaluation is supported.

Add the following after your `atom_style` command:::

    atom_modify map yes
    newton on

Your pair commands should look something like this:::

    pair_style mace no_domain_decomposition
    pair_coeff * * my_mace.model-lammps.pt C H N O

With no_domain_decomposition, LAMMPS builds a periodic graph rather than treating ghost atoms as independent nodes.

Finally, you should initiate Kokkos by calling LAMMPS with something like the following::

    lmp -k on g 1 -sf kk -in in.lammps

Instructions for CPU
====================

Installation
------------

These instructions are for Cambridge-relevant machines and should be adapted as needed.

CSD3 Cascade Lake Nodes
^^^^^^^^^^^^^^^^^^^^^^^

This environment is specific to the Cascade Lake compute nodes. If you want to use Ice Lake, you'll need something slightly different - see the CSD3 documentation. Moreover, you may see MPI errors if you try running on a CSD3 head node; just use the compute nodes.::

    module purge
    module load rhel7/default-ccl
    module load gcc/9

Download libtorch::

    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.0%2Bcpu.zip
    unzip libtorch-shared-with-deps-1.13.0+cpu.zip
    rm libtorch-shared-with-deps-1.13.0+cpu.zip

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

Using the model in LAMMPS
-------------------------

Your pair commands should look something like this:::

    pair_style mace
    pair_coeff * * my_mace.model-lammps.pt C H N O

If you are using a single MPI process with threading (recommended for small systems), use the no_domain_decomposition option for speedups:::

    # add this atom_modify command after your atom_style command
    atom_modify map yes

    # add the no_domain decomposition option to the pair_style declaration
    pair_stye mace no_domain_decomposition

With no_domain_decomposition, LAMMPS builds a periodic graph rather than treating ghost atoms as independent nodes.

Here is an example slurm script (for Cascade Lake). For now, it is best to 
rely on threading for smaller systems. For larger systems, you'll need to 
experiment - multiple-node jobs will work, but it is likely best to use 
a small number of MPI processes per node and threading for the rest.
You may want the --exclusive option to get access to the full-node memory.::

    #!/bin/bash
    
    #SBATCH -J lammps-mace
    #SBATCH -A MY-ACCOUNT-CPU
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
