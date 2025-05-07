.. _lammps_mliap:

***********************
MACE in LAMMPS with ML-IAP
***********************

Introduction
===========

MACE models can now be used in LAMMPS through two different interfaces:

- The original interface (as described in the :ref:`lammps` documentation)
- The new ML-IAP unified interface, which offers improved performance on GPUs

This document focuses on the new ML-IAP interface, which calls the model from python, and supports both cuEquivariance acceleration and multi-GPU inference, and atomic virials.

Preparing Your Model for ML-IAP
==============================

.. warning::

   **Very Important NOTE:** This operation needs to be done on a GPU, and to be extra safe, on the same GPU architecture as the one that will be used for inference.

To use a MACE model with the ML-IAP interface, you need to convert your trained model using the ``create_lammps_model.py`` script with the ``mliap`` format option::

    python mace/cli/create_lammps_model.py your_trained_model.model --format=mliap

This will create a file named ``your_trained_model.model-mliap_lammps.pt`` that is compatible with the ML-IAP interface.


Installation Requirements
========================

Python Dependencies
-----------------

In addition to the standard MACE dependencies, you'll need:

- **cuEquivariance**: Base, torch, and torch backend from `NVIDIA cuEquivariance <https://github.com/NVIDIA/cuEquivariance>`_
- **cupy-cuda12x**: Compatible with your CUDA version
- **lammps Python package**: Generated when building LAMMPS (see below)

.. note::
    It's recommended to use ``torch<=2.5.0`` to avoid new model loading warnings. While some MACE dependencies require ``numpy<2``, cuEquivariance works with ``numpy>=2``.

LAMMPS Compilation
----------------

The ML-IAP interface requires the ``develop`` branch of LAMMPS. You need to compile LAMMPS with specific options for Kokkos, Python, and ML-IAP packages.

Required CMake Options
^^^^^^^^^^^^^^^^^^^^^

Your LAMMPS build needs these options:

- ``BUILD_MPI=ON``
- ``PKG_ML-IAP=ON``
- ``MLIAP_ENABLE_PYTHON=ON``
- ``PKG_ML-SNAP=ON``
- ``PKG_PYTHON=ON``
- ``BUILD_SHARED_LIBS=ON``
- Kokkos options appropriate for your hardware

Step-by-Step Compilation Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Clone the LAMMPS repository::

    git clone https://github.com/lammps/lammps.git
    cd lammps

2. Create a build directory::

    mkdir build-mliap
    cd build-mliap

3. Copy and customize Kokkos settings for your GPU architecture::

    cp ../cmake/presets/kokkos-cuda.cmake ./
    # Edit kokkos-cuda.cmake to set the correct architecture
    # Find your architecture in: https://docs.lammps.org/Build_extras.html#kokkos

4. Configure with CMake (activate your Python virtual environment before this step)::

    cmake -C kokkos-cuda.cmake \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$(pwd) \
      -D BUILD_MPI=ON \
      -D PKG_ML-IAP=ON \
      -D PKG_ML-SNAP=ON \
      -D MLIAP_ENABLE_PYTHON=ON \
      -D PKG_PYTHON=ON \
      -D BUILD_SHARED_LIBS=ON \
      ../cmake

5. Build LAMMPS::

    make -j 8

.. note::
    If you encounter compilation errors, you might need to remove certain CUDA compiler flags with the following command::
    
        sed -i 's/ -Xcudafe --diag_suppress=unrecognized_pragma,--diag_suppress=128//' build/CMakeFiles/lmp.dir/flags.make7
    
    Then retry the compilation.

6. Create and install the LAMMPS Python package::

    make install-python

Using MACE with ML-IAP in LAMMPS
===============================

LAMMPS Input File
---------------

Your LAMMPS input should begin with standard settings::

    units         metal
    atom_style    atomic
    atom_modify   map yes
    newton        on

Then define the ML-IAP pair style with your converted model::

    pair_style      mliap unified your_model-mliap_lammps.pt 0
    pair_coeff      * * C H O N

The ``0`` after the model filename is a required parameter for the unified ML-IAP interface.

The element list after ``pair_coeff * *`` should be ordered as you want them to appear in LAMMPS, and must be a subset of the elements your model was trained on.

Command Line Options
-----------------

When running LAMMPS with MACE/ML-IAP, use these command line options for GPU acceleration::

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in your_input.in

This enables 1 GPU with Kokkos. You can change ``g 1`` to use multiple GPUs if your system supports it.

For multi-GPU simulations with MPI, use::

    mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.in

This example uses 2 MPI processes with 2 GPUs. Adjust the number of processes (``-np``) and GPUs (``g``) based on your hardware.

Performance Considerations
========================

- The ML-IAP interface is optimized for GPU execution and offers better performance than the original MACE interface in LAMMPS.
- ML-IAP now supports both single (fp32) and double (fp64) precision calculation.
- For multi-GPU simulations, the standard Kokkos domain decomposition is used.

Limitations and Caveats
=====================

- This interface is in beta testing - please report any issues, especially discrepancies compared to standard MACE calculations.
- The plugin currently only works with Kokkos on GPU acceleration.
- The plugin uses cuEquivariance by default for symmetric contraction and channelwise operations.
- Multiple model heads are not currently supported.

Debugging and Environment Variables
================================

You can enable timing information by setting the environment variable::

    export MACE_TIME=true

This will print timing information for each calculation step.

Additional environment variables for debugging include:

- ``MACE_PROFILE=true``: Enable profiling (with MACE_PROFILE_START and MACE_PROFILE_END to set step range)
- ``MACE_ALLOW_CPU=true``: Allow CPU calculation (not recommended for performance)
- ``MACE_FORCE_CPU=true``: Force CPU calculation regardless of Kokkos settings

Example LAMMPS Script
===================

Here's a complete example LAMMPS script for using MACE with ML-IAP::

    # MACE ML-IAP example
    units         metal
    atom_style    atomic
    atom_modify   map yes
    newton        on

    # Read structure
    read_data     structure.data

    # Set up MACE potential
    pair_style    mliap unified model-mliap_lammps.pt 0
    pair_coeff    * * C H O N

    # Run settings
    timestep      0.0001
    thermo        100

    # MD run
    fix           1 all nvt temp 300 300 100
    run           1000

Run this script with::

    lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.in