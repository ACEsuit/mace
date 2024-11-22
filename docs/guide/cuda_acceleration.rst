.. _cuda_acceleration:

==============================================
CUDA Acceleration with cuEquivariance Library
==============================================

.. warning::
    The cuEquivariance support is only accessible by installing the MACE code from source. It is currently available only for 'MACE' and 'ScaleShiftMACE' models.

The `cuEquivariance <https://github.com/NVIDIA/cuEquivariance>`_ library is a CUDA-accelerated library for equivariant neural networks developed by NVIDIA. It implements CUDA kernels to accelerate some of MACE's operations. MACE supports the use of the cuEquivariance library to accelerate training and inference. LAMMPS export is not yet supported. We have observed up to a 5x speed-up in inference and training for large models (e.g., MACE-MP large) on a single GPU.

############
Installation
############

To install the cuEquivariance library, follow the instructions on the `cuEquivariance <https://github.com/NVIDIA/cuEquivariance>`_ repository.
Make sure to install the cuda package `pip install cuequivariance-ops-torch-cu12` (or cu11 depending on your cuda version).
**NOTE:** The acceleration is only available for GPU devices with CUDA support. Make sure to use PyTorch 2.4.0 or higher.

Install the MACE code from source using the following command:

.. code-block:: bash

    git clone https://github.com/ACEsuit/mace
    pip install mace/.

#####
Usage
#####

To accelerate training, add the `--enable_cueq=True` flag to the `mace_run_train` command.

To accelerate inference, add the `--enable_cueq=True` flag to the MACE ASE calculator.  
For example, to use the MACE-MP model with cuEquivariance acceleration:

.. code-block:: python

    from mace.calculators import mace_mp
    from ase import build

    macemp = mace_mp(enable_cueq=True)  # Return ASE calculator
    atoms = build.molecule('H2O')
    descriptors_mp = macemp.get_descriptors(atoms)

Another example:

.. code-block:: python

    from mace.calculators import MACECalculator
    from ase import build

    mace_calc = MACECalculator(model_paths="mace_agnesi_small.model", enable_cueq=True)
    atoms = build.molecule('H2O')
    atoms.calc = mace_calc
