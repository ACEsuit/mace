.. _cuda_acceleration:

==============================================
CUDA acceleration with cuequivariance library
==============================================

.. warning::
    The cuequivariance support is only accessible by installing the mace code from source.

The `cuequivariance <https://github.com/NVIDIA/cuEquivariance>` library is a CUDA-accelerated library for equivariant neural networks developed by NVIDIA.
It implemens CUDA kernels to accelerate some of MACE operations. MACE supports the use of cuequivariance library to accelerate training and inference. LAMMPS export is not supported yet.
We have observed up to x5 times speed up in inference and trainig for large models (e.g. MACE-MP large) on a single GPU.

## Installation

To install the cuequivariance library, you can follow the instructions on the `cuequivariance <https://github.com/NVIDIA/cuEquivariance>`.
**NOTE** The acceleration is only for GPU devices with CUDA support. Make sure to use Pytorch 2.4.0 or higher.

Install the MACE code from source with the following command:

.. code-block:: bash

    git clone https://github.com/ACEsuit/mace
    pip install mace/.
    
## Usage

To use to accelerate training, add the `--enable_cueq=True` flag to the `mace_run_train` command.

To use to accelerate inference, add the `--enable_cueq=True` flag to the MACE ase calculator.
For example to use the MACE-MP model with cuequivariance acceleration:

.. code-block:: python

    from mace.calculators import mace_mp
    from ase import build

    macemp = mace_mp(enable_cueq=True) # return ASE calculator
    atoms = build.molecule('H2O')
    descriptors_mp = macemp.get_descriptors(atoms)

.. code-block:: python
    from mace.calculators import MACECalculator
    from ase import build

    mace_calc = MACECalculator(model_paths="mace_agnesi_small.model", enable_cueq=True)
    atoms = build.molecule('H2O')
    atoms.calc = mace_calc