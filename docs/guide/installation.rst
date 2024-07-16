.. _installation:

============
Installation
============

Requirements:

- Python >= 3.7 (for openMM, use Python = 3.9)
- `Pytorch <https://pytorch.org/get-started/locally/>`_ >= 1.12 **(training with float64 is not supported with PyTorch 2.1 but is supported with 2.2 and later.)**.

**Make sure to install PyTorch.** Please refer to the `official PyTorch installation <https://pytorch.org/get-started/locally/>`_ for the installation instructions. Select the appropriate options for your system.

pypi installation
----------------

This is the recommended way to install MACE. 

.. code-block:: bash

    pip install --upgrade pip
    pip install mace-torch

**Note:** The homonymous package on `PyPI <https://pypi.org/project/MACE/>`_ has nothing to do with this one.

pip installation from source
----------------------------

To install via `pip`, follow the steps below:

.. code-block:: bash

    git clone https://github.com/ACEsuit/mace.git
    pip install ./mace

