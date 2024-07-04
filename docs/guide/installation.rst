.. _installation:

============
Installation
============

Requirements:

- Python >= 3.7
- [PyTorch](https://pytorch.org/) >= 1.12 **(training with float64 is not supported with PyTorch 2.1 but is supported with 2.2 and later.)**.

(for openMM, use Python = 3.9)

pypi installation
----------------

This is the **recommended** way to install MACE. 

**First, make sure to install PyTorch.** Please refer to the [official PyTorch installation](https://pytorch.org/get-started/locally/) for the installation instructions. Select the appropriate options for your system. For GPU installation, make sure to select pip + the appropriate CUDA version for your system. For recent GPUs, the latest cuda version is usually the best choice.

To install via `pip`, follow the steps below:


.. code-block:: bash

    pip install --upgrade pip
    pip install mace-torch


For CPU or MPS (Apple Silicon) installation, use `pip install torch torchvision torchaudio` instead.

conda installation
-------------------

To install via `conda` (not recommended), follow the steps below:

.. code-block:: bash

    # Create a virtual environment and activate it
    conda create mace_env
    conda activate mace_env

    # Install PyTorch
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

    # (optional) Install MACE's dependencies from Conda as well
    conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn

    # Clone and install MACE (and all required packages)
    git clone https://github.com/ACEsuit/mace.git 
    pip install ./mace


pip installation from source
----------------------------

To install via `pip`, follow the steps below:

.. code-block:: bash

    # Create a virtual environment and activate it
    python -m venv mace-venv
    source mace-venv/bin/activate

    # Install PyTorch (for example, for CUDA 11.6 [cu116])
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

    # Clone and install MACE (and all required packages)
    git clone https://github.com/ACEsuit/mace.git
    pip install ./mace

**Note:** The homonymous package on `PyPI <https://pypi.org/project/MACE/>`_ has nothing to do with this one.