.. _installation:

==============
Installation
==============

Requirements:
* Python >= 3.7
* [PyTorch](https://pytorch.org/) >= 1.12

(for openMM, use Python = 3.9)


pip installation
----------------

To install via `pip`, follow the steps below:

.. code-block:: bash

    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install mace-torch

For CPU or MPS (Apple Silicon) installation, use `pip install torch torchvision torchaudio` instead.

conda installation
-------------------

If you do not have CUDA pre-installed, it is **recommended** to follow the conda installation process:

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