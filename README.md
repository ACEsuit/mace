<span style="font-size:larger;">MACE</span>
========
[![GitHub release](https://img.shields.io/github/release/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/releases/)
[![Paper](https://img.shields.io/badge/Paper-NeurIPs2022-blue)](https://openreview.net/forum?id=YPpSngE-ZU)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/ACEsuit/mace.svg)](https://GitHub.com/ACEsuit/mace/issues/)
[![Documentation Status](https://readthedocs.org/projects/mace/badge/)](https://mace-docs.readthedocs.io/en/latest/)

# Table of contents
- [About MACE](#about-mace)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [Tutorial](#tutorial)
- [Weights and Biases](#weights-and-biases-for-experiment-tracking)
- [Development](#development)
- [References](#references)
- [Contact](#contact)
- [License](#license)


##  About MACE
MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing.

This repository contains the MACE reference implementation developed by
Ilyes Batatia, Gregor Simm, and David Kovacs.

Also available: 
* [MACE in JAX](https://github.com/ACEsuit/mace-jax), currently about 2x times faster at evaluation, but training is recommended in Pytorch for optimal performances.
* [MACE layers](https://github.com/ACEsuit/mace-layer) for constructing higher order equivariant graph neural networks for arbitrary 3D point clouds.

## Documentation

A partial documentation is available at: https://mace-docs.readthedocs.io/en/latest/

## Installation

Requirements:
* Python >= 3.7
* [PyTorch](https://pytorch.org/) >= 1.12

(for openMM, use Python = 3.9)

### conda installation

If you do not have CUDA pre-installed, it is **recommended** to follow the conda installation process:
```sh
# Create a virtual environment and activate it
conda create --name mace_env
conda activate mace_env

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# (optional) Install MACE's dependencies from Conda as well
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn

# Clone and install MACE (and all required packages)
git clone git@github.com:ACEsuit/mace.git 
pip install ./mace
```

### pip installation

To install via `pip`, follow the steps below:
```sh
# Create a virtual environment and activate it
python -m venv mace-venv
source mace-venv/bin/activate

# Install PyTorch (for example, for CUDA 11.6 [cu116])
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Clone and install MACE (and all required packages)
git clone git@github.com:ACEsuit/mace.git
pip install ./mace
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

## Usage

### Training 

To train a MACE model, you can use the `run_train.py` script:

```sh
python ./mace/scripts/run_train.py \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
```

To give a specific validation set, use the argument `--valid_file`. To set a larger batch size for evaluating the validation set, specify `--valid_batch_size`. 

To control the model's size, you need to change `--hidden_irreps`. For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`. It is also possible to specify the model using the     `--num_channels=128` and `--max_L=1`keys. 

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. To label them in the training set, set `config_type=IsolatedAtom` in their info fields. If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression. 

If the keyword `--swa` is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_swa` epochs). This setting usually helps lower the energy errors. 

The precision can be changed using the keyword ``--default_dtype``, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).

The keywords ``--batch_size`` and ``--max_num_epochs`` should be adapted based on the size of the training set. The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules. In this example, to make the code ignore stress on molecules, append to your molecules configuration a ``config_stress_weight = 0.0``.

To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify ``--device=mps``. 

### Evaluation

To evaluate your MACE model on an XYZ file, run the `eval_configs.py`:

```sh
python3 ./mace/scripts/eval_configs.py \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

## Tutorial

You can run our [Colab tutorial](https://colab.research.google.com/drive/1D6EtMUjQPey_GkuxUAbPgld6_9ibIa-V?authuser=1#scrollTo=Z10787RE1N8T) to quickly get started with MACE.

## On-line data loading for large datasets

If you have a large dataset that might not fit into the GPU memory it is recommended to preprocess the data on a CPU and use on-line dataloading for training the model. To preprocess your dataset specified as an xyz file run the `preprocess_data.py` script. An example is given here:

```sh
mkdir processed_data
python ./mace/scripts/preprocess_data.py \
    --train_file="/path/to/train_large.xyz" \
    --valid_fraction=0.05 \
    --test_file="/path/to/test_large.xyz" \
    --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \
    --r_max=4.5 \
    --h5_prefix="processed_data/" \
    --compute_statistics \
    --E0s="average" \
    --seed=123 \
```

To see all options and a little description of them run `python ./mace/scripts/preprocess_data.py --help` . The script will create a number of HDF5 files in the `processed_data` folder which can be used for training. There wiull be one file for trainin, one for validation and a separate one for each `config_type` in the test set. To train the model use the `run_train.py` script as follows:

```sh
python ./mace/scripts/run_train.py \
    --name="MACE_on_big_data" \
    --num_workers=16 \
    --train_file="./processed_data/train.h5" \
    --valid_file="./processed_data/valid.h5" \
    --test_dir="./processed_data" \
    --statistics_file="./processed_data/statistics.json" \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=100 \
    --swa \
    --start_swa=60 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --error_table='PerAtomMAE' \
    --device=cuda \
    --seed=123 \
```

## Weights and Biases for experiment tracking

If you would like to use MACE with Weights and Biases to log your experiments simply install with 

```sh
pip install ./mace[wandb]
```

And specify the necessary keyword arguments (`--wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_name`, `--wandb_log_hypers`)

## Development

We use `black`, `isort`, `pylint`, and `mypy`.
Run the following to format and check your code:
```sh
bash ./scripts/run_checks.sh
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:
```text
@inproceedings{
Batatia2022mace,
title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr.

For bugs or feature requests, please use [GitHub Issues](https://github.com/ACEsuit/mace/issues).

## License

MACE is published and distributed under the [MIT License](MIT.md).
