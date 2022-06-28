# MACE

This repository contains the MACE reference implementation developed by
Ilyes Batatia, Gregor Simm, and David Kovacs.

## Installation

Requirements:
* Python >= 3.7
* [PyTorch](https://pytorch.org/) >= 1.8

To install, follow the steps below:
```sh
# Create a virtual environment and activate it
python3.8 -m venv mace-venv
source mace-venv/bin/activate

# Select CUDA version
CUDA="cu102"  # or, for instance, "cpu"

# Install PyTorch
pip install torch==1.8.2 --extra-index-url "https://download.pytorch.org/whl/lts/1.8/${CUDA}"

# Clone and install MACE (and all required packages)
git clone git@github.com:ACEsuit/mace.git
pip install ./mace
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

## Usage

### Training 

To train a MACE model, you can use the `run_train.py` script :

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
    --restart_latest \
    --device=cuda \
```

To give a specific validation set use the keyword `--valid_file`. To set a larger batch size for evaluating the validation set use the key `--valid_batch_size`. 

To control the size of the model you need to change `--hidden_irreps`. For most applications the recommended default model size is `--hidden_irreps='128x0e'` meaning 128 invariant messages or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough than you can increase it's size either to higher order equivariant features eg. `128x0e + 128x1o + 128x2e` or increasing the number of messages to `256`. 

It is usually preferred to add the isolated atoms to the training set, rather than parsing in their energies like above. They should be labelled by having in the `info` field `config_type`  set to `IsolatedAtom`. If you do not want to use or do not know the isolated atom energy you should set `--model=ScaleShiftMACE` and pass in a dictionary of 0-s: `E0s='{1:0.0, 6:0.0}'` or similar. 

The keyword `--swa` is used to enable an option where at the end of the training for the last ca 20% of the epochs (from `--start_swa` epochs) the loss is changed such that the energy weight is increased. This setting usually helps lower the energy errors. 

### Evaluation

To evaluate your MACE model on an xyz file, run the `eval_configs.py` :

```sh
python3 ./LieACE-real/scripts/eval_configs.py \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

## Tutorial

You can run our [Colab tutorial](https://colab.research.google.com/drive/1D6EtMUjQPey_GkuxUAbPgld6_9ibIa-V?authuser=1#scrollTo=Z10787RE1N8T) to quickly get started with MACE.

## Development

We use `black`, `isort`, `pylint`, and `mypy`.
Run the following to format and check your code:
```sh
bash ./scripts/run_checks.sh
```

We have CI set up to check this, but we _highly_ recommend that you run those commands
before you commit (and push) to avoid accidentally committing bad code.

## References

If you use this code, please cite our papers:
```text
@misc{Batatia2022MACE,
  title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2206.07697},
  eprint = {2206.07697},
  eprinttype = {arxiv},
  doi = {10.48550/ARXIV.2206.07697},
  archiveprefix = {arXiv}
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

For bugs or request of new features please use github Issues.
