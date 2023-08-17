###########################################################################################
# Parsing functionalities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
from typing import Optional
import os


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name and seed
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=123)

    # Directories
    parser.add_argument(
        "--log_dir", help="directory for log files", type=str, default="logs"
    )
    parser.add_argument(
        "--model_dir", help="directory for final model", type=str, default="."
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="directory for checkpoint files",
        type=str,
        default="checkpoints",
    )
    parser.add_argument(
        "--results_dir", help="directory for results", type=str, default="results"
    )
    parser.add_argument(
        "--downloads_dir", help="directory for downloads", type=str, default="downloads"
    )

    # Device and logging
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--distributed",
        help="train in multi-GPU data parallel mode",
        action="store_true",
        default=False,
    )
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    parser.add_argument(
        "--error_table",
        help="Type of error table produced at the end of the training",
        type=str,
        choices=[
            "PerAtomRMSE",
            "TotalRMSE",
            "PerAtomRMSEstressvirials",
            "PerAtomMAE",
            "TotalMAE",
            "DipoleRMSE",
            "DipoleMAE",
            "EnergyDipoleRMSE",
        ],
        default="PerAtomRMSE",
    )

    # Model
    parser.add_argument(
        "--model",
        help="model type",
        default="MACE",
        choices=[
            "BOTNet",
            "MACE",
            "ScaleShiftMACE",
            "ScaleShiftBOTNet",
            "AtomicDipolesMACE",
            "EnergyDipolesMACE",
        ],
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", 
        type=float, 
        default=5.0
    )
    parser.add_argument(
        "--radial_type",
        help="type of radial basis functions",
        type=str,
        default="bessel",
        choices=["bessel", "gaussian"],
    )
    parser.add_argument(
        "--num_radial_basis",
        help="number of radial basis functions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_cutoff_basis",
        help="number of basis functions for smooth cutoff",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--interaction",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
        ],
    )
    parser.add_argument(
        "--interaction_first",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
        ],
    )
    parser.add_argument(
        "--max_ell", help=r"highest \ell of spherical harmonics", type=int, default=3
    )
    parser.add_argument(
        "--correlation", help="correlation order at each layer", type=int, default=3
    )
    parser.add_argument(
        "--num_interactions", help="number of interactions", type=int, default=2
    )
    parser.add_argument(
        "--MLP_irreps",
        help="hidden irreps of the MLP in last readout",
        type=str,
        default="16x0e",
    )
    parser.add_argument(
        "--radial_MLP",
        help="width of the radial MLP",
        type=str,
        default="[64, 64, 64]",
    )
    parser.add_argument(
        "--hidden_irreps",
        help="irreps for hidden node states",
        type=str,
        default="128x0e + 128x1o",
    )
    # add option to specify irreps by channel number and max L
    parser.add_argument(
        "--num_channels",
        help="number of embedding channels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_L",
        help="max L equivariance of the message",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gate",
        help="non linearity for last readout",
        type=str,
        default="silu",
        choices=["silu", "tanh", "abs", "None"],
    )
    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--avg_num_neighbors",
        help="normalization factor for the message",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--compute_avg_num_neighbors",
        help="normalization factor for the message",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--compute_stress",
        help="Select True to compute stress",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--compute_forces",
        help="Select True to compute forces",
        type=bool,
        default=True,
    )

    # Dataset
    parser.add_argument(
        "--train_file", help="Training set file, format is .xyz or .h5", type=str, 
        required=True,
    )
    parser.add_argument(
        "--valid_file",
        help="Validation set .xyz or .h5 file",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set .xyz pt .h5 file",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        help="Path to directory with test files named as test_*.h5",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--multi_processed_test",
        help="Boolean value for whether the test data was multiprocessed",
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for data loading",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin_memory",
        help="Pin memory for data loading",
        default=True, 
        type=bool,
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--mean",
        help="Mean energy per atom of training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--std",
        help="Standard deviation of force components in the training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--statistics_file",
        help="json file containing statistics of training set",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default="virials",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default="dipole",
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default="charges",
    )

    # Loss and optimization
    parser.add_argument(
        "--loss",
        help="type of loss",
        default="weighted",
        choices=[
            "ef",
            "weighted",
            "forces_only",
            "virials",
            "stress",
            "dipole",
            "huber",
            "energy_forces_dipole",
        ],
    )
    parser.add_argument(
        "--forces_weight", help="weight of forces loss", type=float, default=100.0
    )
    parser.add_argument(
        "--swa_forces_weight",
        help="weight of forces loss after starting swa",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--energy_weight", help="weight of energy loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_energy_weight",
        help="weight of energy loss after starting swa",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--virials_weight", help="weight of virials loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_virials_weight",
        help="weight of virials loss after starting swa",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--stress_weight", help="weight of virials loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_stress_weight",
        help="weight of stress loss after starting swa",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--dipole_weight", help="weight of dipoles loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_dipole_weight",
        help="weight of dipoles after starting swa",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--huber_delta",
        help="delta parameter for huber loss",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer for parameter optimization",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=10)
    parser.add_argument(
        "--valid_batch_size", help="Validation batch size", type=int, default=10
    )
    parser.add_argument(
        "--lr", help="Learning rate of optimizer", type=float, default=0.01
    )
    parser.add_argument(
        "--swa_lr", help="Learning rate of optimizer in swa", type=float, default=1e-3
    )
    parser.add_argument(
        "--weight_decay", help="weight decay (L2 penalty)", type=float, default=5e-7
    )
    parser.add_argument(
        "--amsgrad",
        help="use amsgrad variant of optimizer",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--scheduler", help="Type of scheduler", type=str, default="ReduceLROnPlateau"
    )
    parser.add_argument(
        "--lr_factor", help="Learning rate factor", type=float, default=0.8
    )
    parser.add_argument(
        "--scheduler_patience", help="Learning rate factor", type=int, default=50
    )
    parser.add_argument(
        "--lr_scheduler_gamma",
        help="Gamma of learning rate scheduler",
        type=float,
        default=0.9993,
    )
    parser.add_argument(
        "--swa",
        help="use Stochastic Weight Averaging, which decreases the learning rate and increases the energy weight at the end of the training to help converge them",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--start_swa",
        help="Number of epochs before switching to swa",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ema",
        help="use Exponential Moving Average",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema_decay",
        help="Exponential Moving Average decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--max_num_epochs", help="Maximum number of epochs", type=int, default=2048
    )
    parser.add_argument(
        "--patience",
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--eval_interval", help="evaluate model every <n> epochs", type=int, default=2
    )
    parser.add_argument(
        "--keep_checkpoints",
        help="keep all checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--restart_latest",
        help="restart optimizer from latest checkpoint",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_cpu",
        help="Save a model to be loaded on cpu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--clip_grad",
        help="Gradient Clipping Value",
        type=check_float_or_none,
        default=10.0,
    )
    # options for using Weights and Biases for experiment tracking
    # to install see https://wandb.ai
    parser.add_argument(
        "--wandb",
        help="Use Weights and Biases for experiment tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_project",
        help="Weights and Biases project name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_entity",
        help="Weights and Biases entity name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_name",
        help="Weights and Biases experiment name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_log_hypers",
        help="The hyperparameters to log in Weights and Biases",
        type=list,
        default=[
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
        ],
    )
    return parser

def build_preprocess_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        help="Training set h5 file",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--valid_file",
        help="Training set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_process",
        help="The user defined number of processes to use, as well as the number of files created.", 
        type=int, 
        default=int(os.cpu_count()/4)
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--h5_prefix",
        help="Prefix for h5 files when saving",
        type=str,
        default="",
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", 
        type=float, 
        default=5.0
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default="virials",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default="dipole",
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default="charges",
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--compute_statistics",
        help="Compute statistics for the dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size", 
        help="batch size to compute average number of neighbours", 
        type=int, 
        default=16,
    )

    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle the training dataset",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for splitting training and validation sets",
        type=int,
        default=123,
    )
    return parser


def check_float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        if value != "None":
            raise argparse.ArgumentTypeError(
                f"{value} is an invalid value (float or None)"
            ) from None
        return None
