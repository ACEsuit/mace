###########################################################################################
# Parsing functionalities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the ASL License (see ASL.md)
###########################################################################################

import argparse
from typing import Optional
from aaargs import ArgumentParser, Argument


def check_float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        if value != "None":
            raise argparse.ArgumentTypeError(
                f"{value} is an invalid value (float or None)"
            ) from None
        return None


class MaceArguments(ArgumentParser):
    # Name and seed
    name = Argument(help="experiment name", required=True)
    seed = Argument(help="random seed", type=int, default=123)
    # Directories
    log_dir = Argument(help="directory for log files", type=str, default="logs")
    model_dir = Argument(help="directory for final model", type=str, default=".")
    checkpoints_dir = Argument(
        help="directory for checkpoint files",
        type=str,
        default="checkpoints",
    )
    results_dir = Argument(help="directory for results", type=str, default="results")
    downloads_dir = Argument(
        help="directory for downloads", type=str, default="downloads"
    )

    # Device and logging
    device = Argument(
        help="select device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    default_dtype = Argument(
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )

    log_level = Argument(help="log level", type=str, default="INFO")
    error_table = Argument(
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
    model = Argument(
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
    r_max = Argument(help="distance cutoff (in Ang)", type=float, default=5.0)
    num_radial_basis = Argument(
        help="number of radial basis functions",
        type=int,
        default=8,
    )
    num_cutoff_basis = Argument(
        help="number of basis functions for smooth cutoff",
        type=int,
        default=5,
    )
    interaction = Argument(
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
        ],
    )
    interaction_first = Argument(
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
        ],
    )
    max_ell = Argument(help=r"highest \ell of spherical harmonics", type=int, default=3)
    correlation = Argument(help="correlation order at each layer", type=int, default=3)
    num_interactions = Argument(help="number of interactions", type=int, default=2)
    MLP_irreps = Argument(
        help="hidden irreps of the MLP in last readout",
        type=str,
        default="16x0e",
    )
    hidden_irreps = Argument(
        help="irreps for hidden node states",
        type=str,
        default="32x0e",
    )
    gate = Argument(
        help="non linearity for last readout",
        type=str,
        default="silu",
        choices=["silu", "tanh", "abs", "None"],
    )
    scaling = Argument(
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    avg_num_neighbors = Argument(
        help="normalization factor for the message",
        type=float,
        default=1,
    )
    compute_avg_num_neighbors = Argument(
        help="normalization factor for the message",
        type=bool,
        default=True,
    )
    compute_stress = Argument(
        help="Select True to compute stress",
        type=bool,
        default=False,
    )
    compute_forces = Argument(
        help="Select True to compute forces",
        type=bool,
        default=True,
    )

    # Dataset
    train_file = Argument(help="Training set xyz file", type=str, required=True)
    valid_file = Argument(
        help="Validation set xyz file",
        default=None,
        type=str,
        required=False,
    )
    valid_fraction = Argument(
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    test_file = Argument(
        help="Test set xyz file",
        type=str,
    )
    E0s = Argument(
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    energy_key = Argument(
        help="Key of reference energies in training xyz",
        type=str,
        default="energy",
    )
    forces_key = Argument(
        help="Key of reference forces in training xyz",
        type=str,
        default="forces",
    )
    virials_key = Argument(
        help="Key of reference virials in training xyz",
        type=str,
        default="virials",
    )
    stress_key = Argument(
        help="Key of reference stress in training xyz",
        type=str,
        default="stress",
    )
    dipole_key = Argument(
        help="Key of reference dipoles in training xyz",
        type=str,
        default="dipole",
    )
    charges_key = Argument(
        help="Key of atomic charges in training xyz",
        type=str,
        default="charges",
    )

    # Loss and optimization
    loss = Argument(
        help="type of loss",
        default="weighted",
        choices=[
            "ef",
            "weighted",
            "forces_only",
            "virials",
            "stress",
            "dipole",
            "energy_forces_dipole",
        ],
    )
    forces_weight = Argument(help="weight of forces loss", type=float, default=10.0)
    swa_forces_weight = Argument(
        help="weight of forces loss after starting swa",
        type=float,
        default=1.0,
    )
    energy_weight = Argument(help="weight of energy loss", type=float, default=1.0)
    swa_energy_weight = Argument(
        help="weight of energy loss after starting swa",
        type=float,
        default=1000.0,
    )
    virials_weight = Argument(help="weight of virials loss", type=float, default=1.0)
    stress_weight = Argument(help="weight of stress loss", type=float, default=1.0)
    dipole_weight = Argument(help="weight of dipoles loss", type=float, default=1.0)
    swa_dipole_weight = Argument(
        help="weight of dipoles after starting swa",
        type=float,
        default=1.0,
    )
    config_type_weights = Argument(
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    optimizer = Argument(
        help="Optimizer for parameter optimization",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
    )
    batch_size = Argument(help="batch size", type=int, default=10)
    valid_batch_size = Argument(help="Validation batch size", type=int, default=10)
    lr = Argument(help="Learning rate of optimizer", type=float, default=0.01)
    swa_lr = Argument(
        help="Learning rate of optimizer in swa", type=float, default=1e-3
    )
    weight_decay = Argument(help="weight decay (L2 penalty)", type=float, default=5e-7)
    amsgrad: bool = Argument(
        help="use amsgrad variant of optimizer",
        default=True,
    )
    scheduler = Argument(
        help="Type of scheduler", type=str, default="ReduceLROnPlateau"
    )
    lr_factor = Argument(help="Learning rate factor", type=float, default=0.8)
    scheduler_patience = Argument(help="Learning rate factor", type=int, default=50)
    lr_scheduler_gamma = Argument(
        help="Gamma of learning rate scheduler",
        type=float,
        default=0.9993,
    )
    swa: bool = Argument(
        help="use Stochastic Weight Averaging, which decreases the learning rate and increases the energy weight at the end of the training to help converge them",
        default=False,
    )
    start_swa = Argument(
        help="Number of epochs before switching to swa",
        type=int,
        default=None,
    )
    ema: bool = Argument(
        help="use Exponential Moving Average",
        default=False,
    )
    ema_decay = Argument(
        help="Exponential Moving Average decay",
        type=float,
        default=0.99,
    )
    max_num_epochs = Argument(help="Maximum number of epochs", type=int, default=2048)
    patience = Argument(
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=2048,
    )
    eval_interval = Argument(
        help="evaluate model every <n> epochs", type=int, default=2
    )
    keep_checkpoints: bool = Argument(
        help="keep all checkpoints",
        default=False,
    )
    restart_latest: bool = Argument(
        help="restart optimizer from latest checkpoint",
        default=False,
    )
    save_cpu: bool = Argument(
        help="Save a model to be loaded on cpu",
        default=False,
    )
    clip_grad = Argument(
        help="Gradient Clipping Value",
        type=check_float_or_none,
        default=10.0,
    )


def build_default_arg_parser() -> argparse.ArgumentParser:
    return MaceArguments.get_parser()
