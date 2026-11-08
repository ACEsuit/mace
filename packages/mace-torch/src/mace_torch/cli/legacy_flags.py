"""Pinned legacy flag grammar. No defaults reach the typed stages directly.

Interface data transcribed from the frozen argparse actions; the parity test
checks every spelling, type, action, default and nargs against that oracle.
"""

from __future__ import annotations

import argparse
import ast
from typing import Any

from mace_core.config.legacy import LEGACY_TRAIN_DESTS, Dropped, Kept, Merged, Reserved

from mace_torch.cli.eval_configs import boolean


def optional_float(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


# Defaults grouped by parser action and value type. Schema validation owns choices.
_GROUPS = (
    (
        str,
        "store",
        {
            "name": None,
            "work_dir": ".",
            "log_dir": None,
            "model_dir": None,
            "checkpoints_dir": None,
            "results_dir": None,
            "downloads_dir": None,
            "device": "cpu",
            "default_dtype": "float64",
            "launcher": "slurm",
            "log_level": "INFO",
            "error_table": "PerAtomRMSE",
            "model": "MACE",
            "radial_type": "bessel",
            "distance_transform": "None",
            "interaction": "RealAgnosticResidualInteractionBlock",
            "interaction_first": "RealAgnosticResidualInteractionBlock",
            "MLP_irreps": "16x0e",
            "radial_MLP": "[64, 64, 64]",
            "hidden_irreps": None,
            "edge_irreps": None,
            "gate": "silu",
            "field_feature_widths": "[1.0]",
            "field_feature_norms": None,
            "fixedpoint_update_config": None,
            "field_readout_config": None,
            "scaling": "rms_forces_scaling",
            "train_file": None,
            "valid_file": None,
            "test_file": None,
            "test_dir": None,
            "atomic_numbers": None,
            "statistics_file": None,
            "E0s": None,
            "heads": None,
            "foundation_head": None,
            "subselect_pt": "random",
            "filter_type_pt": "none",
            "pt_train_file": None,
            "pt_valid_file": None,
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "virials_key": "REF_virials",
            "stress_key": "REF_stress",
            "dipole_key": "dipole",
            "polarizability_key": "polarizability",
            "magmom_key": "REF_magmom",
            "magforces_key": "REF_magforces",
            "head_key": "head",
            "charges_key": "REF_charges",
            "elec_temp_key": "elec_temp",
            "total_spin_key": "total_spin",
            "total_charge_key": "total_charge",
            "embedding_specs": None,
            "skip_evaluate_heads": "pt_head",
            "loss": "weighted",
            "config_type_weights": '{"Default":1.0}',
            "optimizer": "adam",
            "lr_params_factors": (
                '{"embedding_lr_factor": 1.0, "interactions_lr_factor": 1.0, '
                '"products_lr_factor": 1.0, "readouts_lr_factor": 1.0}'
            ),
            "scheduler": "ReduceLROnPlateau",
            "foundation_model": None,
            "foundation_model_kwargs": "{}",
            "wandb_dir": None,
            "wandb_project": "",
            "wandb_entity": "",
            "wandb_name": "",
            "wandb_log_hypers": [
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
            "m_max": None,
            "data_aug_magmom_mode": "non-soc",
        },
    ),
    (
        int,
        "store",
        {
            "seed": 123,
            "plot_frequency": "0",
            "num_radial_basis": 8,
            "num_cutoff_basis": 5,
            "max_ell": 3,
            "correlation": 3,
            "num_interactions": 2,
            "num_channels": None,
            "max_L": None,
            "atomic_multipoles_max_l": 0,
            "field_feature_max_l": 0,
            "num_recursion_steps": 1,
            "num_workers": 0,
            "num_samples_pt": 10000,
            "lora_rank": 4,
            "warmup_steps_schedulefree": 0,
            "batch_size": 10,
            "valid_batch_size": 10,
            "freeze": None,
            "scheduler_patience": 50,
            "start_swa": None,
            "max_num_epochs": 2048,
            "patience": 2048,
            "eval_interval": 1,
            "num_mag_radial_basis_one_body": 10,
            "max_m_ell": 3,
            "num_mag_radial_basis": 8,
        },
    ),
    (
        str,
        "store_true",
        {
            "distributed": False,
            "pair_repulsion": False,
            "amsgrad": True,
            "swa": False,
            "lbfgs": False,
            "ema": False,
            "keep_checkpoints": False,
            "save_all_checkpoints": False,
            "restart_latest": False,
            "save_cpu": False,
            "dry_run": False,
            "wandb": False,
        },
    ),
    (
        boolean,
        "store",
        {
            "plot": True,
            "plot_interaction_e": False,
            "apply_cutoff": True,
            "use_last_readout_only": False,
            "use_embedding_readout": False,
            "use_reduced_cg": False,
            "use_so3": False,
            "use_agnostic_product": False,
            "use_edge_irreps_first": False,
            "field_si": False,
            "include_electrostatic_self_interaction": False,
            "add_local_electron_energy": False,
            "quadrupole_feature_corrections": False,
            "return_electrostatic_potentials": False,
            "compute_avg_num_neighbors": True,
            "compute_stress": False,
            "compute_forces": True,
            "compute_polarizability": False,
            "compute_atomic_dipole": False,
            "compute_magforces": False,
            "multi_processed_test": False,
            "pin_memory": True,
            "pseudolabel_replay": False,
            "pseudolabel_replay_compute_stress": False,
            "multiheads_finetuning": True,
            "force_mh_ft_lr": False,
            "foundation_model_elements": False,
            "keep_isolated_atoms": False,
            "lora": False,
            "foundation_model_readout": True,
            "finetune_dipoles_polarizabilities": False,
            "enable_cueq": False,
            "only_cueq": False,
            "enable_oeq": False,
            "use_magmom_one_body": False,
            "train_one_body_contribution": True,
            "data_aug_magmom": False,
        },
    ),
    (
        float,
        "store",
        {
            "r_max": 5.0,
            "kspace_cutoff_factor": 1.5,
            "atomic_multipoles_smearing_width": 1.0,
            "field_norm_factor": 0.02,
            "avg_num_neighbors": 1,
            "valid_fraction": 0.1,
            "mean": None,
            "std": None,
            "weight_pt_head": 1.0,
            "real_pt_data_ratio_threshold": 0.1,
            "lora_alpha": 1.0,
            "forces_weight": 100.0,
            "swa_forces_weight": 100.0,
            "magforces_weight": 100.0,
            "swa_magforces_weight": 100.0,
            "energy_weight": 1.0,
            "swa_energy_weight": 1000.0,
            "virials_weight": 1.0,
            "swa_virials_weight": 10.0,
            "stress_weight": 1.0,
            "swa_stress_weight": 10.0,
            "dipole_weight": 1.0,
            "swa_dipole_weight": 1.0,
            "swa_polarizability_weight": 1.0,
            "polarizability_weight": 1.0,
            "huber_delta": 0.01,
            "beta": 0.9,
            "beta1_schedulefree": 0.9,
            "beta2_schedulefree": 0.98,
            "lr": 0.01,
            "swa_lr": 0.001,
            "weight_decay": 5e-07,
            "lr_factor": 0.8,
            "lr_scheduler_gamma": 0.9993,
            "ema_decay": 0.99,
        },
    ),
    (
        ast.literal_eval,
        "store",
        {
            "les_arguments": None,
        },
    ),
    (
        str,
        "store_false",
        {
            "allow_random_padding_pt": True,
        },
    ),
    (
        optional_float,
        "store",
        {
            "clip_grad": 10.0,
        },
    ),
)

_OPTIONS = {
    "allow_random_padding_pt": ("--disallow_random_padding_pt",),
    "swa_forces_weight": ("--swa_forces_weight", "--stage_two_forces_weight"),
    "swa_magforces_weight": ("--swa_magforces_weight", "--stage_two_magforces_weight"),
    "swa_energy_weight": ("--swa_energy_weight", "--stage_two_energy_weight"),
    "swa_virials_weight": ("--swa_virials_weight", "--stage_two_virials_weight"),
    "swa_stress_weight": ("--swa_stress_weight", "--stage_two_stress_weight"),
    "swa_dipole_weight": ("--swa_dipole_weight", "--stage_two_dipole_weight"),
    "swa_polarizability_weight": (
        "--swa_polarizability_weight",
        "--stage_two_polarizability_weight",
    ),
    "swa_lr": ("--swa_lr", "--stage_two_lr"),
    "swa": ("--swa", "--stage_two"),
    "start_swa": ("--start_swa", "--start_stage_two"),
    "foundation_model_readout": ("--foundation_model_readout",),
}
_NARGS = {"foundation_model_readout": "?", "wandb_log_hypers": "+", "m_max": "+"}

FLAGS = tuple(
    (dest, kind, action, default, _NARGS.get(dest), _OPTIONS.get(dest, (f"--{dest}",)))
    for kind, action, values in _GROUPS
    for dest, default in values.items()
)
DEFAULTS = {dest: default for dest, _, _, default, _, _ in FLAGS}


# Flags that directly address the V1 document. Head paths are resolved by
# the configuration reader; the remaining legacy spellings use the translator.
DIRECT_FLAGS = {
    "name": ("runtime.name", "Name of the run and its files."),
    "seed": ("runtime.seed", "Random seed."),
    "work_dir": ("runtime.work_dir", "Where the run writes."),
    "device": ("runtime.device", "cpu, cuda, mps or xpu."),
    "train_file": ("data.heads.<head>.train_file", "Training structures for one head."),
    "valid_file": (
        "data.heads.<head>.valid_file",
        "Validation structures for one head.",
    ),
    "foundation_model": ("finetune.foundation_model", "V1 foundation checkpoint."),
    "max_num_epochs": ("training.max_num_epochs", "Epoch ceiling."),
}


def _help(dest: str) -> str:
    if dest in DIRECT_FLAGS:
        path, description = DIRECT_FLAGS[dest]
        return f"{description} Configuration: {path}."
    descriptions = {
        "batch_size": "Structures per training batch; positive integer.",
        "valid_batch_size": "Structures per validation batch; positive integer.",
        "r_max": "Neighbour cutoff radius in Angstrom.",
        "lr": "Base learning rate; nonnegative finite number.",
        "eval_interval": "Epochs between evaluations; positive integer.",
        "huber_delta": (
            "Positive Huber crossover in the scored quantity's units; "
            "requires huber or universal."
        ),
        "dry_run": (
            "Build data and model, then stop before training; "
            "use config validate for validation only."
        ),
    }
    if dest in descriptions:
        return descriptions[dest]
    disposition = LEGACY_TRAIN_DESTS.get(dest)
    if isinstance(disposition, Kept):
        return f"Set {disposition.path}."
    if isinstance(disposition, Merged):
        return f"Configures {disposition.into}: {disposition.note}."
    if isinstance(disposition, Dropped):
        return f"Retired on V1: {disposition.reason}."
    if isinstance(disposition, Reserved):
        return f"Configures {disposition.section}: {disposition.note}."
    return f"Legacy compatibility option {dest}; translated to the V1 configuration."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", help="Training configuration file: YAML, JSON or TOML."
    )
    parser.add_argument(
        "--foundation_filter_elements",
        dest="foundation_model_readout",
        type=boolean,
        default=argparse.SUPPRESS,
        help="Copy the foundation readout; alias of --foundation_model_readout.",
    )
    for dest, kind, action, _, nargs, options in FLAGS:
        settings: dict[str, Any] = {
            "dest": dest,
            "action": action,
            "default": argparse.SUPPRESS,
            "help": _help(dest),
        }
        if action == "store":
            settings.update(type=kind, nargs=nargs)
            if dest == "foundation_model_readout":
                # Legacy's bare flag disables copying; an explicit bool wins.
                settings["const"] = False
        parser.add_argument(*options, **settings)
