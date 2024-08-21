import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HeadConfig:
    train_file: Optional[str]
    valid_file: Optional[str]
    test_file: Optional[str]
    E0s: Optional[Any]
    statistics_file: Optional[str]
    valid_fraction: Optional[float]
    config_type_weights: Optional[Dict[str, float]]
    energy_key: Optional[str]
    forces_key: Optional[str]
    stress_key: Optional[str]
    virials_key: Optional[str]
    dipole_key: Optional[str]
    charges_key: Optional[str]
    keep_isolated_atoms: Optional[bool]
    atomic_numbers: Optional[Dict[str, int]]
    mean: Optional[float]
    std: Optional[float]
    avg_num_neighbors: Optional[float]
    compute_avg_num_neighbors: Optional[bool]


def dict_head_to_dataclass(head: Dict[str, Any]) -> HeadConfig:
    return HeadConfig(
        train_file=head.get("train_file"),
        valid_file=head.get("valid_file"),
        test_file=head.get("test_file"),
        E0s=head.get("E0s"),
        statistics_file=head.get("statistics_file"),
        valid_fraction=head.get("valid_fraction"),
        config_type_weights=head.get("config_type_weights"),
        energy_key=head.get("energy_key"),
        forces_key=head.get("forces_key"),
        stress_key=head.get("stress_key"),
        virials_key=head.get("virials_key"),
        dipole_key=head.get("dipole_key"),
        charges_key=head.get("charges_key"),
        keep_isolated_atoms=head.get("keep_isolated_atoms"),
    )


def prepare_default_head(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "Default": {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "E0s": args.E0s,
            "statistics_file": args.statistics_file,
            "valid_fraction": args.valid_fraction,
            "config_type_weights": args.config_type_weights,
            "energy_key": args.energy_key,
            "forces_key": args.forces_key,
            "stress_key": args.stress_key,
            "virials_key": args.virials_key,
            "dipole_key": args.dipole_key,
            "charges_key": args.charges_key,
            "keep_isolated_atoms": args.keep_isolated_atoms,
        }
    }
