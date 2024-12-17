import argparse
import dataclasses
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from mace.cli.fine_tuning_select import select_samples
from mace.tools.scripts_utils import (
    SubsetCollection,
    dict_to_namespace,
    get_dataset_from_xyz,
)


@dataclasses.dataclass
class HeadConfig:
    head_name: str
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    test_file: Optional[str] = None
    test_dir: Optional[str] = None
    E0s: Optional[Any] = None
    statistics_file: Optional[str] = None
    valid_fraction: Optional[float] = None
    config_type_weights: Optional[Dict[str, float]] = None
    energy_key: Optional[str] = None
    forces_key: Optional[str] = None
    stress_key: Optional[str] = None
    virials_key: Optional[str] = None
    dipole_key: Optional[str] = None
    charges_key: Optional[str] = None
    keep_isolated_atoms: Optional[bool] = None
    atomic_numbers: Optional[Union[List[int], List[str]]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: Optional[bool] = None
    collections: Optional[SubsetCollection] = None
    train_loader: torch.utils.data.DataLoader = None
    z_table: Optional[Any] = None
    atomic_energies_dict: Optional[Dict[str, float]] = None


def dict_head_to_dataclass(
    head: Dict[str, Any], head_name: str, args: argparse.Namespace
) -> HeadConfig:

    return HeadConfig(
        head_name=head_name,
        train_file=head.get("train_file", args.train_file),
        valid_file=head.get("valid_file", args.valid_file),
        test_file=head.get("test_file", None),
        test_dir=head.get("test_dir", None),
        E0s=head.get("E0s", args.E0s),
        statistics_file=head.get("statistics_file", args.statistics_file),
        valid_fraction=head.get("valid_fraction", args.valid_fraction),
        config_type_weights=head.get("config_type_weights", args.config_type_weights),
        compute_avg_num_neighbors=head.get(
            "compute_avg_num_neighbors", args.compute_avg_num_neighbors
        ),
        atomic_numbers=head.get("atomic_numbers", args.atomic_numbers),
        mean=head.get("mean", args.mean),
        std=head.get("std", args.std),
        avg_num_neighbors=head.get("avg_num_neighbors", args.avg_num_neighbors),
        energy_key=head.get("energy_key", args.energy_key),
        forces_key=head.get("forces_key", args.forces_key),
        stress_key=head.get("stress_key", args.stress_key),
        virials_key=head.get("virials_key", args.virials_key),
        dipole_key=head.get("dipole_key", args.dipole_key),
        charges_key=head.get("charges_key", args.charges_key),
        keep_isolated_atoms=head.get("keep_isolated_atoms", args.keep_isolated_atoms),
    )


def prepare_default_head(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "default": {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "test_dir": args.test_dir,
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


def assemble_mp_data(
    args: argparse.Namespace, tag: str, head_configs: List[HeadConfig]
) -> Dict[str, Any]:
    try:
        checkpoint_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mp_traj_combined.xyz"
        descriptors_url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/descriptors.npy"
        cache_dir = (
            Path(os.environ.get("XDG_CACHE_HOME", "~/")).expanduser() / ".cache/mace"
        )
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
        )
        cached_dataset_path = f"{cache_dir}/{checkpoint_url_name}"
        descriptors_url_name = "".join(
            c for c in os.path.basename(descriptors_url) if c.isalnum() or c in "_"
        )
        cached_descriptors_path = f"{cache_dir}/{descriptors_url_name}"
        if not os.path.isfile(cached_dataset_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            logging.info("Downloading MP structures for finetuning")
            _, http_msg = urllib.request.urlretrieve(
                checkpoint_url, cached_dataset_path
            )
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Dataset download failed, please check the URL {checkpoint_url}"
                )
            logging.info(f"Materials Project dataset to {cached_dataset_path}")
        if not os.path.isfile(cached_descriptors_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            logging.info("Downloading MP descriptors for finetuning")
            _, http_msg = urllib.request.urlretrieve(
                descriptors_url, cached_descriptors_path
            )
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Descriptors download failed, please check the URL {descriptors_url}"
                )
            logging.info(f"Materials Project descriptors to {cached_descriptors_path}")
        dataset_mp = cached_dataset_path
        descriptors_mp = cached_descriptors_path
        msg = f"Using Materials Project dataset with {dataset_mp}"
        logging.info(msg)
        msg = f"Using Materials Project descriptors with {descriptors_mp}"
        logging.info(msg)
        config_pt_paths = [head.train_file for head in head_configs]
        args_samples = {
            "configs_pt": dataset_mp,
            "configs_ft": config_pt_paths,
            "num_samples": args.num_samples_pt,
            "seed": args.seed,
            "model": args.foundation_model,
            "head_pt": "pbe_mp",
            "head_ft": "Default",
            "weight_pt": args.weight_pt_head,
            "weight_ft": 1.0,
            "filtering_type": "combination",
            "output": f"mp_finetuning-{tag}.xyz",
            "descriptors": descriptors_mp,
            "subselect": args.subselect_pt,
            "device": args.device,
            "default_dtype": args.default_dtype,
        }
        select_samples(dict_to_namespace(args_samples))
        collections_mp, _ = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=f"mp_finetuning-{tag}.xyz",
            valid_path=None,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            energy_key="energy",
            forces_key="forces",
            stress_key="stress",
            head_name="pt_head",
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        return collections_mp
    except Exception as exc:
        raise RuntimeError(
            "Model or descriptors download failed and no local model found"
        ) from exc
