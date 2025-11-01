import argparse
import ast
import dataclasses
import logging
import os
import urllib.request
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch

from mace.cli.fine_tuning_select import (
    FilteringType,
    SelectionSettings,
    SubselectType,
    select_samples,
)
from mace.data import AtomicData, KeySpecification
from mace.data.utils import Configuration
from mace.tools import torch_geometric
from mace.tools.scripts_utils import SubsetCollection, get_dataset_from_xyz
from mace.tools.utils import AtomicNumberTable, get_cache_dir


@dataclasses.dataclass
class HeadConfig:
    head_name: str
    key_specification: KeySpecification
    train_file: Optional[Union[str, List[str]]] = None
    valid_file: Optional[Union[str, List[str]]] = None
    test_file: Optional[str] = None
    test_dir: Optional[str] = None
    E0s: Optional[Any] = None
    statistics_file: Optional[str] = None
    valid_fraction: Optional[float] = None
    config_type_weights: Optional[Dict[str, float]] = None
    keep_isolated_atoms: Optional[bool] = None
    atomic_numbers: Optional[Union[List[int], List[str]]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    avg_num_neighbors: Optional[float] = None
    compute_avg_num_neighbors: Optional[bool] = None
    collections: Optional[SubsetCollection] = None
    train_loader: Optional[torch.utils.data.DataLoader] = None
    z_table: Optional[Any] = None
    atomic_energies_dict: Optional[Dict[str, float]] = None


def dict_head_to_dataclass(
    head: Dict[str, Any], head_name: str, args: argparse.Namespace
) -> HeadConfig:
    """Convert head dictionary to HeadConfig dataclass."""
    # parser+head args that have no defaults but are required
    if (args.train_file is None) and (head.get("train_file", None) is None):
        raise ValueError(
            "train file is not set in the head config yaml or via command line args"
        )

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
        key_specification=head["key_specification"],
        keep_isolated_atoms=head.get("keep_isolated_atoms", args.keep_isolated_atoms),
    )


def prepare_default_head(args: argparse.Namespace) -> Dict[str, Any]:
    """Prepare a default head from args."""
    return {
        "Default": {
            "train_file": args.train_file,
            "valid_file": args.valid_file,
            "test_file": args.test_file,
            "test_dir": args.test_dir,
            "E0s": args.E0s,
            "statistics_file": args.statistics_file,
            "key_specification": args.key_specification,
            "valid_fraction": args.valid_fraction,
            "config_type_weights": args.config_type_weights,
            "keep_isolated_atoms": args.keep_isolated_atoms,
        }
    }


def prepare_pt_head(
    args: argparse.Namespace,
    pt_keyspec: KeySpecification,
    foundation_model_num_neighbours: float,
) -> Dict[str, Any]:
    """Prepare a pretraining head from args."""
    if (
        args.foundation_model in ["small", "medium", "large"]
        or args.pt_train_file == "mp"
    ):
        logging.info(
            "Using foundation model for multiheads finetuning with Materials Project data"
        )
        pt_keyspec.update(
            info_keys={"energy": "energy", "stress": "stress"},
            arrays_keys={"forces": "forces"},
        )
        pt_head = {
            "train_file": "mp",
            "E0s": "foundation",
            "statistics_file": None,
            "key_specification": pt_keyspec,
            "avg_num_neighbors": foundation_model_num_neighbours,
            "compute_avg_num_neighbors": False,
        }
    else:
        pt_head = {
            "train_file": args.pt_train_file,
            "valid_file": args.pt_valid_file,
            "E0s": "foundation",
            "statistics_file": args.statistics_file,
            "valid_fraction": args.valid_fraction,
            "key_specification": pt_keyspec,
            "avg_num_neighbors": foundation_model_num_neighbours,
            "keep_isolated_atoms": args.keep_isolated_atoms,
            "compute_avg_num_neighbors": False,
        }

    return pt_head


def assemble_replay_data(
    name: str,
    args: argparse.Namespace,
    head_config_pt: HeadConfig,
    tag: str,
) -> SubsetCollection:
    """Assemble data for replay fine-tuning."""
    try:
        if name == "mp":
            checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b/mp_traj_combined.xyz"
        elif name == "matpes_pbe":
            checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/matpes-pbe-replay-data.xyz"
        elif name == "matpes_r2scan":
            checkpoint_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/matpes-r2scan-replay-data.extxyz"
        else:
            raise ValueError(f"Unknown replay dataset name {name}")

        cache_dir = get_cache_dir()
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
        )
        cached_dataset_path = f"{cache_dir}/{checkpoint_url_name}"
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
        output = f"mp_finetuning-{tag}.xyz"
        atomic_numbers = (
            ast.literal_eval(args.atomic_numbers)
            if args.atomic_numbers is not None
            else None
        )
        settings = SelectionSettings(
            configs_pt=cached_dataset_path,
            output=f"mp_finetuning-{tag}.xyz",
            atomic_numbers=atomic_numbers,
            num_samples=args.num_samples_pt,
            seed=args.seed,
            head_pt="pbe_mp",
            weight_pt=args.weight_pt_head,
            filtering_type=FilteringType(args.filter_type_pt),
            subselect=SubselectType(args.subselect_pt),
            default_dtype=args.default_dtype,
            allow_random_padding=args.allow_random_padding_pt,
        )
        select_samples(settings)
        head_config_pt.train_file = [output]
        collections_mp, _ = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=output,
            valid_path=None,
            valid_fraction=args.valid_fraction,
            config_type_weights=None,
            test_path=None,
            seed=args.seed,
            key_specification=head_config_pt.key_specification,
            head_name="pt_head",
            keep_isolated_atoms=args.keep_isolated_atoms,
            no_data_ok=(
                args.pseudolabel_replay
                and args.multiheads_finetuning
                and head_config_pt.head_name == "pt_head"
            ),
            prefix=args.name,
        )
        return collections_mp
    except Exception as exc:
        raise RuntimeError(
            "Foundation model replay data or descriptors cached data not found and download failed"
        ) from exc


def generate_pseudolabels_for_configs(
    model: torch.nn.Module,
    configs: List[Configuration],
    z_table: AtomicNumberTable,
    r_max: float,
    device: torch.device,
    batch_size: int,
) -> List[Configuration]:
    """
    Generate pseudolabels for a list of Configuration objects.

    Args:
        model: The foundation model
        configs: List of Configuration objects
        z_table: Atomic number table
        r_max: Cutoff radius
        device: Device to run model on
        batch_size: Batch size for inference

    Returns:
        List of Configuration objects with updated properties
    """

    model.eval()
    updated_configs = []

    # Disable gradient tracking for model parameters
    original_requires_grad = {}
    for param in model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = False

    # Process configs in batches
    for i in range(0, len(configs), batch_size):
        batch_configs = configs[i : i + batch_size]

        try:
            # Create temporary AtomicData objects for this batch
            batch_data = [
                AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
                for config in batch_configs
            ]

            # Create a batch for model inference
            batch = torch_geometric.Batch.from_data_list(batch_data).to(device)
            batch_dict = batch.to_dict()

            # Run model inference with computation of all properties
            out = model(
                batch_dict,
                training=False,
                compute_force=True,
                compute_virials=True,
                compute_stress=True,
            )

            # Process each configuration in the batch
            for j, config in enumerate(batch_configs):
                # Create a deepcopy to avoid modifying the original
                config_copy = deepcopy(config)

                # Ensure properties dict exists
                if not hasattr(config_copy, "properties"):
                    config_copy.properties = {}

                # Update config properties with pseudolabels
                if "energy" in out and out["energy"] is not None:
                    config_copy.properties["energy"] = (
                        out["energy"][j].detach().cpu().item()
                    )
                if "forces" in out and out["forces"] is not None:
                    # Forces are per atom
                    node_start = batch.ptr[j].item()
                    node_end = batch.ptr[j + 1].item()

                    config_copy.properties["forces"] = (
                        out["forces"][node_start:node_end].detach().cpu().numpy()
                    )
                if "stress" in out and out["stress"] is not None:
                    config_copy.properties["stress"] = (
                        out["stress"][j].detach().cpu().numpy()
                    )
                if "virials" in out and out["virials"] is not None:
                    config_copy.properties["virials"] = (
                        out["virials"][j].detach().cpu().numpy()
                    )
                if "dipole" in out and out["dipole"] is not None:
                    config_copy.properties["dipole"] = (
                        out["dipole"][j].detach().cpu().numpy()
                    )
                if "charges" in out and out["charges"] is not None:
                    # Charges are per atom
                    node_start = batch.ptr[j].item()
                    node_end = batch.ptr[j + 1].item()

                    config_copy.properties["charges"] = (
                        out["charges"][node_start:node_end].detach().cpu().numpy()
                    )

                updated_configs.append(config_copy)

        except Exception as e:  # pylint: disable=broad-except
            logging.error(
                f"Error generating pseudolabels for batch {i//batch_size + 1}: {str(e)}"
            )
            # On error, return the original configs for this batch
            updated_configs.extend([deepcopy(config) for config in batch_configs])

    # Restore original requires_grad settings
    for param, requires_grad in original_requires_grad.items():
        param.requires_grad = requires_grad

    logging.info(f"Generated pseudolabels for {len(updated_configs)} configurations")
    return updated_configs


def apply_pseudolabels_to_pt_head_configs(
    foundation_model: torch.nn.Module,
    pt_head_config: HeadConfig,
    r_max: float,
    device: torch.device,
    batch_size: int,
) -> bool:
    """
    Apply pseudolabels to pt_head configurations using the foundation model.

    Args:
        foundation_model: The pre-loaded foundation model
        pt_head_config: The HeadConfig object for pt_head
        r_max: Cutoff radius
        device: Device to run model on
        batch_size: Batch size for inference

    Returns:
        bool: True if pseudolabeling was successful, False otherwise
    """

    try:
        logging.info(
            "Applying pseudolabels to pt_head configurations using foundation model"
        )

        foundation_model.to(device)

        # Use foundation model's z_table if available
        if hasattr(foundation_model, "atomic_numbers"):
            z_table = AtomicNumberTable(
                sorted(foundation_model.atomic_numbers.tolist())
            )
            logging.info(
                f"Using foundation model's atomic numbers for pseudolabeling: {z_table.zs}"
            )
        elif hasattr(pt_head_config, "z_table") and pt_head_config.z_table is not None:
            z_table = pt_head_config.z_table
            logging.info(f"Using pt_head's z_table for pseudolabeling: {z_table.zs}")
        else:
            logging.warning("No atomic number table available for pseudolabeling")
            return False

        # Process training configurations
        if (
            hasattr(pt_head_config.collections, "train")
            and pt_head_config.collections.train
        ):
            logging.info(
                f"Generating pseudolabels for {len(pt_head_config.collections.train)} pt_head training configurations"
            )
            updated_train_configs = generate_pseudolabels_for_configs(
                model=foundation_model,
                configs=pt_head_config.collections.train,
                z_table=z_table,
                r_max=r_max,
                device=device,
                batch_size=batch_size,
            )

            # Replace the original configurations with updated ones
            pt_head_config.collections.train = updated_train_configs
            logging.info(
                f"Successfully applied pseudolabels to {len(updated_train_configs)} training configurations"
            )

        # Process validation configurations if they exist
        if (
            hasattr(pt_head_config.collections, "valid")
            and pt_head_config.collections.valid
        ):
            logging.info(
                f"Generating pseudolabels for {len(pt_head_config.collections.valid)} pt_head validation configurations"
            )
            updated_valid_configs = generate_pseudolabels_for_configs(
                model=foundation_model,
                configs=pt_head_config.collections.valid,
                z_table=z_table,
                r_max=r_max,
                device=device,
                batch_size=batch_size,
            )

            # Replace the original configurations with updated ones
            pt_head_config.collections.valid = updated_valid_configs
            logging.info(
                f"Successfully applied pseudolabels to {len(updated_valid_configs)} validation configurations"
            )

        return True

    except Exception as e:  # pylint: disable=broad-except
        logging.error(f"Error applying pseudolabels: {str(e)}")
        return False
