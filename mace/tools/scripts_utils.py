###########################################################################################
# Training utils
# Authors: David Kovacs, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel

from mace import data, modules, tools
from mace.tools.train import SWAContainer


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    work_dir: str,
    train_path: str,
    valid_path: Optional[str],
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    keep_isolated_atoms: bool = False,
    head_name: str = "Default",
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "virials",
    dipole_key: str = "dipoles",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
        extract_atomic_energies=True,
        keep_isolated_atoms=keep_isolated_atoms,
        head_name=head_name,
    )
    logging.info(
        f"Training set [{len(all_train_configs)} configs, {np.sum([1 if config.energy else 0 for config in all_train_configs])} energy, {np.sum([config.forces.size for config in all_train_configs])} forces] loaded from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        logging.info(
            f"Validation set [{len(valid_configs)} configs, {np.sum([1 if config.energy else 0 for config in valid_configs])} energy, {np.sum([config.forces.size for config in valid_configs])} forces] loaded from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed, work_dir
        )
        logging.info(
            f"Validaton set contains {len(valid_configs)} configurations [{np.sum([1 if config.energy else 0 for config in valid_configs])} energy, {np.sum([config.forces.size for config in valid_configs])} forces]"
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            dipole_key=dipole_key,
            stress_key=stress_key,
            virials_key=virials_key,
            charges_key=charges_key,
            head_key=head_key,
            extract_atomic_energies=False,
            head_name=head_name,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Test set ({len(all_test_configs)} configs) loaded from '{test_path}':"
        )
        for name, tmp_configs in test_configs:
            logging.info(
                f"{name}: {len(tmp_configs)} configs, {np.sum([1 if config.energy else 0 for config in tmp_configs])} energy, {np.sum([config.forces.size for config in tmp_configs])} forces"
            )

    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


def get_config_type_weights(ct_weights):
    """
    Get config type weights from command line argument
    """
    try:
        config_type_weights = ast.literal_eval(ct_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}
    return config_type_weights


def print_git_commit():
    try:
        import git

        repo = git.Repo(search_parent_directories=True)
        commit = repo.head.commit.hexsha
        logging.debug(f"Current Git commit: {commit}")
        return commit
    except Exception as e:  # pylint: disable=W0703
        logging.debug(f"Error accessing Git repository: {e}")
        return "None"


def extract_config_mace_model(model: torch.nn.Module) -> Dict[str, Any]:
    if model.__class__.__name__ != "ScaleShiftMACE":
        return {"error": "Model is not a ScaleShiftMACE model"}

    def radial_to_name(radial_type):
        if radial_type == "BesselBasis":
            return "bessel"
        if radial_type == "GaussianBasis":
            return "gaussian"
        if radial_type == "ChebychevBasis":
            return "chebyshev"
        return radial_type

    def radial_to_transform(radial):
        if not hasattr(radial, "distance_transform"):
            return None
        if radial.distance_transform.__class__.__name__ == "AgnesiTransform":
            return "Agnesi"
        if radial.distance_transform.__class__.__name__ == "SoftTransform":
            return "Soft"
        return radial.distance_transform.__class__.__name__

    scale = model.scale_shift.scale
    shift = model.scale_shift.shift
    config = {
        "r_max": model.r_max.item(),
        "num_bessel": len(model.radial_embedding.bessel_fn.bessel_weights),
        "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
        "max_ell": model.spherical_harmonics._lmax,  # pylint: disable=protected-access
        "interaction_cls": model.interactions[-1].__class__,
        "interaction_cls_first": model.interactions[0].__class__,
        "num_interactions": model.num_interactions.item(),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": o3.Irreps(str(model.products[0].linear.irreps_out)),
        "MLP_irreps": (
            o3.Irreps(str(model.readouts[-1].hidden_irreps))
            if model.num_interactions.item() > 1
            else 1
        ),
        "gate": (
            model.readouts[-1]  # pylint: disable=protected-access
            .non_linearity._modules["acts"][0]
            .f
            if model.num_interactions.item() > 1
            else None
        ),
        "atomic_energies": model.atomic_energies_fn.atomic_energies.cpu().numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers,
        "correlation": len(
            model.products[0].symmetric_contractions.contractions[0].weights
        )
        + 1,
        "radial_type": radial_to_name(
            model.radial_embedding.bessel_fn.__class__.__name__
        ),
        "radial_MLP": model.interactions[0].conv_tp_weights.hs[1:-1],
        "pair_repulsion": hasattr(model, "pair_repulsion_fn"),
        "distance_transform": radial_to_transform(model.radial_embedding),
        "atomic_inter_scale": scale.cpu().numpy(),
        "atomic_inter_shift": shift.cpu().numpy(),
    }
    return config


def extract_load(f: str, map_location: str = "cpu") -> torch.nn.Module:
    return extract_model(
        torch.load(f=f, map_location=map_location), map_location=map_location
    )


def remove_pt_head(
    model: torch.nn.Module, head_to_keep: Optional[str] = None
) -> torch.nn.Module:
    """Converts a multihead MACE model to a single head model by removing the pretraining head.

    Args:
        model (ScaleShiftMACE): The multihead MACE model to convert
        head_to_keep (Optional[str]): The name of the head to keep. If None, keeps the first non-PT head.

    Returns:
        ScaleShiftMACE: A new MACE model with only the specified head

    Raises:
        ValueError: If the model is not a multihead model or if the specified head is not found
    """
    if not hasattr(model, "heads") or len(model.heads) <= 1:
        raise ValueError("Model must be a multihead model with more than one head")

    # Get index of head to keep
    if head_to_keep is None:
        # Find first non-PT head
        try:
            head_idx = next(i for i, h in enumerate(model.heads) if h != "pt_head")
        except StopIteration as e:
            raise ValueError("No non-PT head found in model") from e
    else:
        try:
            head_idx = model.heads.index(head_to_keep)
        except ValueError as e:
            raise ValueError(f"Head {head_to_keep} not found in model") from e

    # Extract config and modify for single head
    model_config = extract_config_mace_model(model)
    model_config["heads"] = [model.heads[head_idx]]
    model_config["atomic_energies"] = (
        model.atomic_energies_fn.atomic_energies[head_idx]
        .unsqueeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    model_config["atomic_inter_scale"] = model.scale_shift.scale[head_idx].item()
    model_config["atomic_inter_shift"] = model.scale_shift.shift[head_idx].item()
    mlp_count_irreps = model_config["MLP_irreps"].count((0, 1)) // len(model.heads)
    model_config["MLP_irreps"] = o3.Irreps(f"{mlp_count_irreps}x0e")

    new_model = model.__class__(**model_config)
    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        if "atomic_energies" in name:
            new_state_dict[name] = param[head_idx : head_idx + 1]
        elif "scale" in name or "shift" in name:
            new_state_dict[name] = param[head_idx : head_idx + 1]
        elif "readouts" in name:
            channels_per_head = param.shape[0] // len(model.heads)
            start_idx = head_idx * channels_per_head
            end_idx = start_idx + channels_per_head
            if "linear_2.weight" in name:
                end_idx = start_idx + channels_per_head // 2
            # if (
            #     "readouts.0.linear.weight" in name
            #     or "readouts.1.linear_2.weight" in name
            # ):
            #     new_state_dict[name] = param[start_idx:end_idx] / (
            #         len(model.heads) ** 0.5
            #     )
            if "readouts.0.linear.weight" in name:
                new_state_dict[name] = param.reshape(-1, len(model.heads))[
                    :, head_idx
                ].flatten()
            elif "readouts.1.linear_1.weight" in name:
                new_state_dict[name] = param.reshape(
                    -1, len(model.heads), mlp_count_irreps
                )[:, head_idx, :].flatten()
            elif "readouts.1.linear_2.weight" in name:
                new_state_dict[name] = param.reshape(
                    len(model.heads), -1, len(model.heads)
                )[head_idx, :, head_idx].flatten() / (len(model.heads) ** 0.5)
            else:
                new_state_dict[name] = param[start_idx:end_idx]

        else:
            new_state_dict[name] = param

    # Load state dict into new model
    new_model.load_state_dict(new_state_dict)

    return new_model


def extract_model(model: torch.nn.Module, map_location: str = "cpu") -> torch.nn.Module:
    model_copy = model.__class__(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    return model_copy.to(map_location)


def convert_to_json_format(dict_input):
    for key, value in dict_input.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            dict_input[key] = value.tolist()
        # # check if the value is a class and convert it to a string
        elif hasattr(value, "__class__"):
            dict_input[key] = str(value)
    return dict_input


def convert_from_json_format(dict_input):
    dict_output = dict_input.copy()
    if (
        dict_input["interaction_cls"]
        == "<class 'mace.modules.blocks.RealAgnosticResidualInteractionBlock'>"
    ):
        dict_output["interaction_cls"] = (
            modules.blocks.RealAgnosticResidualInteractionBlock
        )
    if (
        dict_input["interaction_cls"]
        == "<class 'mace.modules.blocks.RealAgnosticInteractionBlock'>"
    ):
        dict_output["interaction_cls"] = modules.blocks.RealAgnosticInteractionBlock
    if (
        dict_input["interaction_cls_first"]
        == "<class 'mace.modules.blocks.RealAgnosticResidualInteractionBlock'>"
    ):
        dict_output["interaction_cls_first"] = (
            modules.blocks.RealAgnosticResidualInteractionBlock
        )
    if (
        dict_input["interaction_cls_first"]
        == "<class 'mace.modules.blocks.RealAgnosticInteractionBlock'>"
    ):
        dict_output["interaction_cls_first"] = (
            modules.blocks.RealAgnosticInteractionBlock
        )
    dict_output["r_max"] = float(dict_input["r_max"])
    dict_output["num_bessel"] = int(dict_input["num_bessel"])
    dict_output["num_polynomial_cutoff"] = float(dict_input["num_polynomial_cutoff"])
    dict_output["max_ell"] = int(dict_input["max_ell"])
    dict_output["num_interactions"] = int(dict_input["num_interactions"])
    dict_output["num_elements"] = int(dict_input["num_elements"])
    dict_output["hidden_irreps"] = o3.Irreps(dict_input["hidden_irreps"])
    dict_output["MLP_irreps"] = o3.Irreps(dict_input["MLP_irreps"])
    dict_output["avg_num_neighbors"] = float(dict_input["avg_num_neighbors"])
    dict_output["gate"] = torch.nn.functional.silu
    dict_output["atomic_energies"] = np.array(dict_input["atomic_energies"])
    dict_output["atomic_numbers"] = dict_input["atomic_numbers"]
    dict_output["correlation"] = int(dict_input["correlation"])
    dict_output["radial_type"] = dict_input["radial_type"]
    dict_output["radial_MLP"] = ast.literal_eval(dict_input["radial_MLP"])
    dict_output["pair_repulsion"] = ast.literal_eval(dict_input["pair_repulsion"])
    dict_output["distance_transform"] = dict_input["distance_transform"]
    dict_output["atomic_inter_scale"] = float(dict_input["atomic_inter_scale"])
    dict_output["atomic_inter_shift"] = float(dict_input["atomic_inter_shift"])

    return dict_output


def load_from_json(f: str, map_location: str = "cpu") -> torch.nn.Module:
    extra_files_extract = {"commit.txt": None, "config.json": None}
    model_jit_load = torch.jit.load(
        f, _extra_files=extra_files_extract, map_location=map_location
    )
    model_load_yaml = modules.ScaleShiftMACE(
        **convert_from_json_format(json.loads(extra_files_extract["config.json"]))
    )
    model_load_yaml.load_state_dict(model_jit_load.state_dict())
    return model_load_yaml.to(map_location)


def get_atomic_energies(E0s, train_collection, z_table) -> dict:
    if E0s is not None:
        logging.info(
            "Isolated Atomic Energies (E0s) not in training file, using command line argument"
        )
        if E0s.lower() == "average":
            logging.info(
                "Computing average Atomic Energies using least squares regression"
            )
            # catch if colections.train not defined above
            try:
                assert train_collection is not None
                atomic_energies_dict = data.compute_average_E0s(
                    train_collection, z_table
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not compute average E0s if no training xyz given, error {e} occured"
                ) from e
        else:
            if E0s.endswith(".json"):
                logging.info(f"Loading atomic energies from {E0s}")
                with open(E0s, "r", encoding="utf-8") as f:
                    atomic_energies_dict = json.load(f)
                    atomic_energies_dict = {
                        int(key): value for key, value in atomic_energies_dict.items()
                    }
            else:
                try:
                    atomic_energies_eval = ast.literal_eval(E0s)
                    if not all(
                        isinstance(value, dict)
                        for value in atomic_energies_eval.values()
                    ):
                        atomic_energies_dict = atomic_energies_eval
                    else:
                        atomic_energies_dict = atomic_energies_eval
                    assert isinstance(atomic_energies_dict, dict)
                except Exception as e:
                    raise RuntimeError(
                        f"E0s specified invalidly, error {e} occured"
                    ) from e
    else:
        raise RuntimeError(
            "E0s not found in training file and not specified in command line"
        )
    return atomic_energies_dict


def get_avg_num_neighbors(head_configs, args, train_loader, device):
    if all(head_config.compute_avg_num_neighbors for head_config in head_configs):
        logging.info("Computing average number of neighbors")
        avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        if args.distributed:
            num_graphs = torch.tensor(len(train_loader.dataset)).to(device)
            num_neighbors = num_graphs * torch.tensor(avg_num_neighbors).to(device)
            torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_neighbors, op=torch.distributed.ReduceOp.SUM
            )
            avg_num_neighbors_out = (num_neighbors / num_graphs).item()
        else:
            avg_num_neighbors_out = avg_num_neighbors
    else:
        assert any(
            head_config.avg_num_neighbors is not None for head_config in head_configs
        ), "Average number of neighbors must be provided in the configuration"
        avg_num_neighbors_out = max(
            head_config.avg_num_neighbors
            for head_config in head_configs
            if head_config.avg_num_neighbors is not None
        )
    if avg_num_neighbors_out < 2 or avg_num_neighbors_out > 100:
        logging.warning(
            f"Unusual average number of neighbors: {avg_num_neighbors_out:.1f}"
        )
    else:
        logging.info(f"Average number of neighbors: {avg_num_neighbors_out}")
    return avg_num_neighbors_out


def get_loss_fn(
    args: argparse.Namespace,
    dipole_only: bool,
    compute_dipole: bool,
) -> torch.nn.Module:
    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    elif args.loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            virials_weight=args.virials_weight,
        )
    elif args.loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
        )
    elif args.loss == "huber":
        loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args.loss == "universal":
        loss_fn = modules.UniversalLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args.loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    return loss_fn


def get_swa(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    swas: List[bool],
    dipole_only: bool = False,
):
    assert dipole_only is False, "Stage Two for dipole fitting not implemented"
    swas.append(True)
    if args.start_swa is None:
        args.start_swa = max(1, args.max_num_epochs // 4 * 3)
    else:
        if args.start_swa >= args.max_num_epochs:
            logging.warning(
                f"Start Stage Two must be less than max_num_epochs, got {args.start_swa} > {args.max_num_epochs}"
            )
            swas[-1] = False
    if args.loss == "forces_only":
        raise ValueError("Can not select Stage Two with forces only loss.")
    if args.loss == "virials":
        loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
            virials_weight=args.swa_virials_weight,
        )
        logging.info(
            f"Stage Two (after {args.start_swa} epochs) with loss function: {loss_fn_energy}, energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight},  virials_weight: {args.swa_virials_weight} and learning rate : {args.swa_lr}"
        )
    elif args.loss == "stress":
        loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
            stress_weight=args.swa_stress_weight,
        )
        logging.info(
            f"Stage Two (after {args.start_swa} epochs) with loss function: {loss_fn_energy}, energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, stress weight : {args.stress_weight} and learning rate : {args.swa_lr}"
        )
    elif args.loss == "energy_forces_dipole":
        loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
            args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
            dipole_weight=args.swa_dipole_weight,
        )
        logging.info(
            f"Stage Two (after {args.start_swa} epochs) with loss function: {loss_fn_energy}, with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
        )
    elif args.loss == "universal":
        loss_fn_energy = modules.UniversalLoss(
            energy_weight=args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
            stress_weight=args.swa_stress_weight,
            huber_delta=args.huber_delta,
        )
        logging.info(
            f"Stage Two (after {args.start_swa} epochs) with loss function: {loss_fn_energy}, with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, stress weight : {args.swa_stress_weight} and learning rate : {args.swa_lr}"
        )
    else:
        loss_fn_energy = modules.WeightedEnergyForcesLoss(
            energy_weight=args.swa_energy_weight,
            forces_weight=args.swa_forces_weight,
        )
        logging.info(
            f"Stage Two (after {args.start_swa} epochs) with loss function: {loss_fn_energy}, with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
        )
    swa = SWAContainer(
        model=AveragedModel(model),
        scheduler=SWALR(
            optimizer=optimizer,
            swa_lr=args.swa_lr,
            anneal_epochs=1,
            anneal_strategy="linear",
        ),
        start=args.start_swa,
        loss_fn=loss_fn_energy,
    )
    return swa, swas


def get_params_options(
    args: argparse.Namespace, model: torch.nn.Module
) -> Dict[str, Any]:
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.node_embedding.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": args.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
        amsgrad=args.amsgrad,
        betas=(args.beta, 0.999),
    )
    return param_options


def get_optimizer(
    args: argparse.Namespace, param_options: Dict[str, Any]
) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    elif args.optimizer == "schedulefree":
        try:
            from schedulefree import adamw_schedulefree
        except ImportError as exc:
            raise ImportError(
                "`schedulefree` is not installed. Please install it via `pip install schedulefree` or `pip install mace-torch[schedulefree]`"
            ) from exc
        _param_options = {k: v for k, v in param_options.items() if k != "amsgrad"}
        optimizer = adamw_schedulefree.AdamWScheduleFree(**_param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)
    return optimizer


def setup_wandb(args: argparse.Namespace):
    logging.info("Using Weights and Biases for logging")
    import wandb

    wandb_config = {}
    args_dict = vars(args)

    for key, value in args_dict.items():
        if isinstance(value, np.ndarray):
            args_dict[key] = value.tolist()

    args_dict_json = json.dumps(args_dict)
    for key in args.wandb_log_hypers:
        wandb_config[key] = args_dict[key]
    tools.init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=wandb_config,
        directory=args.wandb_dir,
    )
    wandb.run.summary["params"] = args_dict_json


def get_files_with_suffix(dir_path: str, suffix: str) -> List[str]:
    return [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(suffix)
    ]


def dict_to_array(input_data, heads):
    if all(isinstance(value, np.ndarray) for value in input_data.values()):
        return np.array([input_data[head] for head in heads])
    if not all(isinstance(value, dict) for value in input_data.values()):
        return np.array([[input_data[head]] for head in heads])
    unique_keys = set()
    for inner_dict in input_data.values():
        unique_keys.update(inner_dict.keys())
    unique_keys = list(unique_keys)
    sorted_keys = sorted([int(key) for key in unique_keys])
    result_array = np.zeros((len(input_data), len(sorted_keys)))
    for _, (head_name, inner_dict) in enumerate(input_data.items()):
        for key, value in inner_dict.items():
            key_index = sorted_keys.index(int(key))
            head_index = heads.index(head_name)
            result_array[head_index][key_index] = value
    return result_array


class LRScheduler:
    def __init__(self, optimizer, args) -> None:
        self.scheduler = args.scheduler
        self._optimizer_type = (
            args.optimizer
        )  # Schedulefree does not need an optimizer but checkpoint handler does.
        if args.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=args.lr_scheduler_gamma
            )
        elif args.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=args.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    def step(self, metrics=None, epoch=None):  # pylint: disable=E1123
        if self._optimizer_type == "schedulefree":
            return  # In principle, schedulefree optimizer can be used with a scheduler but the paper suggests it's not necessary
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step(  # pylint: disable=E1123
                metrics=metrics, epoch=epoch
            )

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)


def check_folder_subfolder(folder_path):
    entries = os.listdir(folder_path)
    for entry in entries:
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            return True
    return False


def check_path_ase_read(filename: str) -> str:
    filepath = Path(filename)
    if filepath.is_dir():
        if len(list(filepath.glob("*.h5")) + list(filepath.glob("*.hdf5"))) == 0:
            raise RuntimeError(f"Got directory {filename} with no .h5/.hdf5 files")
        return False
    if filepath.suffix in (".h5", ".hdf5"):
        return False
    return True


def dict_to_namespace(dictionary):
    # Convert the dictionary into an argparse.Namespace
    namespace = argparse.Namespace()
    for key, value in dictionary.items():
        setattr(namespace, key, value)
    return namespace
