###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import ast
import glob
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch.distributed
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, modules, tools
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations
from mace.tools.scripts_utils import (
    LRScheduler,
    convert_to_json_format,
    create_error_table,
    extract_config_mace_model,
    get_atomic_energies,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    print_git_commit,
)
from mace.tools.slurm_distributed import DistributedEnvironment
from mace.tools.utils import AtomicNumberTable


def main() -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args = tools.build_default_arg_parser().parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    """
    This script runs the training/fine tuning for mace
    """
    args, input_log_messages = tools.check_args(args)
    tag = tools.get_tag(name=args.name, seed=args.seed)
    if args.distributed:
        try:
            distr_env = DistributedEnvironment()
        except Exception as e:  # pylint: disable=W0703
            logging.error(f"Failed to initialize distributed environment: {e}")
            return
        world_size = distr_env.world_size
        local_rank = distr_env.local_rank
        rank = distr_env.rank
        if rank == 0:
            print(distr_env)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = int(0)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir, rank=rank)
    logging.info("===========VERIFYING SETTINGS===========")
    for message, loglevel in input_log_messages:
        logging.log(level=loglevel, msg=message)

    if args.distributed:
        torch.cuda.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.debug(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)
    commit = print_git_commit()
    if args.foundation_model is not None:
        if args.foundation_model in ["small", "medium", "large"]:
            logging.info(
                f"Using foundation model mace-mp-0 {args.foundation_model} as initial checkpoint."
            )
            calc = mace_mp(
                model=args.foundation_model,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        elif args.foundation_model in ["small_off", "medium_off", "large_off"]:
            model_type = args.foundation_model.split("_")[0]
            logging.info(
                f"Using foundation model mace-off-2023 {model_type} as initial checkpoint. ASL license."
            )
            calc = mace_off(
                model=model_type,
                device=args.device,
                default_dtype=args.default_dtype,
            )
            model_foundation = calc.models[0]
        else:
            model_foundation = torch.load(args.foundation_model, map_location=device)
            logging.info(
                f"Using foundation model {args.foundation_model} as initial checkpoint."
            )
        args.r_max = model_foundation.r_max.item()

    if args.statistics_file is not None:
        with open(args.statistics_file, "r") as f:  # pylint: disable=W1514
            statistics = json.load(f)
        logging.info("Using statistics json file")
        args.r_max = (
            statistics["r_max"] if args.foundation_model is None else args.r_max
        )
        args.atomic_numbers = statistics["atomic_numbers"]
        args.mean = statistics["mean"]
        args.std = statistics["std"]
        args.avg_num_neighbors = statistics["avg_num_neighbors"]
        args.compute_avg_num_neighbors = False
        args.E0s = statistics["atomic_energies"]

    logging.info("")
    logging.info("===========LOADING INPUT DATA===========")
    # Data preparation
    if args.train_file.endswith(".xyz"):
        if args.valid_file is not None:
            assert args.valid_file.endswith(
                ".xyz"
            ), "valid_file if given must be same format as train_file"
        config_type_weights = get_config_type_weights(args.config_type_weights)
        collections, atomic_energies_dict = get_dataset_from_xyz(
            work_dir=args.work_dir,
            train_path=args.train_file,
            valid_path=args.valid_file,
            valid_fraction=args.valid_fraction,
            config_type_weights=config_type_weights,
            test_path=args.test_file,
            seed=args.seed,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            stress_key=args.stress_key,
            virials_key=args.virials_key,
            dipole_key=args.dipole_key,
            charges_key=args.charges_key,
            keep_isolated_atoms=args.keep_isolated_atoms,
        )
        if len(collections.train) < args.batch_size:
            logging.error(
                f"Batch size ({args.batch_size}) is larger than the number of training data ({len(collections.train)})"
            )
        if len(collections.valid) < args.valid_batch_size:
            logging.warning(
                f"Validation batch size ({args.valid_batch_size}) is larger than the number of validation data ({len(collections.valid)})"
            )
            args.valid_batch_size = len(collections.valid)

    else:
        atomic_energies_dict = None

    # Atomic number table
    # yapf: disable
    if args.atomic_numbers is None:
        assert args.train_file.endswith(".xyz"), "Must specify atomic_numbers when using .h5 train_file input"
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        if args.statistics_file is None:
            logging.info("Using atomic numbers from command line argument")
        else:
            logging.info("Using atomic numbers from statistics file")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)
    # yapf: enable
    logging.info(f"Atomic Numbers used: {z_table.zs}")

    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s.lower() == "foundation":
            assert args.foundation_model is not None
            z_table_foundation = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            atomic_energies_dict = {
                z: model_foundation.atomic_energies_fn.atomic_energies[
                    z_table_foundation.z_to_index(z)
                ].item()
                for z in z_table.zs
            }
            logging.info(
                f"Using Atomic Energies from foundation model [z, eV]: {', '.join([f'{z}: {atomic_energies_dict[z]}' for z in z_table_foundation.zs])}"
            )
        else:
            if args.train_file.endswith(".xyz"):
                atomic_energies_dict = get_atomic_energies(
                    args.E0s, collections.train, z_table
                )
            else:
                atomic_energies_dict = get_atomic_energies(args.E0s, None, z_table)

    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        args.compute_forces = False
        compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            args.compute_forces = True
            compute_virials = False
            args.compute_stress = False
        else:
            compute_energy = True
            compute_dipole = False
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(
            f"Atomic Energies used (z: eV): {{{', '.join([f'{z}: {atomic_energies_dict[z]}' for z in z_table.zs])}}}"
        )

    if args.train_file.endswith(".xyz"):
        train_set = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ]
        valid_set = [
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ]
    elif args.train_file.endswith(".h5"):
        train_set = data.HDF5Dataset(args.train_file, r_max=args.r_max, z_table=z_table)
        valid_set = data.HDF5Dataset(args.valid_file, r_max=args.r_max, z_table=z_table)
    else:  # This case would be for when the file path is to a directory of multiple .h5 files
        train_set = data.dataset_from_sharded_hdf5(
            args.train_file, r_max=args.r_max, z_table=z_table
        )
        valid_set = data.dataset_from_sharded_hdf5(
            args.valid_file, r_max=args.r_max, z_table=z_table
        )

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=(train_sampler is None),
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=valid_set,
        batch_size=args.valid_batch_size,
        sampler=valid_sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    logging.info("")
    logging.info("===========MODEL DETAILS===========")
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
        # Unweighted Energy and Forces loss by default
        loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)

    if args.compute_avg_num_neighbors:
        avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        if args.distributed:
            num_graphs = torch.tensor(len(train_loader.dataset)).to(device)
            num_neighbors = num_graphs * torch.tensor(avg_num_neighbors).to(device)
            torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_neighbors, op=torch.distributed.ReduceOp.SUM
            )
            args.avg_num_neighbors = (num_neighbors / num_graphs).item()
        else:
            args.avg_num_neighbors = avg_num_neighbors
    if args.avg_num_neighbors < 2 or args.avg_num_neighbors > 100:
        logging.warning(
            f"Unusual average number of neighbors: {args.avg_num_neighbors:.1f}"
        )
    else:
        logging.info(f"Average number of neighbors: {args.avg_num_neighbors:.1f}")

    # Selecting outputs
    compute_virials = False
    if args.loss in ("stress", "virials", "huber", "universal"):
        compute_virials = True
        args.compute_stress = True
        if "MAE" in args.error_table:
            args.error_table = "PerAtomMAEstressvirials"
        else:
            args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }

    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )

    if args.scaling == "no_scaling":
        args.std = 1.0
        logging.info("No scaling selected")
    elif (args.mean is None or args.std is None) and args.model != "AtomicDipolesMACE":
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
    # Build model
    if args.foundation_model is not None and args.model in ["MACE", "ScaleShiftMACE"]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_numbers"] = z_table.zs
        model_config_foundation["num_elements"] = len(z_table)
        args.max_L = model_config_foundation["hidden_irreps"].lmax
        args.num_channels = list(
            {irrep.mul for irrep in o3.Irreps(model_config_foundation["hidden_irreps"])}
        )[0]
        model_config_foundation["atomic_inter_shift"] = (
            model_foundation.scale_shift.shift.item()
        )
        model_config_foundation["atomic_inter_scale"] = (
            model_foundation.scale_shift.scale.item()
        )
        model_config_foundation["atomic_energies"] = atomic_energies
        args.model = "FoundationMACE"
        model_config = model_config_foundation  # pylint
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} Å (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} Å)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} Å (total receptive field for each atom: {args.r_max * args.num_interactions} Å)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )
        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
        )

    model: torch.nn.Module

    if args.model == "MACE":
        model = modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=0.0,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "ScaleShiftMACE":
        model = modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    elif args.model == "FoundationMACE":
        model = modules.ScaleShiftMACE(**model_config_foundation)
    elif args.model == "ScaleShiftBOTNet":
        model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
        )
    elif args.model == "BOTNet":
        model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    elif args.model == "AtomicDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        model = modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            # dipole_scale=1,
            # dipole_shift=0,
        )
    elif args.model == "EnergyDipolesMACE":
        # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        model = modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    if args.foundation_model is not None:
        model = load_foundations(
            model,
            model_foundation,
            z_table,
            load_readout=True,
            max_L=args.max_L,
        )
    model.to(device)

    logging.debug(model)
    logging.info(f"Total number of parameters: {tools.count_parameters(model)}")
    logging.info("")
    logging.info("===========OPTIMIZER INFORMATION===========")
    logging.info(f"Using {args.optimizer.upper()} as parameter optimizer")
    logging.info(f"Batch size: {args.batch_size}")
    if args.ema:
        logging.info(f"Using Exponential Moving Average with decay: {args.ema_decay}")
    logging.info(
        f"Number of gradient updates: {int(args.max_num_epochs*len(collections.train)/args.batch_size)}"
    )
    logging.info(f"Learning rate: {args.lr}, weight decay: {args.weight_decay}")
    logging.info(loss_fn)

    # Optimizer
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

    optimizer: torch.optim.Optimizer
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

    logger = tools.MetricsLogger(
        directory=args.results_dir, tag=tag + "_train"
    )  # pylint: disable=E1123

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        assert dipole_only is False, "Stage Two for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = max(1, args.max_num_epochs // 4 * 3)
        logging.info(
            f"Stage Two will start after {args.start_swa} epochs with loss function:"
        )
        if args.loss == "forces_only":
            raise ValueError("Can not select Stage Two with forces only loss.")
        if args.loss == "virials":
            loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                virials_weight=args.swa_virials_weight,
            )
        elif args.loss == "stress":
            loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                stress_weight=args.swa_stress_weight,
            )
        elif args.loss == "energy_forces_dipole":
            loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
                args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
                dipole_weight=args.swa_dipole_weight,
            )
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
        logging.info(loss_fn_energy)
        swa = tools.SWAContainer(
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

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    if args.restart_latest:
        try:
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=True,
                device=device,
            )
        except Exception:  # pylint: disable=W0703
            opt_start_epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(model, optimizer, lr_scheduler),
                swa=False,
                device=device,
            )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    else:
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    if args.wandb:
        logging.info("Using Weights and Biases for logging")
        import wandb

        wandb_config = {}
        args_dict = vars(args)
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

    if args.distributed:
        distributed_model = DDP(model, device_ids=[local_rank])
    else:
        distributed_model = None

    tools.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        save_all_checkpoints=args.save_all_checkpoints,
        output_args=output_args,
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
        log_errors=args.error_table,
        log_wandb=args.wandb,
        distributed=args.distributed,
        distributed_model=distributed_model,
        train_sampler=train_sampler,
        rank=rank,
    )
    logging.info("")
    logging.info("===========RESULTS===========")
    logging.info("Computing metrics for training, validation, and test sets")

    all_data_loaders = {
        "train": train_loader,
        "valid": valid_loader,
    }

    test_sets = {}
    if args.train_file.endswith(".xyz"):
        for name, subset in collections.tests:
            test_sets[name] = [
                data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
                for config in subset
            ]
    elif not args.multi_processed_test:
        test_files = get_files_with_suffix(args.test_dir, "_test.h5")
        for test_file in test_files:
            name = os.path.splitext(os.path.basename(test_file))[0]
            test_sets[name] = data.HDF5Dataset(
                test_file, r_max=args.r_max, z_table=z_table
            )
    else:
        test_folders = glob(args.test_dir + "/*")
        for folder in test_folders:
            name = os.path.splitext(os.path.basename(test_file))[0]
            test_sets[name] = data.dataset_from_sharded_hdf5(
                folder, r_max=args.r_max, z_table=z_table
            )

    for test_name, test_set in test_sets.items():
        test_sampler = None
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
        try:
            drop_last = test_set.drop_last
        except AttributeError as e:  # pylint: disable=W0612
            drop_last = False
        test_loader = torch_geometric.dataloader.DataLoader(
            test_set,
            batch_size=args.valid_batch_size,
            shuffle=(test_sampler is None),
            drop_last=drop_last,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        all_data_loaders[test_name] = test_loader

    train_valid_data_loader = {
        k: v for k, v in all_data_loaders.items() if k in ["train", "valid"]
    }
    test_data_loader = {
        k: v for k, v in all_data_loaders.items() if k not in ["train", "valid"]
    }

    for swa_eval in swas:
        epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=swa_eval,
            device=device,
        )
        model.to(device)
        if args.distributed:
            distributed_model = DDP(model, device_ids=[local_rank])
        model_to_evaluate = model if not args.distributed else distributed_model
        if swa_eval:
            logging.info(f"Loaded Stage two model from epoch {epoch} for evaluation")
        else:
            logging.info(f"Loaded Stage one model from epoch {epoch} for evaluation")

        for param in model.parameters():
            param.requires_grad = False

        table_train = create_error_table(
            table_type=args.error_table,
            all_data_loaders=train_valid_data_loader,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        table_test = create_error_table(
            table_type=args.error_table,
            all_data_loaders=test_data_loader,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        logging.info("Error-table on TRAIN and VALID:\n" + str(table_train))
        logging.info("Error-table on TEST:\n" + str(table_test))

        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_stagetwo.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            if args.save_cpu:
                model = model.to("cpu")
            torch.save(model, model_path)
            extra_files = {
                "commit.txt": commit.encode("utf-8") if commit is not None else b"",
                "config.yaml": json.dumps(
                    convert_to_json_format(extract_config_mace_model(model))
                ),
            }
            if swa_eval:
                torch.save(
                    model, Path(args.model_dir) / (args.name + "_stagetwo.model")
                )
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_stagetwo_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata {path_complied}")
                    model_compiled = jit.compile(deepcopy(model))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    pass
            else:
                torch.save(model, Path(args.model_dir) / (args.name + ".model"))
                try:
                    path_complied = Path(args.model_dir) / (
                        args.name + "_compiled.model"
                    )
                    logging.info(f"Compiling model, saving metadata to {path_complied}")
                    model_compiled = jit.compile(deepcopy(model))
                    torch.jit.save(
                        model_compiled,
                        path_complied,
                        _extra_files=extra_files,
                    )
                except Exception as e:  # pylint: disable=W0703
                    pass

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
