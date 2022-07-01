import ast
import dataclasses
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn.functional
from e3nn import o3
from prettytable import PrettyTable
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data, modules, tools
from mace.tools import torch_geometric


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset_from_xyz(
    train_path: str,
    valid_path: str,
    valid_fraction: float,
    config_type_weights: Dict,
    test_path: str = None,
    seed: int = 1234,
    energy_key: str = "energy",
    forces_key: str = "forces",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from xyz file"""
    atomic_energies_dict, all_train_configs = data.load_from_xyz(
        file_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        extract_atomic_energies=True,
    )
    logging.info(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )
    if valid_path is not None:
        _, valid_configs = data.load_from_xyz(
            file_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
        )
        logging.info(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    else:
        logging.info(
            "Using random %s%% of training set for validation", 100 * valid_fraction
        )
        train_configs, valid_configs = data.random_train_valid_split(
            all_train_configs, valid_fraction, seed
        )

    test_configs = []
    if test_path is not None:
        _, all_test_configs = data.load_from_xyz(
            file_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
        )
        # create list of tuples (config_type, list(Atoms))
        test_configs = data.test_config_types(all_test_configs)
        logging.info(
            f"Loaded {len(all_test_configs)} test configurations from '{train_path}'"
        )
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=test_configs),
        atomic_energies_dict,
    )


def main() -> None:
    args = tools.build_default_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=ast.literal_eval(args.config_type_weights),
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
    )

    logging.info(
        f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
        f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
    )

    # Atomic number table
    # yapf: disable
    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    # yapf: enable
    logging.info(z_table)
    if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
        if args.E0s is not None:
            logging.info(
                "Atomic Energies not in training file, using command line argument E0s"
            )
            atomic_energies_dict = ast.literal_eval(args.E0s)
        else:
            raise RuntimeError(
                "E0s not found in training file and not specified in command line"
            )
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.valid
        ],
        batch_size=args.valid_batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn: torch.nn.Module
    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    else:
        loss_fn = modules.EnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors:.3f}")

    # Build model
    logging.info("Building model")
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
        if args.scaling == "no_scaling":
            std = 1.0
            logging.info("No scaling selected")
        else:
            mean, std = modules.scaling_classes[args.scaling](
                train_loader, atomic_energies
            )
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            device=args.device,
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
        )
    elif args.model == "ScaleShiftMACE":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            device=args.device,
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "ScaleShiftBOTNet":
        mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
        model = modules.ScaleShiftBOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
    elif args.model == "BOTNet":
        model = modules.BOTNet(
            **model_config,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    else:
        raise RuntimeError(f"Unknown model: '{args.model}'")

    model.to(device)

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
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)

    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    if args.scheduler == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=args.lr_scheduler_gamma
        )
    elif args.scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=args.lr_factor,
            patience=args.scheduler_patience,
        )
    else:
        raise RuntimeError(f"Unknown scheduler: '{args.scheduler}'")

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints
    )

    start_epoch = 0
    if args.restart_latest:
        opt_start_epoch = checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
        )
        if opt_start_epoch is not None:
            start_epoch = opt_start_epoch

    swa: Optional[tools.SWAContainer] = None
    if args.swa:
        if args.loss == "forces_only":
            logging.info("Can not select swa with forces only loss.")
        loss_fn_energy = modules.WeightedEnergyForcesLoss(
            energy_weight=args.swa_energy_weight, forces_weight=args.swa_forces_weight
        )
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
        logging.info(
            f"Using stochastic weight averaging (after {swa.start} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
        )

    ema: Optional[ExponentialMovingAverage] = None
    if args.ema:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")

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
        device=device,
        swa=swa,
        ema=ema,
        max_grad_norm=args.clip_grad,
    )

    epoch = checkpoint_handler.load_latest(
        state=tools.CheckpointState(model, optimizer, lr_scheduler), device=device
    )
    logging.info(f"Loaded model from epoch {epoch}")

    # Evaluation on test datasets
    logging.info("Computing metrics for training, validation, and test sets")

    table = PrettyTable()
    table.field_names = ["config_type", "rmse E / meV", "rmse F / meV / A"]
    for name, subset in [
        ("train", collections.train),
        ("valid", collections.valid),
    ] + collections.tests:
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
                for config in subset
            ],
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=False,
        )

        logging.info(f"Evaluating {name} ...")
        _, metrics = tools.evaluate(
            model, loss_fn=loss_fn, data_loader=data_loader, device=device
        )
        table.add_row(
            [name, f"{metrics['rmse_e'] * 1000:.1f}", f"{metrics['rmse_f'] * 1000:.1f}"]
        )

    logging.info("\n" + str(table))

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + ".model")
    logging.info(f"Saving model to {model_path}")
    if args.save_cpu:
        model = model.to("cpu")
    torch.save(model, model_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
