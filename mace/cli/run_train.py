###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import ast
import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional
import urllib.request
import itertools

import ase
import numpy as np
import torch.distributed
import torch.nn.functional
from e3nn import o3
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

from torch.utils.data import random_split

import mace
from mace import data, modules, tools
from mace.calculators.foundations_models import mace_mp, mace_off
from mace.cli.fine_tuning_select import select_samples
from mace.tools import torch_geometric
from mace.tools import create_memory_tracked_loader
from mace.tools.scripts_utils import (
    LRScheduler,
    create_error_table,
    dict_to_namespace,
    get_atomic_energies,
    get_config_type_weights,
    get_dataset_from_xyz,
    get_files_with_suffix,
    dict_to_array,
    check_folder_subfolder,
)
from mace.tools.slurm_distributed import DistributedEnvironment, AsyncDistributedEnvironment
from mace.tools.finetuning_utils import (
    load_foundations_elements,
    extract_config_mace_model,
)
from mace.tools.utils import AtomicNumberTable
from torch.utils.data import ConcatDataset
from box import Box
from tqdm import tqdm
import multiprocessing as mp
from tqdm import tqdm

import time

def get_loss(dataset, model_path="/lustre/fsn1/projects/rech/gax/unh55hx/mace_multi_head_interface_bk/checkpoints/RASimpleDensityResidualIntBlockx2_run-123_epoch-35.model"):
    from torch_scatter import scatter
    # Load the model
    model = torch.load(model_path, map_location=f'cuda')
    model.eval()  # Set the model to evaluation mode

    # Create DataLoader with DistributedSampler
    dataloader = torch_geometric.dataloader.DataLoader(dataset, batch_size=128, num_workers=12, shuffle=False)

    device = torch.device(f'cuda')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False


    #label_l1, pred_l1, delta_l1 = [], [], []
    label_l2, pred_l2, delta_l2 = [], [], []
    dataloader_iter = tqdm(dataloader, desc=f"Processing batches ", unit="batch")

    for batch in dataloader_iter:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        
        # Ensure positions require gradients
        batch_dict['positions'].requires_grad_(True)
        
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        
        # Calculate force delta
        pred_force = output['forces'].detach()
        label_force = batch.forces.detach()
        delta = (label_force - pred_force).detach()
        
        # Scatter across batch
        batch_index = batch_dict['batch']
        num_graphs = int(batch_index.max()) + 1
        
        ## L1 norm calculations
        #label_l1.append(scatter(torch.norm(label_force, dim=-1, p=1), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())
        #pred_l1.append(scatter(torch.norm(pred_force, dim=-1, p=1), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())
        #delta_l1.append(scatter(torch.norm(delta, dim=-1, p=1), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())

        # L2 norm calculations
        label_l2.append(scatter(torch.norm(label_force, dim=-1, p=2), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())
        pred_l2.append(scatter(torch.norm(pred_force, dim=-1, p=2), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())
        delta_l2.append(scatter(torch.norm(delta, dim=-1, p=2), batch_index, dim=0, dim_size=num_graphs, reduce='mean').cpu())
        
    
    #local_label_l1 = torch.cat(label_l1, dim=0)
    #local_pred_l1 = torch.cat(pred_l1, dim=0)
    #local_delta_l1 = torch.cat(delta_l1, dim=0)
    local_label_l2 = torch.cat(label_l2, dim=0)
    local_pred_l2 = torch.cat(pred_l2, dim=0)
    local_delta_l2 = torch.cat(delta_l2, dim=0)

    return local_label_l2, local_pred_l2, local_delta_l2

def bad_force(data):
    forces = data["forces"]
    forces_norm_max = forces.norm(dim=-1).max().item()
    if forces_norm_max > 300.0:
        return True
    else:
        return False

def is_stable(data):
    forces = data["forces"]
    forces_norm_max = forces.norm(dim=-1, p=2).max().item()
    if forces_norm_max < 0.005:
        return True
    else:
        return False

def bad_iso(data):
    forces = data["forces"]
    n_atoms = forces.size(0)

    if n_atoms > 1:
        return False
    else:
        return True

def bad_energy(data):
    energy = data["energy"].item()
    forces = data["forces"]
    n_atoms = forces.size(0)
    e_per_atom = energy / n_atoms

    if -20.0 > e_per_atom or e_per_atom > 2.0:
        return True
    else:
        return False

def bad_stress(data):
    stress = data['stress']
    stress_norm = stress.abs().max().item()
    if stress_norm > 1.0:
        return True
    else:
        return False


def is_bad(data):
    """Check if a sample is 'bad' based on force, energy, and stress."""
    return bad_force(data) or bad_energy(data) or bad_stress(data) or bad_iso(data)


def filter_data(train_set):
    """Filter out 'bad' samples from the dataset using multiprocessing."""
    with mp.Pool(4) as pool:
        mask = list(tqdm(pool.imap(is_bad, train_set), total=len(train_set)))

    # Return the indices of the good data points (where mask is False)
    return [i for i, bad in enumerate(mask) if not bad]


def only_stable(train_set, world_size, rank):
    """Filter out 'bad' samples from the dataset using multiprocessing with ProcessPoolExecutor."""
    #sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False)
    #dataloader = torch_geometric.dataloader.DataLoader(train_set, batch_size=1, sampler=sampler)

    import math
    dataset_size = len(train_set)
    chunk_size = math.ceil(dataset_size / world_size)

    start_idx = rank * chunk_size
    end_idx = min((rank + 1) * chunk_size, dataset_size)
    indices = list(range(start_idx, end_idx))
    
    train_set = torch.utils.data.Subset(train_set, indices)
    if rank == 0:
        data_iter = tqdm(train_set)
    else:
        data_iter = train_set

    mask = [is_stable(data) for data in data_iter]

    all_masks = [None] * world_size
    torch.distributed.all_gather_object(all_masks, mask)
    global_mask = [item for sublist in all_masks for item in sublist]

    # Return the indices of the good data points (where mask is False)
    return [i for i, m in enumerate(global_mask) if m]

def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

def main() -> None:
    args = tools.build_default_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    if args.device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            raise ImportError(
                "Error: Intel extension for PyTorch not found, but XPU device was specified"
            )
    if args.distributed:
        try:
            if args.async_start:
                distr_env = AsyncDistributedEnvironment()
            else:
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

    if args.distributed:
        torch.cuda.set_device(local_rank)
        logging.info(f"Process group initialized: {torch.distributed.is_initialized()}")
        logging.info(f"Processes: {world_size}")

    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")

    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)

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
            model_foundation = mace_off(
                model=model_type,
                device=args.device,
                default_dtype=args.default_dtype,
                return_raw_model=True,
            )
        else:
            model_foundation = torch.load(
                args.foundation_model, map_location=args.device
            )
            logging.info(
                f"Using foundation model {args.foundation_model} as initial checkpoint."
            )
        args.r_max = model_foundation.r_max.item()
    else:
        args.multiheads_finetuning = False

    if args.heads is not None:
        args.heads = Box(ast.literal_eval(args.heads)) # using box container for both dict and namespace access

    if args.r_max == "covalent_radii":
        radii_dict = dict()
        scale = 0.5 * args.r_max_scale
        covalent_radii = torch.tensor(ase.data.covalent_radii)
        print((ase.data.covalent_radii * scale).tolist())
        ne = covalent_radii.size(0)
        r_max_matrix = (covalent_radii.repeat(ne, 1) + covalent_radii.repeat(ne, 1).t()).cpu().numpy() * scale 
        r_max_dict = {}
        for i in range(ne):
            for j in range(ne):
                r_max_dict[(i, j)] = r_max_matrix[i][j]
        args.r_max_matrix = r_max_matrix
        args.r_max_dict = r_max_dict

    for head, head_args in args.heads.items():
        logging.info(f"=============    Processing head {head}     ===========")
        
        if 'statistics_file' in head_args:
            with open(head_args.statistics_file, "r") as f:
                statistics = json.load(f)
            logging.info("Using statistics json file")
            
            # eval the string values
            statistics = {k: ast.literal_eval(v) if isinstance(v, str) else v for k,v in statistics.items()} 
            
            head_args.r_max = (
                statistics["r_max"] if args.foundation_model is None else args.r_max
            )
            head_args.atomic_numbers = statistics["atomic_numbers"]
            head_args.mean = statistics["mean"]
            head_args.std = statistics["std"]
            if head_args.r_max == statistics["r_max"]:
                head_args.avg_num_neighbors = statistics["avg_num_neighbors"]
                head_args.compute_avg_num_neighbors = False
            else:
                head_args.avg_num_neighbors = 0
                head_args.compute_avg_num_neighbors = True
            
            if 'E0s' not in head_args: # overide by E0s
                head_args.E0s = (
                    statistics.get("atomic_energies", None) # gets override if provided directly form json
                )
        
        if 'E0s' in head_args:
            if head_args.E0s.endswith(".json"):
                with open(head_args.E0s, "r") as f:
                    E0s_json = json.load(f)
                    assert head in E0s_json, "headname should be contained in json as a key"
                    head_args.E0s = ast.literal_eval(E0s_json[head])
            else:
                head_args.E0s = ast.literal_eval(head_args.E0s)
        
        if 'atomic_numbers' in head_args:
            head_args.E0s = {k:v for k,v in head_args.E0s.items() if k in head_args.atomic_numbers}

        if "atomic_numbers" not in head_args:
            head_args.atomic_numbers = list(head_args.E0s.keys())

        # z_table
        head_args.z_table = tools.get_atomic_number_table_from_zs(head_args.atomic_numbers)
        logging.info(f"num of spicies: {len(head_args.z_table.zs)}")

        # atomic_energies
        head_args.atomic_energies = np.array(
            [head_args.E0s[z] for z in head_args.z_table.zs]
        )

        # overwright args.r_max with head specific r_max
        head_args.r_max = head_args.get('r_max', args.r_max)
        
        if isinstance(head_args.r_max, str):
            if head_args.r_max == "covalent_radii":
                assert args.r_max == head_args.r_max
                head_args.r_max = args.r_max_dict
                head_args.r_max_matrix = args.r_max_matrix
            else:
                raise NotImplementedError(f"r_max type {head_args.r_max} not supported")
                

    # Data preparation
    atomic_energies_dict = {k: v.E0s for k, v in args.heads.items()}
    
    # Atomic number table
    # yapf: disable
    # prioritize getting z_table from E0s
    logging.info("Prioritize getting z_table from heads")
    head_zs = [head_args.atomic_numbers for _, head_args in args.heads.items()]
    flattened_list = [num for sublist in head_zs for num in sublist]
    # union of zs among different datasets
    unique_elements = set(flattened_list)
    zs_list = list(unique_elements)
    assert isinstance(zs_list, list)
    z_table = tools.get_atomic_number_table_from_zs(zs_list)

    logging.info(f"total num of species {len(z_table.zs)}")
    # yapf: enable
    logging.info(f"merged z table: {z_table}")

    atomic_energies = dict_to_array(atomic_energies_dict, list(args.heads.keys()))
    logging.info(f"Atomic energies shape: {atomic_energies.shape}")
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    
    for head, head_args in args.heads.items():
        logging.info(f"=============    Reading dataset {head} and compute     ===========")
        
        if head_args.get("transform", None) and head_args.transform == "stress_kbar2evA":
            def stress_kbar2evA(atomic_data):
                atomic_data["stress"] = atomic_data["stress"] * -1e-1 * ase.units.GPa
                return atomic_data
            head_transform = stress_kbar2evA
        else: 
            head_transform = None

        if head_args.train_file.endswith(".xyz"):
            # TODO: test this branch
            if head_args.valid_file is not None:
                assert head_args.valid_file.endswith(
                    ".xyz"
                ), "valid_file if given must be same format as train_file"
            config_type_weights = get_config_type_weights(head_args.config_type_weights)
            collections, _ = get_dataset_from_xyz(
                train_path=head_args.train_file,
                valid_path=head_args.valid_file,
                valid_fraction=head_args.get('valid_fraction', None),
                config_type_weights=config_type_weights,
                test_path=head_args.get('test_file', None),
                seed=head_args.get('seed', 0),
                energy_key=head_args.get('energy_key', None),
                forces_key=head_args.get('forces_key', None),
                stress_key=head_args.get('stress_key', None),
                virials_key=head_args.get('virials_key', None),
                dipole_key=head_args.get('dipole_key', None),
                charges_key=head_args.get('charges_key', None),
                keep_isolated_atoms=head_args.get('keep_isolated_atoms', None),
            )

            logging.info(
                f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
                f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
            )

            head_args.train_set = [
                data.AtomicData.from_config(config, z_table=z_table, cutoff=head_args.r_max)
                for config in collections.train
            ]
            head_args.valid_set = [
                data.AtomicData.from_config(config, z_table=z_table, cutoff=head_args.r_max)
                for config in collections.valid
            ]
        elif head_args.train_file.endswith(".h5"):
            head_args.train_set = data.HDF5Dataset(head_args.train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
            head_args.valid_set = data.HDF5Dataset(head_args.valid_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
        else:  # This case would be for when the file path is to a directory of multiple .h5 files
            if head_args.get("dataset_type", None) is None or head_args.get("dataset_type", None).lower() == "hdf5":
                # default shared h5
                head_args.train_set = data.dataset_from_sharded_hdf5(
                    head_args.train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()), rank=rank, transform=head_transform
                )
                head_args.valid_set = data.dataset_from_sharded_hdf5(
                    head_args.valid_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()), rank=rank, transform=head_transform
                )
            elif head_args.get("dataset_type", None).lower() == "lmdb":
                head_args.train_set = data.LMDBDataset(head_args.train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
                head_args.valid_set = data.LMDBDataset(head_args.valid_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
            else:
                raise NotImplementedError("dataset_type {head_args.dataset_type} not supported")


        if head == "alex_pbe" and head_args.get("only_stable", False):
            train_set = head_args.train_set
            valid_set = head_args.valid_set

            
            # Paths to the mask files
            train_mask_path = "lmdb_train_stable_mask.pt"
            val_mask_path = "lmdb_val_stable_mask.pt"
            # Check if both mask files exist
            if os.path.exists(train_mask_path) and os.path.exists(val_mask_path):
                # Load the stable masks if they already exist
                train_mask = torch.load(train_mask_path)
                val_mask = torch.load(val_mask_path)
                if rank == 0:
                    print("Loaded existing stable masks.")
            else:
                # Generate stable masks using `only_stable` if they don’t exist
                train_mask = only_stable(train_set, world_size, rank)
                val_mask = only_stable(valid_set, world_size, rank)

                if rank == 0:
                    # Save the masks to ensure stability for future runs
                    torch.save(train_mask, train_mask_path)
                    torch.save(val_mask, val_mask_path)
                    print("Generated and saved new stable masks.")


            head_args.train_set = torch.utils.data.Subset(train_set, train_mask)
            head_args.valid_set = torch.utils.data.Subset(valid_set, val_mask)
        elif head == "alex_pbe" and head_args.get("stable_first", False):
            train_set = head_args.train_set
            valid_set = head_args.valid_set

            
            # Paths to the mask files
            train_mask_path = "lmdb_train_stable_mask.pt"
            val_mask_path = "lmdb_val_stable_mask.pt"
            # Check if both mask files exist
            if os.path.exists(train_mask_path) and os.path.exists(val_mask_path):
                # Load the stable masks if they already exist
                train_mask = torch.load(train_mask_path)
                val_mask = torch.load(val_mask_path)
                if rank == 0:
                    print("Loaded existing stable masks.")
            else:
                # Generate stable masks using `only_stable` if they don’t exist
                train_mask = only_stable(train_set, world_size, rank)
                val_mask = only_stable(valid_set, world_size, rank)

                if rank == 0:
                    # Save the masks to ensure stability for future runs
                    torch.save(train_mask, train_mask_path)
                    torch.save(val_mask, val_mask_path)
                    print("Generated and saved new stable masks.")

            # make the train_mask include the last last
            train_mask = train_mask + [i-1 for i in train_mask[1:]]

            head_args.train_set = torch.utils.data.Subset(train_set, train_mask)
            head_args.valid_set = torch.utils.data.Subset(valid_set, val_mask)
        
        if head_args.get("additional_set", None):
            additional_set = head_args.additional_set
            assert "train_file" in additional_set and "valid_file" in additional_set
            assert "additional_type" in additional_set
            train_file = head_args.additional_set.train_file
            valid_file = head_args.additional_set.valid_file
            additional_type = head_args.additional_set["additional_type"]
            if additional_type.lower() == "hdf5":
                additional_train_set = data.dataset_from_sharded_hdf5(
                    train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()), rank=rank, transform=head_transform
                )
                additional_valid_set = data.dataset_from_sharded_hdf5(
                    valid_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()), rank=rank, transform=head_transform
                )
            elif additional_type.lower() == "lmdb":
                additional_train_set = data.LMDBDataset(train_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
                additional_valid_set = data.LMDBDataset(valid_file, r_max=head_args.r_max, z_table=z_table, head=head, heads=list(args.heads.keys()))
            else:
                raise NotImplementedError("additiional_type not supported")
            head_args.train_set = ConcatDataset([head_args.train_set, additional_train_set]) 
            head_args.valid_set = ConcatDataset([head_args.valid_set, additional_valid_set])
            
        
        logging.info(f"Dataset {head} size --> {format_number(len(head_args.train_set))}")

        # subset train ratio
        if "train_ratio" in head_args.keys():
            ratio = head_args.train_ratio
            # Calculate the size for the 10% subset
            subset_size = int(ratio * len(head_args.train_set))
            remaining_size = len(head_args.train_set) - subset_size

            val_subset_size = int(ratio * len(head_args.valid_set))
            val_remaining_size = len(head_args.valid_set) - val_subset_size
            # Split the dataset
            RANDOM_SPLIT = False

            if RANDOM_SPLIT:
                head_args.train_set, _ = random_split(head_args.train_set, [subset_size, remaining_size])
                head_args.valid_set, _ = random_split(head_args.valid_set, [val_subset_size, val_remaining_size])

            if not RANDOM_SPLIT:
                train_indices = list(range(0, subset_size))
                val_indices = list(range(0, val_subset_size))
                head_args.train_set = torch.utils.data.Subset(head_args.train_set, train_indices)
                head_args.valid_set = torch.utils.data.Subset(head_args.valid_set, val_indices)

            logging.info(f"Dataset {head} subsampled size --> {format_number(len(head_args.train_set))}")
            logging.info(f"Dataset {head} subsampled valid size --> {format_number(len(head_args.valid_set))}")


        # head specific train_sampler
        head_args.train_sampler = None
        if args.distributed:
            head_args.train_sampler = torch.utils.data.distributed.DistributedSampler(
                head_args.train_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )

        # head specific train_loader
        head_args.train_loader = torch_geometric.dataloader.DataLoader(
            dataset=head_args.train_set,
            batch_size=args.batch_size,
            sampler=head_args.train_sampler,
            shuffle=(head_args.train_sampler is None),
            drop_last=(head_args.train_sampler is None),
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )

        if 'avg_num_neighbors' in head_args and head_args.avg_num_neighbors > 0:
            head_args.compute_avg_num_neighbors = False

        if head_args.get("plot_neighbor_distribution", False):
            avg_num_neighbors_per_elem, num_neighbor_per_atom = modules.compute_avg_num_neighbors_per_elem(head_args.train_loader, rank=rank)
            if args.distributed:
                num_graphs = torch.tensor(len(head_args.train_loader.dataset)).to(device)
                num_neighbors_per_elem = num_graphs * torch.tensor(avg_num_neighbors_per_elem).to(device)
                torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(
                    num_neighbors_per_elem, op=torch.distributed.ReduceOp.SUM
                )
                avg_num_neighbors_per_elem = num_neighbors_per_elem.cpu().numpy() / num_graphs.cpu().numpy()
            else:
                pass

            import matplotlib.pyplot as plt
            from ase.data import chemical_symbols
            
            
            # Convert atomic numbers to symbols
            element_symbols = [chemical_symbols[z] for z in head_args.z_table.zs]
            
            # Create the bar plot
            plt.figure(figsize=(32, 18))
            plt.bar(element_symbols, avg_num_neighbors_per_elem, color='b')
            
            # Add labels and title
            plt.xlabel('Element Symbol')
            plt.ylabel('Average Number of Neighbors')
            plt.title('Average Number of Neighbors per Element')
            
            # Display the plot
            plt.show()
            plt.savefig(f"avg_neighbor_per_elem.png")


        # TODO: mean std avg_num_neighbor if given
        #  avg number of neighbors
        if head_args.get('compute_avg_num_neighbors', True): # Default True
            logging.info("Computing avg_num_neighbors...")
            avg_num_neighbors = modules.compute_avg_num_neighbors(head_args.train_loader, rank=rank)
            if args.distributed:
                num_graphs = torch.tensor(len(head_args.train_loader.dataset)).to(device)
                num_neighbors = num_graphs * torch.tensor(avg_num_neighbors).to(device)
                torch.distributed.all_reduce(num_graphs, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(
                    num_neighbors, op=torch.distributed.ReduceOp.SUM
                )
                head_args.avg_num_neighbors = (num_neighbors / num_graphs).item()
            else:
                head_args.avg_num_neighbors = avg_num_neighbors
            logging.info("Complete")
        logging.info(f"Average number of neighbors: {head_args.avg_num_neighbors}")

        # scaling
        if args.scaling == "no_scaling":
            head_args.std = 1.0
            head_args.mean = 0.0
            logging.info("No scaling selected")
        elif ('mean' not in head_args or 'std' not in head_args) and args.model != "AtomicDipolesMACE":
            # NOTE: there is only one scaling used.
            logging.info("Computing scaling mean and std...")
            head_args.mean, head_args.std = modules.scaling_classes[args.scaling](
                head_args.train_loader, atomic_energies, rank=rank
            )
            head_args.mean = head_args.mean[-1]
            head_args.std = head_args.std[-1]
            logging.info("Complete")
        logging.info(f"mean {head_args.mean}, std {head_args.std}")

    # mask dataset
    if args.clean_alex:
        #import ipdb; ipdb.set_trace()
        label_l2 = torch.load("cache_files/label_l2.pt")
        pred_l2 = torch.load("cache_files/pred_l2.pt")
        delta_l2 = torch.load("cache_files/delta_l2.pt")

        masked_indices = torch.where(delta_l2 < 10)[0].cpu().numpy()

        # training set cleaning
        logging.info(f"[Alexandria] Removed number of datas --> {(delta_l2 >= 10).sum().item()}")
        train_set = args.heads['alex_pbe'].train_set
        args.heads['alex_pbe'].train_set = torch.utils.data.Subset(train_set, masked_indices)


        

    train_sets = {k:v.train_set for k,v in args.heads.items()}
    valid_sets = {k:v.valid_set for k,v in args.heads.items()}
    
    train_set = ConcatDataset(train_sets.values())


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

        valid_samplers = {}
        for head, valid_set in valid_sets.items():
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=args.seed,
            )
            valid_samplers[head] = valid_sampler
    
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
    

    valid_loaders = {}
    for head, valid_set in valid_sets.items():
        valid_loaders[head] = torch_geometric.dataloader.DataLoader(
            dataset=valid_set,
            batch_size=args.valid_batch_size,
            sampler=valid_samplers[head] if args.distributed else None,
            shuffle=False,
            drop_last=False,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
    
    # LOSS module
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
        head_stress_mask = torch.Tensor([float('mp' in k) for k in args.heads.keys()]).to(device=device) # TODO: make it general
        loss_fn = modules.UniversalLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
            head_stress_mask=head_stress_mask
        )
    elif args.loss == "omat24":
        head_stress_mask = torch.Tensor([float('mp' in k) for k in args.heads.keys()]).to(device=device)
        loss_fn = modules.OMat24Loss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            head_stress_mask=head_stress_mask
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
    logging.info(loss_fn)

    # Selecting outputs
    compute_virials = False
    if args.loss in ("stress", "virials", "huber", "universal", "omat24"):
        compute_virials = True
        args.compute_stress = True
        # args.error_table = "PerAtomRMSEstressvirials"
        logging.info(f"Over-wrighting the error table due to the loss setting -> {args.loss} loss")
        args.error_table = "PerAtomRMSE+EMAEstressvirials"

    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    logging.info(f"Selected the following outputs: {output_args}")

    heads = list(args.heads.keys())


    # Build model
    if args.foundation_model is not None:
        logging.info("Building model")
        model_config = extract_config_mace_model(model_foundation)
        model_config["atomic_energies"] = atomic_energies
        model_config["atomic_numbers"] = z_table.zs
        model_config["num_elements"] = len(z_table)
        args.max_L = model_config["hidden_irreps"].lmax
        if args.model == "MACE" and model_foundation.__class__.__name__ == "MACE":
            model_config["atomic_inter_shift"] = [0.0] * len(heads)
        else:
            model_config["atomic_inter_shift"] = [v.mean for v in args.heads.values()] #[args.mean] * len(heads)
        model_config["atomic_inter_scale"] = [v.std for v in args.heads.values()]  #[1.0] * len(heads)

        args.model = "FoundationMACE"
        model_config["heads"] = heads
        logging.info("Model configuration extracted from foundation model")
        logging.info("Using universal loss function for fine-tuning")
    else:
        logging.info("Building model")
        if args.num_channels is not None and args.max_L is not None:
            assert args.num_channels > 0, "num_channels must be positive integer"
            assert args.max_L >= 0, "max_L must be non-negative integer"
            args.hidden_irreps = o3.Irreps(
                (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
                .sort()
                .irreps.simplify()
            )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        model_config = dict(
            r_max=args.r_max if isinstance(args.r_max, float) else args.r_max_matrix, # TODO: different r_max for heads
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table), # check
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies, # check
            avg_num_neighbors=args.heads[args.avg_num_neighbor_head].avg_num_neighbors,   # Use MP avg_num_neighbors
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
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=[v.std for v in args.heads.values()],
            atomic_inter_shift=[0.0 for v in args.heads.values()],
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            agnostic_int=args.agnostic_int,
            agnostic_con=args.agnostic_con,
        )
    elif args.model == "ScaleShiftMACE": # Contains more parameters than MACE
        model = modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=[v.std for v in args.heads.values()],
            atomic_inter_shift=[0.0 for v in args.heads.values()],
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
            agnostic_int=args.agnostic_int,
            agnostic_con=args.agnostic_con,
        )
    elif args.model == "FoundationMACE":
        model = modules.ScaleShiftMACE(**model_config)
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
        if args.foundation_filter_elements:
            model = load_foundations_elements(
                model,
                model_foundation,
                z_table,
                load_readout=True,
                max_L=args.max_L,
            )
        else:
            model = load_foundations_elements(
                model,
                model_foundation,
                z_table,
                load_readout=False,
                max_L=args.max_L,
            )
    model.to(device)

    # Optimizer
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.interactions.named_parameters():
        if "linear.weight" in name or "skip_tp_full.weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    #import ipdb; ipdb.set_trace()

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
        eps=args.adam_eps,
    )

    optimizer: torch.optim.Optimizer
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    elif args.optimizer.lower() == "rmsprop":
        param_options.pop("amsgrad")
        optimizer = torch.optim.RMSprop(**param_options)
    elif args.optimizer.lower() == "adam":
        if args.adam_betas is not None:
            param_options['betas'] = eval(args.adam_betas)
        optimizer = torch.optim.Adam(**param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)
        logging.info(f"Optimizer {args.optimizer} not supported, using adam as default")
    if args.device == "xpu":
        logging.info("Optimzing model and optimzier for XPU")
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    logger = tools.MetricsLogger(directory=args.results_dir, tag=tag + "_train")

    lr_scheduler = LRScheduler(optimizer, args)

    swa: Optional[tools.SWAContainer] = None
    swas = [False]
    if args.swa:
        assert dipole_only is False, "swa for dipole fitting not implemented"
        swas.append(True)
        if args.start_swa is None:
            args.start_swa = (
                args.max_num_epochs // 4 * 3
            )  # if not set start swa at 75% of training
        else:
            if args.start_swa > args.max_num_epochs:
                logging.info(
                    f"Start swa must be less than max_num_epochs, got {args.start_swa} > {args.max_num_epochs}"
                )
                args.start_swa = args.max_num_epochs // 4 * 3
                logging.info(f"Setting start swa to {args.start_swa}")
        if args.loss == "forces_only":
            logging.info("Can not select swa with forces only loss.")
        elif args.loss == "virials":
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
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
            )
        else:
            loss_fn_energy = modules.WeightedEnergyForcesLoss(
                energy_weight=args.swa_energy_weight,
                forces_weight=args.swa_forces_weight,
            )
            logging.info(
                f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
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

    checkpoint_handler = tools.CheckpointHandler(
        directory=args.checkpoints_dir,
        tag=tag,
        keep=args.keep_checkpoints,
        swa_start=args.start_swa,
    )

    start_epoch = 0
    
    logging.info(model)
    logging.info(f"Number of parameters: {tools.count_parameters(model)}")
    logging.info(f"Optimizer: {optimizer}")
    
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
        valid_loaders=valid_loaders,
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
        restart=args.restart,
        log_opt=args.log_opt,
        async_update=args.async_update,
    )

    logging.info("Computing metrics for training, validation, and test sets")

    all_data_loaders = {
        "train": train_loader,
    }
    for head, valid_loader in valid_loaders.items():
        all_data_loaders[head] = valid_loader

    test_sets = {}
    if (args.train_file is not None) and args.train_file.endswith(".xyz"): # TODO: train_file is now in config.yaml
        for name, subset in collections.tests:
            test_sets[name] = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=args.r_max, heads=heads
                )
                for config in subset
            ]
    elif not args.multi_processed_test:
        assert False, "should not run this [temp]"
        test_files = get_files_with_suffix(args.test_dir, "_test.h5")
        for test_file in test_files:
            name = os.path.splitext(os.path.basename(test_file))[0]
            test_sets[name] = data.HDF5Dataset(
                test_file, r_max=args.r_max, z_table=z_table, heads=heads
            )
    else:
        for head, head_args in args.heads.items():
            if 'test_file' in head_args:
                assert check_folder_subfolder(head_args.test_file), f"test_file of Head {head} is not a directory or does not contains subfolders: {head_args.test_file}"
                test_folders = glob(os.path.join(head_args.test_file) + "/*")
                for folder in test_folders:
                    name = os.path.splitext(os.path.basename(folder))[0]
                    test_sets[head + name] = data.dataset_from_sharded_hdf5(
                        folder, r_max=args.r_max, z_table=z_table, heads=heads, head=head
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
        except AttributeError as e:
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
        logging.info(f"Loaded model from epoch {epoch}")

        for param in model.parameters():
            param.requires_grad = False
        table = create_error_table(
            table_type=args.error_table,
            all_data_loaders=all_data_loaders,
            model=model_to_evaluate,
            loss_fn=loss_fn,
            output_args=output_args,
            log_wandb=args.wandb,
            device=device,
            distributed=args.distributed,
        )
        logging.info("\n" + str(table))

        if rank == 0:
            # Save entire model
            if swa_eval:
                model_path = Path(args.checkpoints_dir) / (tag + "_swa.model")
            else:
                model_path = Path(args.checkpoints_dir) / (tag + ".model")
            logging.info(f"Saving model to {model_path}")
            if args.save_cpu:
                model = model.to("cpu")
            torch.save(model, model_path)

            if swa_eval:
                torch.save(model, Path(args.model_dir) / (args.name + "_swa.model"))
            else:
                torch.save(model, Path(args.model_dir) / (args.name + ".model"))

        if args.distributed:
            torch.distributed.barrier()

    logging.info("Done")
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
