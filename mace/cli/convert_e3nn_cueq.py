import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import cuequivariance as cue

from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scripts_utils import extract_config_mace_model
from mace import modules


def get_transfer_keys() -> List[str]:
    """Get list of keys that need to be transferred"""
    return [
        'node_embedding.linear.weight',
        'radial_embedding.bessel_fn.bessel_weights',
        'atomic_energies_fn.atomic_energies',
        'readouts.0.linear.weight',
        'scale_shift.scale',
        'scale_shift.shift',
        *[f'readouts.1.linear_{i}.weight' for i in range(1,3)]
    ] + [
        s for j in range(2) for s in [
            f'interactions.{j}.linear_up.weight',
            *[f'interactions.{j}.conv_tp_weights.layer{i}.weight' for i in range(4)],
            f'interactions.{j}.linear.weight',
            f'interactions.{j}.skip_tp.weight',
            f'products.{j}.linear.weight'
        ]
    ]

def get_kmax_pairs(max_L: int, correlation: int) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on max_L and correlation"""
    if correlation == 2:
        raise NotImplementedError("Correlation 2 not supported yet")
    elif correlation == 3:
        return [[0, max_L], [1, 0]]
    else:
        raise NotImplementedError(f"Correlation {correlation} not supported")

def transfer_symmetric_contractions(source_dict: Dict[str, torch.Tensor], 
                                 target_dict: Dict[str, torch.Tensor],
                                 max_L: int,
                                 correlation: int):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(max_L, correlation)
    logging.info(f"Using kmax pairs {kmax_pairs} for max_L={max_L}, correlation={correlation}")
        
    for i, kmax in kmax_pairs:
        wm = torch.concatenate([
            source_dict[f'products.{i}.symmetric_contractions.contractions.{k}.weights{j}']
            for k in range(kmax+1) for j in ['_max','.0','.1']],dim=1) #.float()
        target_dict[f'products.{i}.symmetric_contractions.weight'] = wm

def transfer_weights(source_model: torch.nn.Module, target_model: torch.nn.Module, 
                    max_L: int, correlation: int):
    """Transfer weights with proper remapping"""
    # Get source state dict
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Transfer main weights
    transfer_keys = get_transfer_keys()
    logging.info("Transferring main weights...")
    for key in transfer_keys:
        if key in source_dict:  # Check if key exists
            target_dict[key] = source_dict[key]
        else:
            logging.warning(f"Key {key} not found in source model")
    
    # Transfer symmetric contractions
    logging.info("Transferring symmetric contractions...")
    transfer_symmetric_contractions(source_dict, target_dict, max_L, correlation)

    transferred_keys = set(transfer_keys)
    remaining_keys = set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    remaining_keys = {k for k in remaining_keys if 'symmetric_contraction' not in k}

    if remaining_keys:
        logging.info(f"Found {len(remaining_keys)} additional matching keys to transfer")
        for key in remaining_keys:
            if source_dict[key].shape == target_dict[key].shape:
                logging.debug(f"Transferring additional key: {key}")
                target_dict[key] = source_dict[key]
            else:
                logging.warning(
                    f"Shape mismatch for key {key}: "
                    f"source {source_dict[key].shape} vs target {target_dict[key].shape}"
                )
    # Transfer avg_num_neighbors
    for i in range(2):
        target_model.interactions[i].avg_num_neighbors = source_model.interactions[i].avg_num_neighbors
    
    # Load state dict into target model
    target_model.load_state_dict(target_dict)

def run(
    input_model,
    output_model,
    device='cuda',
    layout='mul_ir',
    group='O3',
    return_model=True
):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load original model
    logging.info(f"Loading model from {input_model}")
    # check if input_model is a path or a model
    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model
    
    # Extract configuration
    logging.info("Extracting model configuration")
    config = extract_config_mace_model(source_model)
    
    # Get max_L and correlation from config
    max_L = config["hidden_irreps"].lmax
    correlation = config["correlation"]
    logging.info(f"Extracted max_L={max_L}, correlation={correlation}")
    
    # Add cuequivariance config
    config["cueq_config"] = CuEquivarianceConfig(
        enabled=True,
        layout="mul_ir",
        group="O3_e3nn",
        optimize_all=True,
    )
    
    # Create new model with cuequivariance config
    logging.info("Creating new model with cuequivariance settings")
    target_model = source_model.__class__(**config)
    
    # Transfer weights with proper remapping
    logging.info("Transferring weights with remapping...")
    transfer_weights(source_model, target_model, max_L, correlation)
    
    if return_model:
        return target_model
    else:
        # Save model
        output_model = Path(input_model).parent / output_model
        logging.info(f"Saving cuequivariance model to {output_model}")
        torch.save(target_model, output_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Path to input MACE model')
    parser.add_argument('output_model', help='Path to output cuequivariance model', default='cuequivariance_model.pt')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--layout', default='mul_ir', choices=['mul_ir', 'ir_mul'], help='Memory layout for tensors')
    parser.add_argument('--group', default='O3_e3nn', choices=['O3', 'O3_e3nn'], help='Symmetry group')
    parser.add_argument('--return_model', action='store_false', help='Return model instead of saving to file')
    args = parser.parse_args()
    
    run(
        input_model=args.input_model,
        output_model=args.output_model,
        device=args.device,
        layout=args.layout,
        group=args.group,
        return_model=args.return_model
    )


if __name__ == '__main__':
    main()