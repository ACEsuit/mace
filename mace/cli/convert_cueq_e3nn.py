import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    CUET_AVAILABLE = True
except ImportError:
    raise ImportError("cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled.")

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
        # For 3-body correlations
        return [[0,1], [1,0]]
    elif correlation == 3:
        # For 4-body correlations
        if max_L <= 2:
            return [[0,1], [1,0]]
        else:
            return [[0,2], [1,0]]
    else:
        logging.warning(f"Unexpected correlation {correlation}, defaulting to [[0,1], [1,0]]")
        return [[0,1], [1,0]]

def transfer_symmetric_contractions(source_dict: Dict[str, torch.Tensor], 
                                 target_dict: Dict[str, torch.Tensor],
                                 max_L: int,
                                 correlation: int):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(max_L, correlation)
    logging.info(f"Using kmax pairs {kmax_pairs} for max_L={max_L}, correlation={correlation}")
        
    for i, kmax in kmax_pairs:
        for k in range(kmax + 1):
            for suffix in ['.0', '.1', '_max']:
                key = f'products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}'
                if key in source_dict:  # Check if key exists to avoid errors
                    target_dict[key] = source_dict[key]
                else:
                    logging.warning(f"Key {key} not found in source model")

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
    
    # Transfer avg_num_neighbors
    for i in range(2):
        target_model.interactions[i].avg_num_neighbors = source_model.interactions[i].avg_num_neighbors
    
    # Load state dict into target model
    target_model.load_state_dict(target_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Path to input cuequivariance model')
    parser.add_argument('output_model', help='Path to output e3nn model')
    parser.add_argument('--device', default='cpu', help='Device to use')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load cuequivariance model
    logging.info(f"Loading model from {args.input_model}")
    source_model = torch.load(args.input_model, map_location=args.device)
    
    # Extract configuration
    logging.info("Extracting model configuration")
    config = extract_config_mace_model(source_model)
    
    # Get max_L and correlation from config
    max_L = config["max_ell"]
    correlation = config["correlation"]
    logging.info(f"Extracted max_L={max_L}, correlation={correlation}")
    
    # Replace cuequivariance config with disabled version
    config["cueq_config"] = CuEquivarianceConfig(
        layout_str="ir_mul",
        group="O3",
        max_L=max_L,
        correlation=correlation
    )
    
    # Create new model with e3nn config
    logging.info("Creating new model with e3nn settings")
    if isinstance(source_model, modules.MACE):
        target_model = modules.MACE(**config)
    else:
        target_model = modules.ScaleShiftMACE(**config)
    
    # Transfer weights with proper remapping
    logging.info("Transferring weights with remapping...")
    transfer_weights(source_model, target_model, max_L, correlation)
    
    # Save model
    logging.info(f"Saving e3nn model to {args.output_model}")
    torch.save(target_model, args.output_model)

if __name__ == '__main__':
    main()