import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch

from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scripts_utils import extract_config_mace_model


def get_transfer_keys() -> List[str]:
    """Get list of keys that need to be transferred"""
    return [
        "node_embedding.linear.weight",
        "radial_embedding.bessel_fn.bessel_weights",
        "atomic_energies_fn.atomic_energies",
        "readouts.0.linear.weight",
        "scale_shift.scale",
        "scale_shift.shift",
        *[f"readouts.1.linear_{i}.weight" for i in range(1, 3)],
    ] + [
        s
        for j in range(2)
        for s in [
            f"interactions.{j}.linear_up.weight",
            *[f"interactions.{j}.conv_tp_weights.layer{i}.weight" for i in range(4)],
            f"interactions.{j}.linear.weight",
            f"interactions.{j}.skip_tp.weight",
            f"products.{j}.linear.weight",
        ]
    ]


def get_kmax_pairs(max_L: int, correlation: int) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on max_L and correlation"""
    if correlation == 2:
        raise NotImplementedError("Correlation 2 not supported yet")
    if correlation == 3:
        return [[0, max_L], [1, 0]]
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    max_L: int,
    correlation: int,
):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(max_L, correlation)

    for i, kmax in kmax_pairs:
        wm = torch.concatenate(
            [
                source_dict[
                    f"products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                ]
                for k in range(kmax + 1)
                for j in ["_max", ".0", ".1"]
            ],
            dim=1,
        )
        target_dict[f"products.{i}.symmetric_contractions.weight"] = wm


def transfer_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_L: int,
    correlation: int,
):
    """Transfer weights with proper remapping"""
    # Get source state dict
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # Transfer main weights
    transfer_keys = get_transfer_keys()
    for key in transfer_keys:
        if key in source_dict:  # Check if key exists
            target_dict[key] = source_dict[key]
        else:
            logging.warning(f"Key {key} not found in source model")

    # Transfer symmetric contractions
    transfer_symmetric_contractions(source_dict, target_dict, max_L, correlation)

    transferred_keys = set(transfer_keys)
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    if remaining_keys:
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
        target_model.interactions[i].avg_num_neighbors = source_model.interactions[
            i
        ].avg_num_neighbors

    # Load state dict into target model
    target_model.load_state_dict(target_dict)


def run(
    input_model,
    output_model="_cueq.model",
    device="cpu",
    return_model=True,
):
    # Setup logging

    # Load original model
    # logging.warning(f"Loading model")
    # check if input_model is a path or a model
    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model
    default_dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(default_dtype)
    # Extract configuration
    config = extract_config_mace_model(source_model)

    # Get max_L and correlation from config
    max_L = config["hidden_irreps"].lmax
    correlation = config["correlation"]

    # Add cuequivariance config
    config["cueq_config"] = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
    )

    # Create new model with cuequivariance config
    logging.info("Creating new model with cuequivariance settings")
    target_model = source_model.__class__(**config).to(device)

    # Transfer weights with proper remapping
    transfer_weights(source_model, target_model, max_L, correlation)

    if return_model:
        return target_model

    if isinstance(input_model, str):
        base = os.path.splitext(input_model)[0]
        output_model = f"{base}.{output_model}"
    logging.warning(f"Saving CuEq model to {output_model}")
    torch.save(target_model, output_model)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Path to input MACE model")
    parser.add_argument(
        "--output_model",
        help="Path to output cuequivariance model",
        default="cueq_model.pt",
    )
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument(
        "--return_model",
        action="store_false",
        help="Return model instead of saving to file",
    )
    args = parser.parse_args()

    run(
        input_model=args.input_model,
        output_model=args.output_model,
        device=args.device,
        return_model=args.return_model,
    )


if __name__ == "__main__":
    main()
