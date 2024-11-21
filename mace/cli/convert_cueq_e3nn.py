import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch

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
    """Transfer symmetric contraction weights from CuEq to E3nn format"""
    kmax_pairs = get_kmax_pairs(max_L, correlation)

    for i, kmax in kmax_pairs:
        # Get the combined weight tensor from source
        wm = source_dict[f"products.{i}.symmetric_contractions.weight"]

        # Get split sizes based on target dimensions
        splits = []
        for k in range(kmax + 1):
            for suffix in ["_max", ".0", ".1"]:
                key = f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}"
                target_shape = target_dict[key].shape
                splits.append(target_shape[1])

        # Split the weights using the calculated sizes
        weights_split = torch.split(wm, splits, dim=1)

        # Assign back to target dictionary
        idx = 0
        for k in range(kmax + 1):
            target_dict[
                f"products.{i}.symmetric_contractions.contractions.{k}.weights_max"
            ] = weights_split[idx]
            target_dict[
                f"products.{i}.symmetric_contractions.contractions.{k}.weights.0"
            ] = weights_split[idx + 1]
            target_dict[
                f"products.{i}.symmetric_contractions.contractions.{k}.weights.1"
            ] = weights_split[idx + 2]
            idx += 3


def transfer_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_L: int,
    correlation: int,
):
    """Transfer weights from CuEq to E3nn format"""
    # Get state dicts
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

    # Transfer remaining matching keys
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


def run(input_model, output_model="_e3nn.model", device="cpu", return_model=True):

    # Load CuEq model
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

    # Remove CuEq config
    config.pop("cueq_config", None)

    # Create new model without CuEq config
    logging.info("Creating new model without CuEq settings")
    target_model = source_model.__class__(**config)

    # Transfer weights with proper remapping
    transfer_weights(source_model, target_model, max_L, correlation)

    if return_model:
        return target_model

    # Save model
    if isinstance(input_model, str):
        base = os.path.splitext(input_model)[0]
        output_model = f"{base}.{output_model}"
    logging.warning(f"Saving E3nn model to {output_model}")
    torch.save(target_model, output_model)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Path to input CuEq model")
    parser.add_argument(
        "--output_model", help="Path to output E3nn model", default="e3nn_model.pt"
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
