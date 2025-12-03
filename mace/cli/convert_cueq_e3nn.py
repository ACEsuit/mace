import argparse
import logging
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from e3nn import o3

from mace.tools.cg import O3_e3nn
from mace.tools.cg_cueq_tools import symmetric_contraction_proj
from mace.tools.scripts_utils import extract_config_mace_model

try:
    import cuequivariance as cue

    CUEQQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQQ_AVAILABLE = False
    cue = None

SizeLike = Union[torch.Size, List[int]]


def shapes_match_up_to_unsqueeze(a: SizeLike, b: SizeLike) -> bool:
    if isinstance(a, torch.Tensor):
        a = a.shape
    if isinstance(b, torch.Tensor):
        b = b.shape

    def drop(s):
        return tuple(d for d in s if d != 1)

    return drop(a) == drop(b)


def reshape_like(src: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
    try:
        return src.reshape(ref_shape)
    except RuntimeError:
        return src.clone().reshape(ref_shape)


def get_kmax_pairs(
    num_product_irreps: int, correlation: int, num_layers: int
) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on num_product_irreps and correlation"""
    if correlation == 2:
        kmax_pairs = [[i, num_product_irreps] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, 0]]
        return kmax_pairs
    if correlation == 3:
        kmax_pairs = [[i, num_product_irreps] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, 0]]
        return kmax_pairs
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    num_product_irreps: int,
    products: torch.nn.Module,
    correlation: int,
    num_layers: int,
    use_reduced_cg: bool,
):
    """Transfer symmetric contraction weights from CuEq to E3nn format"""
    kmax_pairs = get_kmax_pairs(num_product_irreps, correlation, num_layers)
    suffixes = ["_max"] + [f".{i}" for i in range(correlation - 1)]
    for i, kmax in kmax_pairs:
        # Get the combined weight tensor from source
        irreps_in = o3.Irreps(
            irrep.ir for irrep in products[i].symmetric_contractions.irreps_in
        )
        irreps_out = o3.Irreps(
            irrep.ir for irrep in products[i].symmetric_contractions.irreps_out
        )
        wm = source_dict[f"products.{i}.symmetric_contractions.weight"]
        if use_reduced_cg:
            _, proj = symmetric_contraction_proj(
                cue.Irreps(O3_e3nn, str(irreps_in)),
                cue.Irreps(O3_e3nn, str(irreps_out)),
                list(range(1, correlation + 1)),
            )
            proj = np.linalg.pinv(proj)
            proj = torch.tensor(proj, dtype=wm.dtype, device=wm.device)
            wm = torch.einsum("zau,ab->zbu", wm, proj)
        # Get split sizes based on target dimensions
        splits = []
        for k in range(kmax + 1):
            for suffix in suffixes:
                key = f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}"
                target_shape = target_dict[key].shape
                splits.append(target_shape[1])
                if (
                    target_dict.get(
                        f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix.replace('.', '_')}"
                        + "_zeroed",
                        False,
                    )
                    and not use_reduced_cg
                ):
                    splits[-1] = 0

        # Split the weights using the calculated sizes
        weights_split = torch.split(wm, splits, dim=1)

        # Assign back to target dictionary
        idx = 0
        for k in range(kmax + 1):
            for suffix in suffixes:
                key = f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix}"
                if (
                    target_dict.get(
                        f"products.{i}.symmetric_contractions.contractions.{k}.weights{suffix.replace('.', '_')}_zeroed",
                        False,
                    )
                    and not use_reduced_cg
                ):
                    continue
                target_dict[key] = (
                    weights_split[idx] if splits[idx] > 0 else target_dict[key]
                )
                idx += 1


def transfer_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    num_product_irreps: int,
    correlation: int,
    num_layers: int,
    use_reduced_cg: bool,
):
    """Transfer weights from CuEq to E3nn format"""
    # Get state dicts
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # Transfer symmetric contractions
    products = target_model.products
    transfer_symmetric_contractions(
        source_dict,
        target_dict,
        num_product_irreps,
        products,
        correlation,
        num_layers,
        use_reduced_cg,
    )

    # Transfer remaining matching keys
    transferred_keys = set()
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}

    if remaining_keys:
        for key in remaining_keys:
            src = source_dict[key]
            tgt = target_dict[key]
            if source_dict[key].shape == target_dict[key].shape:
                logging.debug(f"Transferring additional key: {key}")
                target_dict[key] = source_dict[key]
            elif shapes_match_up_to_unsqueeze(src.shape, tgt.shape):
                logging.debug(
                    f"Transferring key {key} after adapting shape "
                    f"{tuple(src.shape)} → {tuple(tgt.shape)}"
                )
                target_dict[key] = reshape_like(src, tgt.shape)
            else:
                logging.warning(
                    f"Shape mismatch for key {key}: "
                    f"source {source_dict[key].shape} vs target {target_dict[key].shape}"
                )

    # Transfer avg_num_neighbors
    for i in range(num_layers):
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
    num_product_irreps = len(config["hidden_irreps"].slices()) - 1
    correlation = config["correlation"]
    use_reduced_cg = config.get("use_reduced_cg", True)

    # Remove CuEq config
    config.pop("cueq_config", None)

    # Create new model without CuEq config
    logging.info("Creating new model without CuEq settings")
    target_model = source_model.__class__(**config)

    # Transfer weights with proper remapping
    num_layers = config["num_interactions"]
    transfer_weights(
        source_model,
        target_model,
        num_product_irreps,
        correlation,
        num_layers,
        use_reduced_cg,
    )

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
