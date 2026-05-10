"""Hybrid cueq + openequivariance model converter."""

import argparse
import logging
import os

import torch

from mace.modules.wrapper_ops import CuEquivarianceConfig, OEQConfig
from mace.tools.scripts_utils import extract_config_mace_model

try:
    from mace.cli.convert_e3nn_cueq import transfer_symmetric_contractions

    CUEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQ_AVAILABLE = False

try:
    import openequivariance as _oeq  # noqa: F401

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False


def run(
    input_model,
    output_model="_hybrid.model",
    device="cpu",
    return_model=True,
):
    if not CUEQ_AVAILABLE:
        raise ImportError("cuequivariance is required for hybrid conversion")
    if not OEQ_AVAILABLE:
        raise ImportError("openequivariance is required for hybrid conversion")

    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model

    default_dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(default_dtype)

    config = extract_config_mace_model(source_model)

    num_product_irreps = len(config["hidden_irreps"].slices()) - 1
    correlation = config["correlation"]
    num_layers = config["num_interactions"]
    use_reduced_cg = config.get("use_reduced_cg", True)

    config["cueq_config"] = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=False,
        optimize_linear=True,
        optimize_symmetric=True,
        optimize_fctp=True,
        optimize_channelwise=False,
        conv_fusion=False,
    )

    config["oeq_config"] = OEQConfig(
        enabled=True,
        optimize_all=False,
        optimize_channelwise=True,
        conv_fusion="atomic",
    )

    logging.info(
        "Creating hybrid model: cueq for symmetric/linear/fctp, "
        "oeq TensorProductConv for channelwise TP"
    )
    target_model = source_model.__class__(**config).to(device)

    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    transfer_symmetric_contractions(
        source_dict,
        target_dict,
        num_product_irreps,
        source_model.products,
        correlation,
        num_layers,
        use_reduced_cg,
        keep_last_layer_irreps=False,
    )

    transferred_keys = set()
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    remaining_keys = {k for k in remaining_keys if ".conv_tp." not in k}

    for key in remaining_keys:
        src = source_dict[key]
        tgt = target_dict[key]
        if src.shape == tgt.shape:
            target_dict[key] = src
        elif _shapes_match_up_to_unsqueeze(src.shape, tgt.shape):
            target_dict[key] = src.reshape(tgt.shape)
        else:
            logging.warning(
                "Shape mismatch for key %s: source %s vs target %s",
                key,
                src.shape,
                tgt.shape,
            )

    target_model.load_state_dict(target_dict)

    for i in range(num_layers):
        target_model.interactions[i].avg_num_neighbors = source_model.interactions[
            i
        ].avg_num_neighbors

    if return_model:
        return target_model

    if isinstance(input_model, str):
        base = os.path.splitext(input_model)[0]
        output_model = f"{base}.{output_model}"
    logging.warning("Saving hybrid model to %s", output_model)
    torch.save(target_model, output_model)
    return None


def _shapes_match_up_to_unsqueeze(a, b):
    def drop(s):
        return tuple(d for d in s if d != 1)

    return drop(a) == drop(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Path to input MACE model")
    parser.add_argument(
        "--output_model",
        default="hybrid_model.pt",
        help="Path to output hybrid model",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--return_model", action="store_false")
    args = parser.parse_args()

    run(
        input_model=args.input_model,
        output_model=args.output_model,
        device=args.device,
        return_model=args.return_model,
    )


if __name__ == "__main__":
    main()
