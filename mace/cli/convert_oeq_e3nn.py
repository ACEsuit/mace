import argparse
import logging
import os

import torch

from mace.tools.scripts_utils import extract_config_mace_model


def run(input_model, output_model="_e3nn.model", device="cpu", return_model=True):
    # Load OEQ model
    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model
    default_dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(default_dtype)
    # Extract configuration
    config = extract_config_mace_model(source_model)

    # Remove OEQ config
    config.pop("oeq_config", None)

    # Create new model without CuEq config
    logging.info("Creating new model without OEQ settings")
    target_model = source_model.__class__(**config).to(device)

    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    for key in source_dict:
        if ".conv_tp." not in key:
            target_dict[key] = source_dict[key]

    for i in range(2):

        target_model.interactions[i].avg_num_neighbors = source_model.interactions[
            i
        ].avg_num_neighbors

    target_model.load_state_dict(target_dict)

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
    parser.add_argument("input_model", help="Path to input oeq model")
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
