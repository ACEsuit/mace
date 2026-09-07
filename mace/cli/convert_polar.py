"""Convert a trusted legacy PolarMACE whole model for graph_longrange 0.4.3."""

import argparse
from pathlib import Path

import torch

from mace.tools.polar_conversion import PBC_HANDLING_MODES, convert_polar_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pbc-handling", choices=PBC_HANDLING_MODES, default="auto")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be a new path; the input model is preserved")
    model = torch.load(args.input, map_location="cpu", weights_only=False)
    convert_polar_model(model, args.pbc_handling)
    with args.output.open("xb") as stream:
        torch.save(model, stream)
    restored = torch.load(args.output, map_location="cpu", weights_only=False)
    convert_polar_model(restored, args.pbc_handling)
    print(f"Saved converted PolarMACE to {args.output} (mode={args.pbc_handling})")


if __name__ == "__main__":
    main()
