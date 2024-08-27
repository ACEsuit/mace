import argparse

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default="default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path  # takes model name as command-line input
    head = args.head
    model = torch.load(model_path)
    model = model.double().to("cpu")
    lammps_model = LAMMPS_MACE(model, head=head)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
