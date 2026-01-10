# pylint: disable=wrong-import-position
import argparse
import copy
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type of the model to be converted to LAMMPS",
        default="float64",
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Old libtorch format, or new mliap format",
        default="libtorch",
    )
    return parser.parse_args()


def select_head(model):
    heads = model.heads if hasattr(model, "heads") else [None]

    if len(heads) == 1:
        return heads[0]

    for _i, _head in enumerate(heads):
        pass

    # Ask the user to select a head
    selected = input(
        f"Select a head by number (Defaulting to head: {len(heads)}, press Enter to accept): ",
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        return None
    return heads[-1]


def main() -> None:
    args = parse_args()
    model_path = args.model_path  # takes model name as command-line input
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.dtype == "float64":
        model = model.double().to("cpu")
    elif args.dtype == "float32":
        model = model.float().to("cpu")

    if args.format == "mliap":
        # Enabling cuequivariance by default. TODO: switch?
        model = run_e3nn_to_cueq(copy.deepcopy(model))
        model.lammps_mliap = True

    head = select_head(model) if args.head is None else args.head

    lammps_class = LAMMPS_MLIAP_MACE if args.format == "mliap" else LAMMPS_MACE
    lammps_model = (
        lammps_class(model, head=head) if head is not None else lammps_class(model)
    )
    if args.format == "mliap":
        torch.save(lammps_model, model_path + "-mliap_lammps.pt")
    else:
        lammps_model_compiled = jit.compile(lammps_model)
        lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
