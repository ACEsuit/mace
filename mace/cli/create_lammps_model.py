import sys

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE


def main():
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} model_path"

    model_path = sys.argv[1]  # takes model name as command-line input
    model = torch.load(model_path)
    model = model.double().to("cpu")
    lammps_model = LAMMPS_MACE(model)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
