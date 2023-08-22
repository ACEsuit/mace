import sys

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE

model_path = sys.argv[1]  # takes model name as command-line input
model = torch.load(model_path)
model = model.double().to("cpu")
lammps_model = LAMMPS_MACE(model)
lammps_model_compiled = jit.compile(lammps_model)
lammps_model_compiled.save(model_path + "-lammps.pt")
