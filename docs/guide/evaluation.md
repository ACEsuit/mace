# Evaluation, descriptors and response derivatives

```python
from ase import Atoms
from mace_torch.calculators import MACECalculator
atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
atoms.calc = MACECalculator(model_paths="model.json", device="cpu")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The calculator loads the model's metadata and precision. Request only outputs
supported by its observable catalogue. Forces are negative energy gradients;
stress uses the strain derivative divided by physical periodic-cell volume.
A fully aperiodic graph reports zero stress. A mixed batch retains the periodic
members' stresses. Descriptor, Hessian, dipole, polarizability and charge
response requests depend on the model's declared outputs; they are not implied
by an energy-only checkpoint. See [output contracts](../reforge/output_surface.md)
and [command help](../reforge/cli.md) for exact selectors and array layouts.

For ASE molecular dynamics, choose a step appropriate to the potential and
system. Check conservation in NVE before adding a thermostat. The executable
[NVT example](examples.md) is a short integration check, not a stability study
or a validated physical potential.
