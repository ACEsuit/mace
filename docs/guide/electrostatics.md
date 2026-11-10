# Polar, LES and electrostatic models

A local equivariant backbone can supply learned sources to a long-range solver.
The local source, electrostatic energy and differentiable response are separate
contracts. Choose the boundary conditions, cell, charge, external fields and
precision consistently with the model's training data.

Polar and LES models expose additional latent multipoles and response outputs.
LES requires its optional library (`les` test capability). The historical
`polar` capability probes the external graph_longrange oracle; native Polar
coverage also runs without that oracle. Born effective charges and dielectric
response are derivatives, so changing a solver must preserve differentiation,
not just total energy. Read [configuration](../reforge/configuration.md) and
[observable contracts](../reforge/output_surface.md) for supported outputs.

Frozen tests cover local sources, QEq and fixed-point SCF families. The
EnergyFunctionalSCF reference is unavailable and its validation remains pending.
Polar medium/large default-fp32 support has unresolved measured discrepancies;
float64 evidence does not promote those failed precision badges. Native model
URLs/DOIs remain unpublished. See the [model registry](../reforge/model_registry.md)
for loading local converted checkpoints and checking their support status.
