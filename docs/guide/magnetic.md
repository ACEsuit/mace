# Magnetic models

Magnetic MACE adds spin information and magnetic observables to the spatial
message-passing problem. Supply magnetic moment inputs in the configuration's
specified units and coordinate convention. Time reversal and spatial rotation
are distinct symmetries; do not infer one from a test of the other.

The model configuration declares magnetic basis size, angular order, saturation
and optional one-body or SCF terms. Magnetic forces are derivatives with
respect to magnetic moments. Loss masks must distinguish unavailable labels
from a measured zero. Native tests include magnetic derivatives and frozen
model outputs; the historical `magnetic` marker probes the optional sphericart
oracle. See [configuration](../reforge/configuration.md) for the accepted fields
and [output contracts](../reforge/output_surface.md) for dimensions and units.
