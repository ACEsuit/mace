# Tiny dielectric reference

Target `dielectric` builds a seeded, untrained fp64 `AtomicDielectricMACE`
with two interaction layers, correlation two, H/C/O species, and scalar,
vector and rank-two tensor channels in both its hidden features and final
readout. It uses frozen legacy only and needs no downloads or optional backend.

The committed checkpoint, build sidecar (including SHA-256), and reference
record charges, dipole, atomic dipoles, Cartesian and spherical polarizability,
and the complete position Jacobians `dmu_dr` and `dalpha_dr`. They use the
existing `dimer_short`, `isolated_atom` and `water_cluster` fixtures: short
distances, no edges, and a molecular cluster. Fixture selection is explicit
so later additions cannot change this reference's scope. Charged structures
are covered separately by live parity tests.

The numerical values and units are those of the legacy forward, including its
historical dielectric dipole convention; this capture changes no convention.
Comparison uses the existing `fp64_cpu_reference` row. The test also checks
that the seeded recipe reproduces the reference and that the water response,
anisotropy and derivatives are nonzero. Native conversion is checked under
`tests/parity`, independently of this legacy-only capture.

This is the first capture requested for FM-5/ARCH-3, committed separately from
conversion changes. No previous oracle is replaced. Published MACE-MDP remains
FM-2's scope.

```bash
python tests/golden/regenerate.py --target dielectric --i-know-what-i-am-doing
python -m pytest tests/golden/test_tiny_dielectric.py
```

The first command rewrites this family's artifacts and is reserved for an
explicitly authorized capture; normal verification only runs the second.
