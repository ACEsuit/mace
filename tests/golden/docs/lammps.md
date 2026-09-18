# The LAMMPS export goldens

Target: `lammps`, part of `all`.

Two references, pinning two different things about
`mace_create_lammps_model` on the `ScaleShiftMACE` anchor:

* `lammps_export_libtorch_fp64.json` — the numbers. The exported libtorch
  artefact is replayed on an open cluster of `N_REPEAT^3` replicas of one
  fixture, with the central replica as the LOCAL atoms, and its outputs are
  committed at the `fp64_cpu_reference` row.
* `lammps_export_mliap_interface.json` — what LAMMPS reads off the ML-IAP
  artefact *before* it calls the model. The numerics there are deliberately
  not pinned.

**The input is committed, not rebuilt at test time.** The replay needs a
neighbour list in LAMMPS's real-versus-ghost partitioning, and building one
at test time would make the golden a test of this repository's neighbour-list
code rather than of the export. So the recorded input is part of the
reference, and `tests/integrations/lammps/export_golden.py` explains the
partitioning it encodes. This target is the only place in the golden
machinery that reaches into the package's neighbour list, and it does so once,
under the regeneration lock.

Regenerating runs the export CLI in both formats in a temporary directory,
so it needs nothing installed beyond the package itself — no LAMMPS binary.
The reference is written compactly rather than indented: it is mostly a few
thousand edge indices, and one number per line would triple a file nobody
reads by eye.

## Running

The golden tests carry no marker and need no LAMMPS binary — they replay the
exported artefact directly. The `bin_lammps` tier that runs a real LAMMPS is
separate; see `tests/integrations/README.md`.
