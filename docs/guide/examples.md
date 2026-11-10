# Executable examples

[examples/README.md](../../examples/README.md) describes three small training
workflows (MD22-style molecules, ANI1x-style molecular data and periodic water)
and a short ASE NVT run. Their per-PR inputs are deterministic synthetic smoke
fixtures, explicitly labelled as such. They demonstrate file/configuration and
CLI flow; they do not ship or impersonate the scientific datasets.

Supply a local dataset to run the same workflow on real data. The scripts do
not download datasets implicitly. Native public foundation publication is
pending; the NVT example accepts an explicit local native model artifact.

The [standalone JAX tutorial](../../tutorials/README.md) explains spherical
harmonics, CG coupling, neighbour-density correlations and energy assembly.
Its two marimo notebooks execute headlessly in CI. The tutorial is independent
of the production stack; its separate oracle test checks fp64 energies.
