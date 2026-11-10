# Working in MACE V1

Read [CONTRIBUTING.md](CONTRIBUTING.md) and the package entry document for the
code you change. V1 is an unpublished reference; final implementation PRs land
on `mace-reforge`. See the [README](README.md) for the reference branch status.

| Distribution | Import | Purpose |
|---|---|---|
| mace-core | mace_core | Types, configuration, data, graph and backend contracts; framework-free |
| mace-torch | mace_torch | Models, derivatives, training, calculator and deployment |
| mace-jax | mace_jax | Inference only |
| mace-backend-example | mace_backend_example | Independently registered backend example |

`mace` is the hierarchical CLI: `train`, `eval`, `data`, `model`, `export`, `plot`
and `md`. The PyTorch distribution owns it and the retained `mace_*` aliases.
There is no launcher, engine switch or in-tree legacy package. Model weights
are canonical; selecting a backend does not convert their layout.

`mace_core` imports neither framework nor a sibling. PyTorch and JAX do not
import each other. Model inference does not import training. Operators and
weights follow the contracts in the [backend skill](.claude/skills/mace-reforge-backend/SKILL.md).

## Tests and capabilities

Package unit tests live under each package. Shared frozen references live in
`tests/golden/`; native checkpoint and gradient gates are in `tests/retirement/`.
`tests/parity/` retains frozen external electrostatics and runtime comparisons.
`tests/architecture/` checks dependency and metadata contracts; `tests/workflows/`
executes commands and training; integrations have contract and external-binary
tiers. `tutorials/tests/` and documentation examples exercise the teaching surface.

`CAPABILITY_PROBES` in `tests/conftest.py` is authoritative. Current probes:
`gpu`, `cueq`, `oeq`, `polar`, `les`, `magnetic`, `torchsim`, `schedulefree`, `wandb`, `bin_lammps`, `bin_openmm`, `bin_openmm_native`, `bin_libmace`, `bin_gromacs`, `network`.
Missing capabilities skip locally. `MACE_REQUIRE_CAPS` makes missing promised
capabilities fail, including zero collection; a marker expression must actually
select each promised capability. Directory marks include workflows → slow,
foundations → network, benchmarks → benchmark + slow. Historical extension
markers remain registered for recorded test provenance.

A CI job's `.github/actions/run-tests/action.yml` inputs are its local recipe.
See CONTRIBUTING for installation and the ruff/ty/prek checks. Name new package
tests `test_mace_<package>_<topic>.py`, unique across all package trees.

## Frozen numerical evidence

Original reference JSON, models and fixtures are edit-locked. Retired source
and one-time differential checks are available at the source revision recorded
in `tests/retirement/artifacts/provenance.json`. Do not regenerate them to make
an implementation pass. Historical `mace/tools/torch_geometric/` and
`mace/tools/scatter.py` were vendored and lint-exempt; they are absent here.
An explicit converter can run in an external legacy interpreter; ordinary
inference, training and testing must not import that environment.

Skills are indexed in [skills.json](skills.json). `npx skills add
https://github.com/ACEsuit/mace` discovers `.claude/skills/` when this layout is
available on the selected ref. The manifest and links validate offline.
