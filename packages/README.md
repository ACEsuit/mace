# The MACE v1 packages

Three packages, one direction of dependency:

```
mace-core            contract + pure math; no torch, no jax, no e3nn
   ^        ^
   |        |
mace-torch  mace-jax
```

`mace_torch` and `mace_jax` are two implementations of one contract. Neither
imports the other, and neither imports the legacy `mace` package, which stays
in the tree as a frozen numerical oracle until it is retired.

These are scaffolds. Each package installs, imports and runs one test; the
library code arrives with the tickets that build it.

## Names

| Directory | Distribution | Import name | Tag prefix |
|---|---|---|---|
| `packages/mace-core` | `mace-core` | `mace_core` | `mace-core-v*` |
| `packages/mace-torch` | `mace-torch-v1` | `mace_torch` | `mace-torch-v*` |
| `packages/mace-jax` | `mace-jax` | `mace_jax` | `mace-jax-v*` |
| `packages/mace-launcher` | `mace-launcher` | `mace_launcher` | `mace-launcher-v*` |

The import names never collide with the legacy import name `mace`, so both
stacks live in one process.

The distribution names cannot all match their import names. The legacy package
holds the `mace-torch` distribution name, and two distributions cannot both
hold one name in a single environment, so the v1 PyTorch package ships as
**`mace-torch-v1`** while keeping the `mace_torch` import name.

**`mace-torch-v1` is never published to PyPI.** It exists so the two stacks can
be installed side by side while both are in the tree. `mace-torch` on PyPI is
frozen at its final v0.3.x release and stays dormant for the whole rewrite, so
the name is free the moment legacy leaves the tree: RET-6 renames the
distribution to `mace-torch`, and the v1.0.0 release publishes under that name.
PyPI has no rename operation, so publishing `mace-torch-v1` even once would
turn that switch from a no-op into a permanent user migration.

Tag prefixes are keyed on the directory rather than the distribution name, so
each package versions independently from git tags and the RET-6 rename touches
only the `name` field.

`mace-launcher` owns every `mace_*` console script and none of the other three
declares one, because two distributions declaring the same script name is
undefined behaviour in pip. That is not a theoretical hazard: uninstalling one
of two such distributions deletes the script files the other one had
overwritten, and the surviving distribution is left with none.

The launcher is also the only place the two stacks meet. It picks one with
`--engine {legacy,v1}` or `MACE_ENGINE`, defaulting to `legacy`, and installs a
runtime guard that fails if a v1 module imports the frozen legacy package.

## Installing alongside the legacy package

Both stacks in one environment, from a fresh venv at the repository root:

```bash
python -m venv .venv-v1 && source .venv-v1/bin/activate
pip install -e packages/mace-core -e packages/mace-torch \
            -e packages/mace-jax -e packages/mace-launcher
pip install -e .
```

One `pip install` for the packages, not one each: `mace-torch-v1` and
`mace-jax` require `mace-core`, which is not on PyPI, so pip has to see it as a
local requirement in the same resolution.

Then, in one process:

```bash
python -c "import mace, mace_core, mace_torch, mace_jax; print('coexist ok')"
```

The legacy install is the same `pip install -e .` it has always been. Nothing
about it changes, and the scaffold cannot enter the legacy wheel: its build
discovers packages with `setuptools.find_packages()`, which prunes any path
component that is not a Python identifier, and every directory here is
hyphenated. `tests/architecture/test_packaging_isolation.py` asserts that
rather than trusting it.

## Test file names

Test files are collected from several packages in one pytest run, under one
rootdir, and none of the `tests/` directories is an importable package. Two
files sharing a basename therefore collide on import. Prefix each test module
with its package: `test_mace_core_scaffold.py`, not `test_scaffold.py`.

## Dependencies

The scaffolds declare only the dependency edge this layout is about:
`mace-torch-v1` and `mace-jax` require `mace-core`. Torch and jax themselves
are declared by the first ticket that imports them, so a torch environment does
not pull in jax to install a package containing no code.
