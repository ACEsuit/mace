# Install and run

This V1 branch is an unpublished reference. Install from its checkout, rather
than assuming that the public mace-torch wheel contains this code.

```sh
python -m pip install --group dev -e packages/mace-core -e packages/mace-torch
mace --help
mace train --config run.yaml --device cpu
mace eval --model model.json --configs structures.xyz --output predictions.xyz
```

A deployed model is a JSON manifest paired with safetensors weights. Keep both
files together. Read the [configuration guide](../reforge/configuration.md)
before preparing a run, and use the [executed examples](examples.md) for a
small local exercise. [Migration](../reforge/migration.md) maps v0.3 commands.

When reusing a v0.3 checkout, remove the generated top-level
`mace_torch.egg-info` before installing this workspace. Its stale metadata can
shadow the native distribution and hide command/backend entry points even
though importing `mace_torch` succeeds. Native packages use Hatch and do not
generate that top-level setuptools directory.
