# Small native workflows

Run from the repository root after installing core and torch. The `md22`,
`ani1x` and `water` names select molecular or periodic example geometry;
**the default data are synthetic harmonic smoke fixtures**, not samples from
MD22 or ANI1x and not validated liquid-water labels.

```sh
python examples/reference_workflow.py prepare water /tmp/mace-water
mace train --config /tmp/mace-water/run.yaml --device cpu
```

The CLI reports the deployed JSON/safetensors pair. Use that JSON path for
`mace eval --model MODEL.json --configs /tmp/mace-water/synthetic.xyz --output predicted.xyz`
and for a short thermostat check:

```sh
python examples/reference_workflow.py nvt --model MODEL.json --structures /tmp/mace-water/synthetic.xyz --output nvt.json
```

Use `prepare md22 OUTPUT` or `prepare ani1x OUTPUT` for the other smoke cases.
Pass `--dataset /absolute/path/to/labelled.xyz` to train on your own MD22,
ANI1x or liquid-water subset. Adapt E0 policy, keys, split and epochs in the
written config to that dataset. Downloading and licensing those datasets is
explicit work; these scripts never fetch them implicitly.

An NVT run on a converted foundation model uses the same command with its
local native artifact and appropriate structures. Public native foundation
hosting is pending. The tiny trained model only tests execution; its synthetic
labels do not make it a useful dynamics potential.

`python -m pytest examples/tests` executes all three CLI trainings, evaluates
their saved artifacts and runs the periodic-water NVT smoke in CI.
