# Output compatibility

The migration checks cover outputs exposed by the legacy model, ASE calculator
and evaluation CLI. Together these layers exposed **61** distinct names. Their
captured declarations are retained in
`tests/retirement/artifacts/legacy_output_surface.json` so compatibility checks
can run without importing legacy code.

| Layer | Keys | New at this layer |
| --- | --- | --- |
| (a) model forward | 43 | 43 |
| (b) ASE calculator results | 31 | 15 |
| (c) evaluation CLI | 13 | 3 |

`tests/architecture/observable_coverage.py` maps each model output to a V1
observable, derivative or explicit replacement. The checks in
`test_observable_completeness.py` compare these declarations with the frozen
inventory and verify the counts above. The [migration guide](migration.md)
provides the corresponding user-facing names and configuration changes.

Calculator-specific outputs include per-atom energies and stresses, committee
means and variances, magnetic moments and long-range diagnostics. Some CLI
names historically differed from the model names: `BO_contributions`,
`descriptors` and `node_energies`. Compatibility tests cover these boundaries
separately from the model's typed return.

V1's `MACEOutput` contains core fields and declared observable outputs.
Derivatives follow their configured names and signs; the engine handles padding
and per-system ownership from their declarations. See
[configuration](configuration.md) and [evaluation](../guide/evaluation.md)
for requesting energies, forces, stress and response properties.
