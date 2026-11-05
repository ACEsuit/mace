# Remaining published foundation references

Target `foundation-roster` captures the fifteen artifacts that had no individual
reference: twelve MP aliases, OFF23 large and Polar medium/large. Existing MP,
OFF, multi-head, OMOL, MDP and Polar-small references remain owned by their
original targets and are never rewritten here.

Each capture uses the frozen legacy loader on CPU with explicit float64,
records the source URL and the SHA-256 of the file actually loaded, and pins
the full calculator surface. OFF uses the three existing molecular fixtures;
MP and Polar use the six existing energy fixtures. No artifact or geometry
is created by native conversion code.

Legacy reproduces these references at `fp64_cpu_reference`. Native conversion
uses the established `fp32` row for artifacts whose stored constants were
rounded at float32, as FM-2 specifies. Polar's historical stress omits the
long-range cell derivative: its frozen value is retained here, while native
stress keeps the already agreed energy-derivative contract and its separate
finite-difference tests. Other Polar outputs remain parity obligations.

These are first captures, committed separately from implementation work.
They establish numerical evidence; they do not publish converted artifacts
or change any hosting, license, naming or registry contract.

```bash
MACE_CI_ALLOW_NETWORK=1 python tests/golden/regenerate.py \
  --target foundation-roster --i-know-what-i-am-doing
MACE_CI_ALLOW_NETWORK=1 python -m pytest tests/golden/test_foundation_roster_goldens.py
```

The target is opt-in and excluded from `all`, because loading the models can
require downloads and Polar requires its legacy optional dependency.
