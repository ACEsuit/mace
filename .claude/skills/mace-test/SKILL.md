---
name: mace-test
description: Select and run the relevant native MACE test suites and interpret capability skips.
---

Read [CONTRIBUTING](../../../CONTRIBUTING.md) and the relevant CI job.
Its run-tests inputs are the reproduction recipe. Use `python -m pytest` with
explicit paths and marker selection. `tests/conftest.py::CAPABILITY_PROBES`
is the authority; do not maintain a second capability list.
`MACE_REQUIRE_CAPS` turns missing promised capabilities into failures. Do not
list a capability whose tests are all excluded by `-m`. Keep the frozen
references and tolerance table unchanged when diagnosing a mismatch.
