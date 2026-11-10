---
name: mace-reforge
description: Shared working context for the MACE Reforge rewrite (MACE v1.0). Use whenever the task touches the rewrite - any ticket from the reforge board, anything under packages/, the frozen legacy mace/ package, the golden or parity test suites, or a design question about the new architecture. Load this before writing code for a reforge ticket.
---

# MACE Reforge on V1

Read [CLAUDE.md](../../../CLAUDE.md), [contributing](../../../CONTRIBUTING.md)
and the [reference branch README](../../../README.md).

The four distributions under packages/ are the implementation. Legacy mace/
and mace-launcher are retired. Numerical reference values remain frozen in
tests/golden, and native checkpoints and gradient maps in tests/retirement
exercise them without importing legacy. Historical feature-inventory pins are
provenance, not executable coverage.

Use ruff for formatting/lint and ty for package types. Run package tests plus
the relevant frozen and architecture gates. Never change a tolerance or
regenerate numerical references to make a mismatch pass. Consult the backend,
numerics and goldens companion skills for their contracts.

`v1-reference` preserves an AI-generated prototype and will not be maintained.
Reviewed implementation PRs target `mace-reforge`. Check the user-facing guides
for unsupported model and deployment paths. Local validation does not establish
upstream review or release readiness.
