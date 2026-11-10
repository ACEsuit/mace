---
name: mace-convert-export
description: Migrate a legacy checkpoint explicitly, select a head, or export a native deployment bundle.
---

Use [the migration guide](../../../docs/reforge/migration.md) and
[deployment support](../../../docs/reforge/integration_support.md).
Legacy pickle extraction runs in an explicitly selected external interpreter;
its neutral tensor output is imported by the native stack. Never load a legacy
pickle into the normal training/inference environment. Do not promise support
for an unverified model/dtype combination.
Use `mace model select-head`, `mace model convert-device` and `mace export lammps`
as appropriate. Backend selection uses the canonical weights directly.
Real LAMMPS, OpenMM, libmace and GROMACS tests require their external binaries;
contract tests do not. Symmetrix integration is pending a design decision.
