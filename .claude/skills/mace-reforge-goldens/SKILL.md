---
name: mace-reforge-goldens
description: Work with the MACE Reforge frozen oracle - the committed golden references in tests/golden and the in-process parity suite. Use when adding or reading goldens, choosing a tolerance, debugging a golden or parity mismatch, or when asked to regenerate a reference. Also use before touching anything that changes energies, forces or stress.
---

# mace-reforge-goldens

Existing reference JSON, fixtures, tolerance rows and legacy pickle snapshots
under tests/golden are immutable scientific evidence. Native runtime checks
are in tests/retirement and the frozen electrostatics tests in tests/parity.
The in-tree legacy oracle has been removed. Use the documented historical
checkout and pinned interpreter for any explicitly requested recapture.
Never widen a tolerance or refresh a golden to hide a defect. Compare energies,
forces, stress and all-parameter gradient digests before changing numerics.
