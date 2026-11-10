---
name: mace-frozen-parity
description: Verify native MACE predictions and derivatives against committed frozen references.
---

Run `python -m pytest tests/retirement tests/parity` with the appropriate
capability selection. The two energy anchors include both ZBL placements and
all nonempty parameter gradients. Coordinate maps are frozen data, not a
second model implementation. Electrostatics cases retain their external-oracle
provenance. Diagnose precision, units and normalization before changing code;
never loosen a tolerance or regenerate an expected result to pass a test.
Historical live comparisons are recoverable from the recorded git revision.
