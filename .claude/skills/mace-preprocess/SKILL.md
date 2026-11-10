---
name: mace-preprocess
description: Prepare native dataset shards and statistics for MACE training.
---

Use `mace data prepare`; consult [the command guide](../../../docs/reforge/cli.md)
for the requested format and key mapping. Preserve physical cells, periodicity,
labels and missing-label masks. Data readers belong to mace_core and return
framework-free configurations. CPU operation requires no accelerator. Validate
one produced shard through its public reader before starting a large training.
