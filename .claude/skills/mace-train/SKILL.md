---
name: mace-train
description: Compose and execute a native MACE PyTorch training run from a config file.
---

Use `mace train --config run.yaml`. Configuration is file-first; unknown keys
fail. Resolve isolated-atom energies and per-head data sources explicitly.
For legacy YAML, run `mace model migrate` and inspect the translated config.
Use the [configuration guide](../../../docs/reforge/configuration.md).
CPU training needs mace-core and mace-torch; choose CUDA only when available.
ScheduleFree and W&B are optional extras. Write training outputs into the user's
requested work directory; use a short smoke run before an expensive run.
