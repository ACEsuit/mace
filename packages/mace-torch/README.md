# mace-torch-v1

The PyTorch stack for MACE v1: models, training, kernel backends and
deployment. It depends on `mace_core` and torch, and is the primary training
and research backend.

Distribution `mace-torch-v1`, import name `mace_torch`. The distribution name
carries a suffix only because the frozen legacy package holds `mace-torch`; it
is never published and loses the suffix at RET-6. Scaffold only for now.
