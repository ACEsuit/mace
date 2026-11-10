---
name: mace-eval
description: Evaluate a native MACE checkpoint on atomic structures using the CLI or ASE.
---

Read [the command guide](../../../docs/reforge/cli.md) for exact flags.
Use `mace eval` for a structure file and `mace_torch.calculators.MACECalculator`
for ASE. A native checkpoint is JSON plus safetensors, or a deployment bundle.
It does not need legacy, e3nn or a backend-specific weight conversion.
Check species, requested properties, dtype and device before running. GPU and
optional backend availability are explicit; report actual skips or fallbacks.
Foundation artifacts may require network access and an available native registry.
