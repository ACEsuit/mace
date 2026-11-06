# mace-torch-v1

The PyTorch stack for MACE v1: models, training, kernel backends and
deployment. It depends on `mace_core` and torch, and is the primary training
and research backend.

Distribution `mace-torch-v1`, import name `mace_torch`. The distribution name
carries a suffix only because the frozen legacy package holds `mace-torch`; it
is never published and loses the suffix at RET-6.

## Optional LES solver

Install the external latent Ewald solver from a V1 checkout with

```bash
python -m pip install ./packages/mace-core './packages/mace-torch[les]'
```

The `les` extra selects the solver commit used by the frozen LES references.
Set `model.model` to `les` and configure the multipoles under `model.les`.
The training CLI also accepts the historical model name `MACELES` and a
`--les_arguments` YAML file. Request Born effective charges with
`mace eval --compute_bec` or `MACECalculator(..., compute_bec=True)`.

MACE readouts supply the charges to the solver. A new training configuration
rejects `use_atomwise=True`. Conversion records that setting when present
in an old checkpoint and uses the charge readouts that produced its predictions.
