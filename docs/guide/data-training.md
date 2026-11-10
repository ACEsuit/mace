# Data, training and fine-tuning

An extended XYZ supplies atomic numbers, positions, cell and periodicity. The
configured keys map energies, forces, stress and other labels into the dataset.
Keep physical units consistent: eV, Å, eV/Å and eV/Å³. A nonperiodic molecule has
no physical stress; its arbitrary neighbour-list padding cell is not a volume
for a stress label. Missing labels are masked per observable.

Declare one or more named heads in `data.heads`. Each head has its own files,
E0 policy and loss weights. Isolated-atom E0s need labelled isolated atoms;
least-squares E0s are an explicit alternative. Keep train/validation splits
stable when comparing architectures. Dataset statistics set normalisation;
they are saved with the run, not recomputed during evaluation.

```sh
mace config validate --config run.yaml
mace train --config run.yaml --device cpu
mace data prepare --train_file train.xyz --output prepared --compute_statistics
mace model export-config model.json --output resolved.yaml
```

See [configuration](../reforge/configuration.md) for batching, preprocessing,
DDP, losses, optimizers, stage two, EMA, multihead fine-tuning, replay and LoRA.
The head selector is explicit when deploying a multihead artifact. A foundation
fine-tune must preserve the source architecture unless the corresponding
initialisation path supports its change. Use local converted artifacts while
native public foundation URLs remain pending.

Distributed processes must share the same dataset and split, while each rank
gets its own batches. Test a small CPU run before increasing workers. Resume
from training state when preserving optimizer/scheduler progress matters;
a deployed inference artifact is not a complete training checkpoint.
