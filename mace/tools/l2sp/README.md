# L2-SP fine-tuning

L2-SP (L2 starting-point regularization) adapts a pretrained interatomic potential
to a material or target property while discouraging large changes from its
starting weights. The aim is to improve accuracy with limited training data
while retaining useful information from the foundation model and reducing
overfitting. Its targeted variant, L2-TSP, combines regularization with layer
freezing: L2-SP discourages large changes in trainable weights, while freezing
keeps the remaining weights exactly fixed. This concentrates adaptation in
selected layers and reduces the number of trainable parameters, providing an
additional constraint when data are limited. Grandel et al. [1] demonstrate
this strategy for phonon, thermal, and elastic properties using small
material-specific datasets.

Enable L2-SP with `--l2sp=True`, set `--l2sp_strength`, and use `--freeze` to
select which parts of the model stay fixed. The optimizer determines how
regularization is applied.

## Example: targeted L2-SP (L2-TSP)

The targeted method of Grandel et al. [1] adapts the node embedding and first
interaction block of MACE-MP-0b3. The example below uses that layer selection
and the hyperparameters from
[Equitrain's paper recipe](https://bamescience.github.io/equitrain/phonon-finetuning-paper/).
Replace the model and dataset paths with your own:

```bash
mace_run_train \
    --name=material_l2tsp \
    --foundation_model=path/to/mace-mp-0b3.model \
    --train_file=material_train.xyz \
    --valid_file=material_valid.xyz \
    --E0s=foundation \
    --foundation_model_elements=True \
    --foundation_model_readout=True \
    --multiheads_finetuning=False \
    --l2sp=True \
    --l2sp_strength=10.0 \
    --freeze=products,interactions.1,readouts \
    --optimizer=adamw \
    --amsgrad=False \
    --weight_decay=0.0 \
    --lr=0.01 \
    --loss=huber \
    --huber_delta=0.01 \
    --energy_weight=10 \
    --forces_weight=100 \
    --stress_weight=1000 \
    --max_num_epochs=200 \
    --default_dtype=float64
```

Here, `--amsgrad=False` selects AdamW without AMSGrad; MACE's default remains
AMSGrad enabled. EMA and stage two are disabled by default. The example provides
a starting configuration. Matching Equitrain's complete training trajectory
also requires matching data preparation, loss reductions, atomic reference
energies, batching, gradient clipping, and scheduling.

## Required settings

L2-SP requires a foundation model and transfer of its readout weights. Use
`--foundation_model=...`, `--foundation_model_readout=True`, and
`--multiheads_finetuning=False`.

Set `--l2sp_strength` to an explicit, finite, non-negative value. It controls
regularization towards the starting weights; zero disables regularization
while retaining the chosen frozen layers. Set `--weight_decay=0`, since the
regularization coefficient is supplied through `--l2sp_strength`.

These settings can also be added to an existing YAML training configuration
that supplies the foundation model and datasets:

```yaml
l2sp: true
l2sp_strength: 10.0
freeze: products,interactions.1,readouts
optimizer: adamw
amsgrad: false
weight_decay: 0.0
multiheads_finetuning: false
foundation_model_readout: true
```

## Selecting trainable parameters

`--freeze` accepts the existing integer levels or comma-separated module names.
Named modules and all their parameters are frozen; parameters already frozen
stay frozen. This option also works independently of L2-SP.

For the two-interaction-block model in the example,
`--freeze=products,interactions.1,readouts` freezes all product blocks, the second
interaction, and the readouts. Only `node_embedding` and `interactions.0` remain
trainable. For deeper models, also list each later interaction, such as
`interactions.2`.

To apply L2-SP to all currently trainable parameters, omit `--freeze` or use
`--freeze=0`. Every selected parameter receives the same regularization
coefficient, including embedding and radial-network weights.

## Regularization and optimizer choice

Let `theta_0` denote the fixed starting weights and
`delta = theta - theta_0` their change during training. References are captured
after foundation transfer, element filtering, and model setup. The example's
`--foundation_model_elements=True` retains all foundation-model elements.

For Adam, AdamW, and ScheduleFree, L2-SP applies the optimizer's native
weight-decay rule to `delta`. L-BFGS uses an explicit loss penalty:

| Selection | Regularization behavior |
| --- | --- |
| `--optimizer=adam` | Adds `l2sp_strength * delta` to the gradient before the adaptive update. |
| `--optimizer=adamw` | Applies decoupled decay to `delta`, following Equitrain's residual-AdamW formulation. |
| `--optimizer=schedulefree` | Applies the native AdamWScheduleFree decay rule to `delta`. |
| `--lbfgs` | Adds `l2sp_strength / 2 * sum(delta**2)` to the full-dataset loss. |

For the first three optimizers, gradient clipping applies to the data gradient
before regularization. For L-BFGS, the penalty is included once per full-dataset
loss evaluation, including line-search evaluations, and clipping applies to the
combined gradient.

The usual `--lr_params_factors` apply to Adam, AdamW, and ScheduleFree. L-BFGS
follows MACE's existing behavior: one parameter group with an initial step scale
of 1.0, independent of `--lr` and the per-group factors.

Both reduced-CG and unreduced-CG models are supported. Regularization uses the
training model's parameter coordinates, so the same `--l2sp_strength` can produce
different behavior across CG representations. These differences are expected.

## Checkpoints and metrics

Training checkpoints store the original reference weights along with the
optimizer state. `--restart_latest` restores those references, so regularization
continues towards the original starting point. Keep L2-SP enabled and retain the
same strength, selected parameters, optimizer, and model representation when
resuming. Missing or incompatible L2-SP checkpoints cause an error.

Exported models are ordinary MACE models and need no additional L2-SP component
for inference.

Training metrics include `l2sp_squared_distance`, the sum of squared differences
from the reference weights over selected parameters after each optimizer step.
Validation loss and model selection use the data loss. The logged training loss
is the data loss for Adam, AdamW, and ScheduleFree; for L-BFGS it includes the
explicit penalty.

## Supported combinations

The implementation supports single-head MACE/ScaleShiftMACE, all existing MACE
optimizers, EMA, stage two, and distributed data parallel training. LoRA,
multihead replay, new embeddings, and OEQ training are rejected explicitly.

## References

1. Grandel, J., Benner, P. & George, J. Targeted fine-tuning of machine-learning
   interatomic potentials for phonons and phase transitions. *npj Computational
   Materials* **12**, 354 (2026).
   [doi:10.1038/s41524-026-02338-w](https://doi.org/10.1038/s41524-026-02338-w).
