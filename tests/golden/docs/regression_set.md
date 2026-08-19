# The end-to-end regression training set

Target: `regression_set`, part of `all`.

The training contracts need a dataset whose loss can actually go down. The
anchors' `tiny_train.xyz` cannot serve: it exists to make an anchor
reproducible, not to be learnable, so a training run over it says nothing
about whether training works.

This set is generated from a **closed-form labeller** rather than from a
model, which is what makes it a contract rather than a second golden. The
energy and forces of every configuration are an analytic function of the
positions, so "the trainer reduced the error" is a statement about the
trainer and not about whichever checkpoint produced the labels.

Regeneration self-checks before it writes: it differentiates the labeller
numerically and compares against the analytic forces, and prints the worst
disagreement across the set. A labeller whose forces are not the gradient of
its energy would otherwise produce a dataset that no correct trainer can fit,
and the failure would look like a training regression.

## Where it lives

`datasets/regression_train.xyz`, deliberately **not** under `fixtures/`.
That directory is the evaluation set and every file in it must have a
manifest row; a training set there is described by nothing, loaded by
`load_fixtures` as if it were a structure to evaluate, and breaks the
guard that keeps the manifest and the directory in step.

## Running

No marker; runs in the ci-core `unit` job with everything else.
