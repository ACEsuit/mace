# The gradient goldens

Target: `gradients`, part of `all`.

`references/tiny_{scaleshift,mace}_train_step_grad_fp64.json` hold one
forward+backward of an energy+forces loss on the first four labelled
structures of `fixtures/tiny_train.xyz`, at fp64 on CPU, with the committed
weights — so the numbers depend on nothing that was randomly drawn. They
exist because a loss-decrease smoke test and a final-error table are both
green while `d(loss)/d(theta)` is wrong by the size of initialisation noise.

What is committed is a **digest**, not the 37,704-element gradient vector:
per parameter, its shape and count plus `sum`, `abs_sum`, `sq_sum` and
`sum_k g_k·cos(k+1)`. The raw vectors would be about 1.5 MB in every clone —
more than both checkpoints together — and unreadable in a diff. The four
reductions are chosen so nothing plausible survives all of them: the sum
catches a sign flip, the other two a magnitude change that cancels in the
sum, and the positional projection catches a **permutation**, which the first
four cannot see at all and which is what an irreps-layout mistake produces.
`cos` of an integer rather than a seeded vector, so a reimplementation in
another framework can reproduce it without sharing an RNG.

Measured resolution: the response is linear with a gain of 0.49, so at the
1e-6 row the golden resolves a change of about 2e-6 in a single weight, six
orders below the weights themselves. A test asserts that, and another asserts
the permutation claim, rather than leaving either as a paragraph.

The same run records `gradcheck`/`gradgradcheck` pass-flags for the position
and strain derivatives, at the fp64 row rather than at torch's much looser
defaults.

This family consumes the anchors it digests rather than owning them; it reads
them through the shared anchor registry.

## Running

No marker; runs in the ci-core `unit` job with everything else.
