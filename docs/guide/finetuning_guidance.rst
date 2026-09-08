.. _finetuning_guidance:

********************
Fine-tuning Guidance
********************

.. note::

    This guide summarises the recommendations from Tamás Lajos Tompa,
    Eszter Varga-Umbrich, Ilyes Batatia, Alin M. Elena, Noam Bernstein and
    Gábor Csányi, `Fine-tuning MLIP foundation models: strategies for
    accuracy and transferability <https://arxiv.org/abs/2606.12704>`_ (2026).
    The `supplementary dataset
    <https://huggingface.co/datasets/ev-tlt/MACE_finetuning_supplementary>`_
    contains the data, checkpoints and evaluation scripts used in the study.
    For the mechanics of each protocol, see the :ref:`naive/overview <finetuning>`,
    :ref:`multihead replay <multihead_finetuning>` and :ref:`LoRA <lora_finetuning>` guides.

The central lesson is that "training setup" prerequisites routinely matter more than the
choice between naive, LoRA, freezing or replay:

#. the quality of the foundation model you start from,
#. correct atomic reference energy (``E0s``) initialisation, and
#. sensible fine-tuning-specific optimiser hyperparameters.

Once these prerequisites are controlled, choose the protocol according to how
the model will be deployed.

.. contents:: On this page
   :local:
   :depth: 2


Prerequisite 1: Start from the strongest foundation model
---------------------------------------------------------

Foundation-model quality propagates directly into fine-tuned accuracy and
out-of-distribution transfer. In the benchmarks, OMat24-based foundations
(``medium-omat-0``, ``small-omat-0``, MACE-MH-1) outperformed the older
MPTraj-based ``MACE-MP-0`` family **by more than the differences between most
fine-tuning methods**.

This holds even when the target chemistry is absent from pretraining: OMat24
contains no gas-phase molecules, yet OMat24-based models fine-tune to molecular
and reactive tasks *better* than MPTraj-based ones. Richer coverage of inorganic
chemical space transfers broadly. Additionally, MACE-MH-1 is trained on a broader range
of (including organic) chemistry as well, and is the current recommended default.

.. tip::

    Use the newest, largest foundation model your compute budget allows. For
    example, use ``--foundation_model="mh-1"`` for MACE-MH-1.


Prerequisite 2: Use consistent atomic reference energies (E0s)
--------------------------------------------------------------

MACE predicts atomisation energies relative to per-element isolated-atom
reference energies (``E0s``). When the fine-tuning data are computed at a
different level of theory than pretraining, the dominant source of label
inconsistency is the shift in these references. Getting the ``E0s`` wrong is
enough to make a model that looks converged on validation RMSE **fail in
molecular dynamics**.

There are three ways to set the ``E0s``, in decreasing order of preference:

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Method
     - ``--E0s`` value
     - When to use
   * - **Explicit isolated-atom DFT**
     - dict / JSON
     - Best when you control the DFT workflow. Run single-atom calculations,
       spin-polarised, in an asymmetric box large enough that periodic images
       do not interact, at the *same* level of theory as the fine-tuning data.
       Pass as a dict, e.g.
       ``--E0s="{1: -13.733, 6: -1029.185, 8: -2041.219}"``.
   * - **Model-aware reestimation**
     - ``"estimated"``
     - **Recommended default when explicit isolated-atom energies are
       unavailable.** Uses the pretrained
       model's own predictions to solve for the per-element offset that aligns
       the model's E0s with your data.
   * - **Training-set averaging**
     - ``"average"``
     - **Avoid for fine-tuning.** Fits a per-element average over *interacting*
       configurations, which is not the isolated-atom reference and produces unphysical results.

Why reestimation works
~~~~~~~~~~~~~~~~~~~~~~

Model-aware reestimation (``--E0s="estimated"``) exploits the fact that the
pretrained model already predicts interaction energies reasonably well. It
solves a small linear least-squares problem for per-element corrections
:math:`\Delta E_0^Z` that minimise the residual between the model's predictions
and your reference energies. In the limit of a perfect pretrained model it
recovers the exact target-DFT ``E0s``.

It works even when the absolute atomic baselines are enormous relative to the
interaction energies, for example in the case of all-electron energy datasets.

.. warning::

    **MACE-MH1 and other models with learnable E0 biases are a special case.**
    Their isolated-atom reference is shared between the explicit ``E0s`` and
    learnable bias terms, so setting the ``E0s`` correctly does not by itself
    guarantee accurate isolated-atom energies. If your application depends on
    absolute atomic energies (e.g. defect formation energies), also include
    **true isolated-atom configurations as explicit training points**, even when
    you initialise the ``E0s`` from DFT.

.. tip::

    A high *initial* error on your dataset (> 500 meV/atom) is the classic
    symptom of an ``E0`` mismatch. Fix the ``E0s`` before touching anything else.


Prerequisite 3: Use stable hyperparameters
-------------------------------------------

Fine-tuning needs different optimiser settings from training from scratch, and
the best settings differ per method.

.. list-table:: Recommended starting hyperparameters (per the study)
   :widths: 22 16 14 14 18
   :header-rows: 1

   * - Method
     - Learning rate
     - EMA decay
     - Grad clip
     - Trainable params
   * - From-scratch
     - ``1e-2``
     - ``0.99``
     - ``10.0``
     - 100 %
   * - **Naive**
     - ``1e-3``
     - ``0.999``
     - ``1.0``
     - 100 %
   * - Layer freezing
     - ``1e-3``
     - ``0.999``
     - ``1.0``
     - ~5 %
   * - **LoRA** (r = 4–64)
     - ``1e-2``
     - ``0.99``
     - ``10.0``
     - 2.5–30 %
   * - **Multihead / Pseudolabel**
     - ``1e-4``
     - ``0.9999``
     - ``1.0``
     - 100 %

Three rules of thumb hold across the board:

- **Weight decay = 0.** Weight decay pulls parameters toward *zero*, i.e. *away*
  from the pretrained solution — exactly counter to the point of fine-tuning.
  Use ``--weight_decay=0.0``.
- **Constant loss weights, no two-stage schedule.** The force-then-energy
  schedule common in from-scratch training *can destabilise* fine-tuning at the
  transition point, because the model already starts with accurate forces and a
  coherent PES. Use constant target weights throughout — for example
  ``--energy_weight=10 --forces_weight=10``.
- **Finetuning requires lower learning rates** (an order of magnitude below
  the default), **LoRA tolerates higher learning rates** (same as default),
  while **multihead replay needs a low learning rate** (``1e-4``). MACE applies the lower multihead
  learning rate automatically; can be overrided with ``--force_mh_ft_lr=True`` (not recommended).


Choose a method by deployment scope
===================================

Once the prerequisites are met, **most methods reach strong target-task
accuracy and consistently beat training from scratch.** The practical
distinction is not in-domain accuracy, but how the model behaves *away* from
the fine-tuning distribution. Pick based on what the model needs to do:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Your goal
     - Recommended method
   * - **Single system, OOD accuracy irrelevant** (e.g. one reaction, one
       solvation environment)
     - **Use naive fine-tuning.**
   * - **Transfer within a related chemical family** (compositions/structures
       near the training data)
     - Naive or LoRA are fine in-family; **using multihead replay** gives the best transfer to
       the most distant members.
   * - **Transfer to a different chemistry / broad screening / structure search**
     - **Multihead replay** (original-label or pseudolabel). The only methods
       tested that preserve pretraining-distribution accuracy *and* the
       short-range repulsive wall.
   * - **Must broadly preserve foundation-model behaviour** (no catastrophic forgetting,
       robust repulsion in unexplored regions)
     - **Multihead replay.** (original-label or pseudolabel) Naive and LoRA drift furthest from the pretrained
       distribution.

Default recommendation
----------------------

**Start with naive fine-tuning** unless you already know it won't be enough — if
you need broad OOD robustness or want to preserve foundation-model behaviour, go
straight to multihead replay. For most single-system narrow tasks, though, naive is
accurate, efficient, and the strongest baseline.

Ideally, if compute allows, run multihead replay alongside
naive and compare the two on the observable you actually care about (see
*Validate beyond pointwise RMSE* below). This is the surest way to choose, since
replay sometimes wins even when naive looks fine/better on RMSE.


Method notes
============

Full-parameter (naive) fine-tuning
----------------------------------

Restart from the foundation weights and continue training on the target data
with a reduced learning rate and increased EMA decay. Best convergence for
single-system applications. See :ref:`finetuning`.

LoRA
----

For MLIPs, LoRA's value is **capacity control, not parameter efficiency**
(MACE models are small). Low ranks restrict departure from the pretrained
solution; higher ranks recover flexibility for diverse tasks. Empirically the
effect is real but modest: higher ranks help on broad datasets (clearest on
SPICE), and LoRA reduces forgetting slightly versus naive — but far less than
replay. LoRA is a *width-wise* constraint (all layers change, within a low-rank
subspace), which the study found generally more effective for MACE than the
*depth-wise* constraint of layer freezing. See :ref:`lora_finetuning`.

Multihead replay (and pseudolabel replay)
-----------------------------------------

Train on the target task and a structurally diverse "replay" set simultaneously,
with separate readout heads.

- **Pseudolabel replay** (Mode 2): replace the replay set's DFT labels with the
  foundation model's *own* predictions. This **decouples the replay structures
  from the original pretraining corpus** — any structurally diverse dataset can
  serve as the replay source — and gives target-task accuracy
  indistinguishable from original-label replay.
- **Structural diversity matters more than label provenance.** Element-matched
  OMat24, random MPTraj, and their combination gave essentially identical
  results when all were pseudolabelled.
- **Cost:** roughly 3–15× more training compute than naive (more data per epoch,
  plus the lower learning rate).

See :ref:`multihead_finetuning`.

Layer freezing
--------------

Freezing keeps part of the foundation model fixed and trains only the remaining
layers, controlled by how many blocks you freeze:

- **freeze=6** freezes everything but the readouts. This preserves the embedding
  and message-passing layers, but is **too restrictive**. Not recommended.
- **freeze=5** freezes the embedding and interaction layers while leaving the
  product and readout layers trainable. This is the **preferable** setting.


Combined replay + LoRA
----------------------

In nearly every setting tested, replay + LoRA was *worse* than either alone
(over-regularisation). It is not recommended.


Validate beyond pointwise RMSE
==============================

Low validation energy/force RMSE does **not** guarantee a physically sensible
potential. For example, these failure modes are invisible to pointwise metrics:

- **MD stability** — whether a long dynamics run stays physical instead of
  blowing up: atoms drifting into unphysical geometries, runaway energy or
  temperature spikes, or outright simulation crashes. As the trajectory explores
  configurations the fine-tuned model rarely saw, small force errors can compound
  and drive the system off the physical manifold, even when held-out RMSE is low.
  This is worst for naive, intermediate for LoRA and layer freezing, and least
  for replay. Measure it by running long MD (e.g. under a temperature ramp).
- **PES "holes" / repulsive-wall failures** — regions where the learned
  many-body term turns attractive at short range. Notably, replay finetuning suppresses
  them. (The explicit ZBL pair-repulsion term does not prevent these many-body
  holes.)

For the specific deployment, validate on the observable you actually care
about: RDFs from MD, NEB barrier profiles, MD stability under a temperature
ramp, etc. — not just held-out test set RMSE.


Quick reference: practical checklist
====================================

#. **Foundation model:** newest/broadest available (``mh-1`` or
   better). ☐
#. **E0s:** explicit isolated-atom DFT if you can; otherwise
   ``--E0s="estimated"``. Never ``"average"``. ☐
#. **Sanity check:** initial dataset error < 500 meV/atom (if not, make sure E0s are correct). ☐
#. **Optimiser:** ``--weight_decay=0.0``, method-appropriate hyperparameters. ☐
#. **Method:** naive for a single system; multihead replay for breadth / robustness. ☐
#. **Validate** on your downstream task. ☐

A worked example using an S\ :sub:`N`\ 2 reaction is available as the
:download:`reliable fine-tuning of MACE foundation models notebook
<../examples/reliable_finetuning_of_mace_foundation_models.ipynb>`. It includes an
``E0`` sensitivity study, method-specific training commands and checks that go
beyond held-out RMSE. The notebook can also be `opened in Google Colab
<https://colab.research.google.com/github/ACEsuit/mace/blob/docs/docs/examples/reliable_finetuning_of_mace_foundation_models.ipynb>`_.

Citation
========

If you use these recommendations, please cite:

    Tamás Lajos Tompa, Eszter Varga-Umbrich, Ilyes Batatia, Alin M. Elena,
    Noam Bernstein and Gábor Csányi. *Fine-tuning MLIP foundation models:
    strategies for accuracy and transferability*. arXiv:2606.12704 (2026).
    https://doi.org/10.48550/arXiv.2606.12704
