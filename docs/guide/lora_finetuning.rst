.. _lora_finetuning:

*****************
LoRA Fine-tuning
*****************

Introduction
============

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that restricts the parameter space the model can explore during adaptation. Instead of updating all model weights, LoRA injects small, low-rank decomposition matrices into each linear layer while freezing the original (base) weights entirely. Only the injected LoRA parameters are trained.

By constraining updates to a low-rank subspace, LoRA provides two key advantages over standard (naive) fine-tuning:

- **Reduced overfitting risk**: The restricted parameter space acts as an implicit regulariser, which is especially useful when fine-tuning on small datasets.
- **Reduced catastrophic forgetting**: Because the base weights are frozen and only low-rank perturbations are learned, the model is less likely to drift far from the foundation model's learned representations.

The final saved model has the exact same architecture as the original — LoRA weights are merged into the base at save time.

LoRA can be combined with both the naive and :ref:`multihead replay <multihead_finetuning>` fine-tuning protocols. However, we recommend using LoRA with **naive fine-tuning**, since the multihead replay already provides strong regularisation through data replay, making the additional constraint from LoRA less necessary.

How It Works
============

Equivariant LoRA
----------------

Standard LoRA decomposes a weight update as :math:`\Delta W = B A`, where :math:`A` and :math:`B` are low-rank matrices. MACE extends this idea to equivariant linear layers (`o3.Linear`) so that **O(3) equivariance is preserved by construction**.

For an equivariant layer with input irreps :math:`I_{\text{in}}` and output irreps :math:`I_{\text{out}}`, the LoRA bottleneck is built from the irreducible representations shared between input and output. For each shared irrep, ``rank`` copies are allocated to form the bottleneck irreps. This ensures that the low-rank path respects rotational symmetry throughout.

The effective output of a LoRA-wrapped layer is:

.. math::

    y = W_{\text{base}}\, x + \frac{\alpha}{r}\, B(A(x))

where :math:`r` is the rank, :math:`\alpha` is the scaling factor, and :math:`A` and :math:`B` are the low-rank equivariant linear maps.

Supported Layer Types
---------------------

LoRA wrappers are injected into three types of layers found in MACE models:

- **Equivariant linear layers** (``o3.Linear`` and ``cuet.Linear``): Wrapped by ``LoRAO3Linear``, which preserves O(3) equivariance through symmetry-constrained bottleneck irreps.
- **Dense linear layers** (``nn.Linear``): Wrapped by ``LoRADenseLinear``, which uses standard low-rank decomposition.
- **Fully-connected network layers** (e3nn ``FullyConnectedNet`` internal layers): Wrapped by ``LoRAFCLayer``, which patches the weight matrix of each MLP layer.

Parameter Freezing
------------------

When LoRA is injected, all base model parameters are automatically frozen (``requires_grad=False``). Only the LoRA matrices (named ``lora_A`` and ``lora_B`` in the parameter tree) receive gradients during training.

Initialisation
--------------

The ``lora_B`` matrices are initialised to zero and the ``lora_A`` matrices to small random values (std = 1e-3). This means the model output is identical to the original foundation model at the start of training — LoRA begins as an identity perturbation.

Usage
=====

LoRA fine-tuning is enabled by adding three flags to the ``mace_run_train`` command:

- ``--lora=True``: Enable LoRA.
- ``--lora_rank``: Rank of the LoRA matrices (default: 4).
- ``--lora_alpha``: Scaling factor (default: 1.0).

Basic LoRA Fine-tuning
----------------------

To fine-tune a foundation model with LoRA on a new dataset:

.. code-block:: bash

    mace_run_train \
        --name="MACE_lora" \
        --foundation_model="medium_omat" \
        --train_file="train.xyz" \
        --valid_fraction=0.05 \
        --test_file="test.xyz" \
        --lora=True \
        --lora_rank=4 \
        --lora_alpha=1.0 \
        --energy_weight=1.0 \
        --forces_weight=1.0 \
        --E0s="estimated" \
        --lr=0.005 \
        --weight_decay=0.0 \
        --ema \
        --ema_decay=0.995 \
        --amsgrad \
        --clip_grad=10.0 \
        --batch_size=2 \
        --max_num_epochs=6 \
        --default_dtype="float64" \
        --device=cuda \
        --seed=3

Key Parameters
==============

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``--lora``
     - ``False``
     - Enable LoRA fine-tuning. When set to ``True``, low-rank adapters are injected into all eligible layers.
   * - ``--lora_rank``
     - ``4``
     - Rank of the LoRA matrices. Higher rank increases capacity but also the number of trainable parameters. Typical values are 2–16.
   * - ``--lora_alpha``
     - ``1.0``
     - Scaling factor for the LoRA update. The effective scaling applied to the low-rank path is :math:`\alpha / r`. Increasing ``alpha`` relative to ``rank`` amplifies the LoRA contribution.

Weight Merging
==============

At save time, LoRA weights are **automatically merged** into the base model weights. The saved checkpoint is a standard MACE model with no LoRA overhead — it has the same architecture and can be loaded and used exactly like any other MACE model.

The merging operation computes the fused weight:

.. math::

    W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r}\, W_B\, W_A

and writes it back into the base layer, then removes the LoRA wrapper. This means:

- The saved model requires **no extra dependencies** for inference.
- There is **no inference overhead** compared to a standard MACE model.
- The model can be used directly with the :ref:`ASE calculator <ase>`, :ref:`LAMMPS <lammps>`, :ref:`OpenMM <openmm>`, or any other interface.

Python API
==========

For programmatic usage, LoRA can be injected into any MACE model directly:

.. code-block:: python

    from mace.modules.lora import inject_lora, merge_lora_weights

    # Inject LoRA adapters
    inject_lora(model, rank=4, alpha=1.0)

    # ... train the model ...

    # Merge LoRA weights into base model before saving
    merge_lora_weights(model)

The ``inject_lora`` function accepts the following arguments:

- ``module``: The MACE model to modify.
- ``rank`` (int): Rank of the LoRA matrices.
- ``alpha`` (float): Scaling factor.
- ``wrap_equivariant`` (bool): Whether to wrap equivariant ``o3.Linear`` / ``cuet.Linear`` layers. Default: ``True``.
- ``wrap_dense`` (bool): Whether to wrap dense ``nn.Linear`` and e3nn FC layers. Default: ``True``.
- ``cueq_config``: Optional cuequivariance configuration object for creating cueq-compatible LoRA layers.

The ``merge_lora_weights`` function folds all LoRA adaptations into the base weights and replaces each wrapper with the original (now updated) layer. After merging, all parameters have ``requires_grad=True``.

Tips for Successful LoRA Fine-tuning
=====================================

- **Start with the default rank**: A rank of 4 is a good starting point. Increase to 8 or 16 if the model is underfitting; decrease to 2 if you have very little data and are overfitting.
- **Adjust alpha and rank together**: The effective LoRA scaling is :math:`\alpha / r`. If you double the rank, consider doubling alpha to maintain the same effective scaling, then tune from there.
- **Prefer naive over multihead**: LoRA is most useful with naive fine-tuning, where its regularisation effect helps prevent both overfitting and catastrophic forgetting. With :ref:`multihead replay <multihead_finetuning>`, the replay data already provides strong regularisation, so the additional constraint from LoRA is less beneficial.
- **Monitor trainable parameter count**: The training log reports the number of trainable parameters before and after LoRA injection. Use this to verify that LoRA is working as expected and to compare different rank settings.
- **Saved models are standard MACE**: After training, the saved model has no LoRA layers — weights are merged automatically. You can use the model in all the same ways as a regular MACE model.
