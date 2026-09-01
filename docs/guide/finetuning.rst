.. _finetuning:

*********************************
Fine-tuning Foundation Models
*********************************

.. warning::
    Fine-tuning is still experimental and under active development. The API and methods are subject to change.

Fine-tuning is the process of refining a pre-trained model on a new dataset.
This is useful when you want better quantitative performance on a specific task than the available pre-trained models.
Fine-tuning usually leads to significant improvements in performance compared to training a model from scratch.
We have three fine-tuning protocols:

 - The **naive** fine-tuning protocol, where the model is trained on the new dataset just by restarting from the foundation model weights.
 - The **multihead replay** fine-tuning protocol, where the model is trained on the new dataset while replaying a part of the original foundational model training data.
 - The **LoRA** (Low-Rank Adaptation) fine-tuning protocol, where only small low-rank adapters are trained while the base model weights are frozen. See the :ref:`LoRA fine-tuning <lora_finetuning>` guide.

Multihead replay can prevent the catastrophic forgetting that sometimes occurs
during naive fine-tuning. Use it when the fine-tuned model must preserve broad
foundation-model behaviour; for a narrow single-system task, naive fine-tuning
is a strong baseline.

For guidance on selecting a foundation model, setting atomic reference
energies, choosing method-specific hyperparameters and validating physical
behaviour, see :ref:`finetuning_guidance`.

To finetune one of the mace-mp-0 foundation model, you can use the mace_run_train script with the extra argument `--foundation_model=model_type`. 

#################
Naive Fine-tuning
#################

The naive fine-tuning protocol is the simplest way to fine-tune a model.
For example to finetune the small model on a new dataset, you can use:

.. code-block:: bash

    mace_run_train \
        --name="MACE" \
        --foundation_model="small" \
        --multiheads_finetuning=False \
        --train_file="train.xyz" \
        --valid_fraction=0.05 \
        --test_file="test.xyz" \
        --energy_weight=10.0 \
        --forces_weight=10.0 \
        --E0s="estimated" \
        --lr=0.001 \
        --weight_decay=0.0 \
        --scaling="rms_forces_scaling" \
        --batch_size=2 \
        --max_num_epochs=6 \
        --ema \
        --ema_decay=0.999 \
        --amsgrad \
        --clip_grad=1.0 \
        --default_dtype="float64" \
        --device=cuda \
        --seed=3 

Other options are "medium" and "large", or the path to a foundation model. 
If you want to finetune another model, the model will be loaded from the path provided `--foundation_model=$path_model`. The hyperparameters will be automatically extracted from the model.

############################
Multihead Replay Fine-tuning
############################

The multihead replay fine-tuning protocol reduces catastrophic forgetting by
training on the target and replay datasets together. It is recommended when
out-of-distribution robustness or preservation of the foundation model's
behaviour matters.

For more information on the multihead replay fine-tuning protocol, please refer to the `multihead fine-tuning <https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html>`_ guide.

################
LoRA Fine-tuning
################

LoRA (Low-Rank Adaptation) fine-tuning freezes all base model weights and only trains small low-rank adapter matrices injected into each layer. This is particularly useful when you have a small dataset and want to avoid overfitting, or when you want to reduce training memory and compute requirements.

LoRA can be combined with both the naive and multihead replay protocols.

For more information on LoRA fine-tuning, please refer to the :ref:`LoRA fine-tuning <lora_finetuning>` guide.
