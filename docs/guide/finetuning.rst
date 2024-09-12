.. _finetuning:

****************************
Fine-tuning Foundation Models
****************************

.. warning::
    Fine-tuning is still experimental and under active development. The API and methods are subject to change.

Fine-tuning is the process of refining a pre-trained model on a new dataset.
This is useful when you want better quantitative performance on a specific task than the available pre-trained models.
Fine-tuning usually leads to significant improvements in performance compared to training a model from scratch.
We have two types of fine-tuning protocols:

 - The **naive** fine-tuning protocol, where the model is trained on the new dataset just by restarting from the foundation model weights.
 - The **multihead replay** fine-tuning protocol, where the model is trained on the new dataset while replaying a part of the original foundational model training data. Only materials project models are currently supported.

The multihead replay finetuning prevent catastrophic forgetting that occurs sometimes during the naive fine-tuning. 
It usually leads to a more robust and stable model. It is **recommended** to use this protocol to fine-tune any materials project foundation model.

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
        --energy_weight=1.0 \
        --forces_weight=1.0 \
        --E0s="average" \
        --lr=0.01 \
        --scaling="rms_forces_scaling" \
        --batch_size=2 \
        --max_num_epochs=6 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --default_dtype="float64" \
        --device=cuda \
        --seed=3 

Other options are "medium" and "large", or the path to a foundation model. 
If you want to finetune another model, the model will be loaded from the path provided `--foundation_model=$path_model`, but you will need to provide the full set of hyperparameters (hidden irreps, r_max, etc.) matching the model.

############################
Multihead Replay Fine-tuning
############################

The multihead replay fine-tuning protocol prevents catastrophic forgetting that occurs sometimes during the naive fine-tuning.
It usually leads to a more robust and stable model. It is the **recommended** way to fine-tune any materials project foundation model.

For this fine-tuning, it is important to use one of the MACE-mp-0b models that you can download here: https://github.com/ACEsuit/mace-mp/releases/tag/mace_mp_0b.
It is also very important that you compute your own E0s and not use "average" option. When computing your E0s, please use spin polarized calculations.
If you are using MP compatible DFT, you can use the option --E0s="foundation" to use the same E0s as the foundation model.

To fine-tune a small mp0b model, you can use:

.. code-block:: bash

    mace_run_train \
        --name="MACE" \
        --foundation_model="mace_agnesi_small.model" \
        --multiheads_finetuning=True \
        --train_file="train.xyz" \
        --valid_fraction=0.05 \
        --test_file="test.xyz" \
        --energy_weight=1.0 \
        --forces_weight=1.0 \
        --E0s="{"1": 130.0 }" \
        --lr=0.01 \
        --scaling="rms_forces_scaling" \
        --batch_size=2 \
        --max_num_epochs=6 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --default_dtype="float64" \
        --device=cuda \
        --seed=3 