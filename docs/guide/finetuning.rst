.. _finetuning:

****************************
Fine-tuning Foundation Models
****************************

.. warning::
    Fine-tuning is still experimental and under active development. The API and methods are subject to change.

Fine-tuning is the process of refining a pre-trained model on a new dataset.
This is useful when you want better quantitative performance on a specific task than the available pre-trained models.
Fine-tuning usually leads to significant improvements in performance compared to training a model from scratch.

To finetune one of the mace-mp-0 foundation model, you can use the mace_run_train script with the extra argument `--foundation_model=model_type`. 
For example to finetune the small model on a new dataset, you can use:

.. code-block:: bash

    mace_run_train \
        --name="MACE" \
        --foundation_model="small" \
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

