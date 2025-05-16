.. _multihead_finetuning:

*******************************
MACE Multihead Finetuning Documentation
*******************************

Introduction
============

Multihead finetuning is a technique in MACE that allows you to simultaneously finetune a foundation model on both your target dataset and a “replay” dataset from the foundation model. This approach helps prevent catastrophic forgetting and maintains the model’s generalization capabilities while adapting to your specific use case.

This document explains how to perform multihead finetuning in MACE using three different approaches for selecting the replay dataset, and describes the available replay datasets.

Replay Dataset Options
======================

Before starting the finetuning process, you need to obtain a replay dataset. These are available from:

https://github.com/ACEsuit/mace-foundations/releases

There are two types of replay datasets available:

Mode 1 Replay: Original training data with true DFT labels
----------------------------------------------------------

- Contains actual DFT calculations used to train the foundation model
- Provides genuine ground truth energies and forces

Mode 2 Replay: Diverse configurations evaluated with the foundation model
--------------------------------------------------------------------------

- Contains configurations from various materials evaluated using the foundation model itself
- Initial metrics on forces and energy will show zeros (since predictions match labels exactly)
- Focuses on preserving the model’s existing behavior rather than ground truth accuracy

Most release contains corresponding dataset that can be used as the starting point for finetuning.

Method 1: Preprocessing via CLI
===============================

The first approach uses the ``fine_tuning_select.py`` CLI tool to prepare your replay dataset before training.

Step 1: Select Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``fine_tuning_select.py`` script to select configurations from the replay dataset based on your target dataset::

    python -m mace.cli.fine_tuning_select \
      --configs_pt path/to/replay_dataset.xyz \
      --configs_ft path/to/your_dataset.xyz \
      --num_samples 10000 \
      --subselect fps \
      --model path/to/foundation_model.model \
      --output selected_configs.xyz \
      --filtering_type combinations \
      --head_pt pt_head \
      --head_ft target_head \
      --weight_pt 1.0 \
      --weight_ft 10.0

Key parameters:

- ``--configs_pt``: Path to the replay dataset  
- ``--configs_ft``: Path to your target dataset  
- ``--num_samples``: Number of configurations to select from the replay dataset  
- ``--subselect``: Method for subselection (``fps`` for Farthest Point Sampling or ``random``)  
- ``--filtering_type``: How to filter configurations based on elements:

  - ``combinations``: Keep configurations with combinations of elements in your target dataset  
  - ``exclusive``: Keep configurations containing only elements in your target dataset  
  - ``inclusive``: Keep configurations containing all elements in your target dataset  
  - ``none``: No filtering  

- ``--atomic_numbers``: Optionally specify specific atomic numbers to filter by  

Step 2: Train with the Combined Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After creating the combined dataset, use it for training::

    python -m mace.cli.run_train \
      --name mymodel_finetuned \
      --pt_train_file selected_configs.xyz \
      --train_file path/to/your_dataset.xyz \
      --valid_fraction 0.05 \
      --foundation_model path/to/foundation_model.model \
      --energy_weight 1.0 \
      --forces_weight 100.0 \
      --swa \
      --swa_energy_weight 10.0 \
      --swa_forces_weight 100.0


Method 2: MP Shortcut for Compatible Models
=================================

MACE provides automatic pre-trained data for foundation models based on the Materials Project (MP) database or database with compatible DFT. 
You can use the special value ``mp`` for the ``--pt_train_file`` parameter to automatically download and use an appropriate replay dataset::

    python -m mace.cli.run_train \
      --train_file path/to/your_dataset.xyz \
      --foundation_model medium \
      --pt_train_file mp \
      --atomic_numbers "[1, 6, 7, 8]" \
      --multiheads_finetuning True

Compatible Foundation Models
----------------------------

- **MACE-MP-0** family (small, medium, large):  
  ``--foundation_model small``  (or ``medium``, ``large``)

- **MACE-MP-0b**:  
  ``--foundation_model path/to/mace-mp-0b.model``

- **MACE-MP-0b2**:  
  ``--foundation_model path/to/mace-mp-0b2.model``

- **MACE-MP-0b3**:  
  ``--foundation_model path/to/mace-mp-0b3.model``

- **MACE-MPA-0**:  
  ``--foundation_model path/to/mace-mpa-0.model``

Method 3: Direct Multihead Finetuning with run_train
====================================================

The second approach integrates dataset selection directly into the training process using ``run_train.py``.

.. code-block:: bash

    python -m mace.cli.run_train \
      --name mymodel_finetuned \
      --train_file path/to/your_dataset.xyz \
      --foundation_model path/to/foundation_model.model \
      --pt_train_file path/to/replay_dataset.xyz \
      --num_samples_pt 10000 \
      --filter_type_pt combinations \
      --subselect_pt fps \
      --weight_pt 1.0 \
      --atomic_numbers "[1, 6, 7, 8]" \
      --multiheads_finetuning True \
      --force_mh_ft_lr False

Key parameters:

- ``--train_file``: Your target dataset  
- ``--foundation_model``: Path to the foundation model  
- ``--pt_train_file``: Path to the replay dataset (or use “mp” for Materials Project data)  
- ``--num_samples_pt``: Number of samples to use from the replay dataset  
- ``--filter_type_pt``: Filtering strategy for configurations  
- ``--subselect_pt``: Method for subselection (``fps`` or ``random``)  
- ``--weight_pt``: Weight for the pretraining head loss  
- ``--atomic_numbers``: Critical parameter specifying which elements to keep in the replay dataset  
- ``--multiheads_finetuning``: Enable multihead finetuning  
- ``--force_mh_ft_lr``: By default, multihead finetuning uses a lower learning rate (0.0001) and enables EMA. Set to True to override this behavior.

Important Note on Atomic Numbers
--------------------------------

When using Method 2, you must provide ``--atomic_numbers`` to specify which elements to include in the replay dataset. This should include all elements in your target dataset. For example::

    python -m mace.cli.run_train \
      --atomic_numbers "[1, 6, 7, 8]"  # For a dataset with H, C, N, O

This parameter determines which elements from the replay dataset to keep during finetuning.

Advanced Configuration: Using Heads Dictionary
---------------------------------------------

For more complex scenarios, you can define a heads dictionary to customize each head::

    python -m mace.cli.run_train \
      --name mymodel_finetuned \
      --foundation_model path/to/foundation_model.model \
      --multiheads_finetuning True \
      --heads "{'target_head': {'train_file': 'path/to/your_dataset.xyz', 'E0s': 'path/to/e0s.json'}, 'pt_head': {'train_file': 'path/to/replay_dataset.xyz', 'E0s': 'foundation'}}" \
      --atomic_numbers "[1, 6, 7, 8]" \
      --energy_weight 1.0 \
      --forces_weight 100.0

Monitoring and Evaluating
=========================

During training, MACE will report metrics for both your target dataset and the replay dataset. It is normal to see different performance between the heads – your model is balancing learning on your specific data while preserving knowledge from the foundation model’s training.

To extract the final model head after training, you can use the mace_select_head CLI tool, to get a single head model for deployment.

If you use the two head model as an ASE calculator, you can specify to use the fine-tuned head (called 'default' by default) or the pretraining head (called 'pt_head' by default):

.. code-block:: python

    calc = MACECalculator(
        model_paths="path/to/finetuned_model.model",
        device="cuda",  # or "cpu"
        head="default"  # Specify your target head name here
    )

Tips for Successful Multihead Finetuning
========================================

- **Check the initial performance**: Monitor the initial performance of the models when finetuning. If you see that the initial error on your dataset is very high (> 0.4 eV/atom), it is likely that you need to adjust your E0s. You can use the ``--E0s`` parameter to provide a custom E0s file.
- **Use the right E0s**: It is also very important that you compute your own E0s and not use "average" option. When computing your E0s, please use spin polarized calculations. If you are using MP compatible DFT, you can use the option --E0s="foundation" to use the same E0s as the foundation model.
- **Balance head weights**: Adjust ``--weight_pt`` and ``--weight_ft`` to control the importance of each dataset. Higher values for your target dataset will focus more on performance for your specific application.  
- **Choose the right replay mode**: If you have access to true DFT labels, compare carefully mode 1 and mode 2 of replay datasets.
- **Element selection**: Carefully choose which elements to include using ``--atomic_numbers``. Including unnecessary elements increases computational cost.  
- **Dataset size ratio**: It usually gives best performance to use as many replay sample as you can. Use ``--num_samples_pt`` to control this (30000 is a good value). As your input data will be repeatedly sampled, you can use a smaller number of epochs.
- **Number of epochs for training**: The number of epochs for convergence is between 10 and 30 epochs. Larger number of sampler might reduce the number of epochs to convergence.
- **EMA and learning rate**: Use ``--ema`` and ``--ema_decay`` to stabilize training. To change the default values for EMA and learning rate, use the --force_mh_ft_lr parameter. The default learning rate for multihead finetuning is 0.0001, which is lower than the default learning rate for single head finetuning (0.001). This helps to stabilize the training process and prevent overfitting.
- **Weight Decay**: Turning down the weight decay might help boost performance, you can experiment with setting values from 5e-7 to 0.0, going down by factors of 10.
