.. _multigpu:

====================
Multi-GPUs Training
====================

Multi-nodes training
-------------------

For multi-GPU training, use the `--distributed` flag.
This will use PyTorch's DistributedDataParallel module to train the model on multiple GPUs.
Combine with on-line data loading for large datasets (see .. _multipreprocessing:).
Here is an example command to train a model on 4 GPUs on Slurm:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --partition=gpu
    #SBATCH --job-name=train
    #SBATCH --output=train.out
    #SBATCH --nodes=2
    #SBATCH --ntasks=20
    #SBATCH --ntasks-per-node=10
    #SBATCH --gpus-per-node=10
    #SBATCH --cpus-per-task=8
    #SBATCH --exclusive
    #SBATCH --time=1:00:00

    srun python <mace_repo_dir>/mace/cli/run_train.py \
        --name='model' \
        --model='MACE' \
        --num_interactions=2 \
        --num_channels=128 \
        --max_L=2 \
        --correlation=3 \
        --E0s='average' \
        --r_max=5.0 \
        --train_file='./h5_data/train.h5' \
        --valid_file='./h5_data/valid.h5' \
        --statistics_file='./h5_data/statistics.json' \
        --num_workers=8 \
        --batch_size=20 \
        --valid_batch_size=80 \
        --max_num_epochs=100 \
        --loss='weighted' \
        --error_table='PerAtomRMSE' \
        --default_dtype='float32' \
        --device='cuda' \
        --distributed \
        --seed=2222 \

This script will train the model on 20 GPUs (--ntasks=20) on 2 nodes (--nodes=2) with 10 GPUs per node (--ntasks-per-node=10).

For Slurm users, the necessary environment variables should be set automatically in the file `mace/tools/slurm_distributed.py`.
For other systems, you may need to set the environment variables manually by modifying the file.


Single-node multi-GPU training
------------------------------

For training on a single node with multiple GPUs, you can use the following command:

.. code-block:: bash

    torchrun --standalone --nnodes=1 --nproc_per_node=4 <mace_repo_dir>/mace/cli/run_train.py \
        --name='model' \
        --model='MACE' \
        --num_interactions=2 \
        --num_channels=128 \
        --max_L=2 \
        --correlation=3 \
        --E0s='average' \
        --r_max=5.0 \
        --train_file='./h5_data/train.h5' \
        --valid_file='./h5_data/valid.h5' \
        --statistics_file='./h5_data/statistics.json' \
        --num_workers=8 \
        --batch_size=20 \
        --valid_batch_size=80 \
        --max_num_epochs=100 \
        --loss='weighted' \
        --error_table='PerAtomRMSE' \
        --default_dtype='float32' \
        --device='cuda' \
        --distributed \
        --seed=2222 \

This script will train the model on 4 GPUs (--nproc_per_node=4) on a single node (--nnodes=1).