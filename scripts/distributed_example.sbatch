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

srun python mace/scripts/run_train.py \
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
