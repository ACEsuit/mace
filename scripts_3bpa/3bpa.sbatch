#!/bin/bash
#SBATCH --job-name=train-mace            # job name
#SBATCH --partition=gpu_p6s
#SBATCH --account=gax@h100           # account
#SBATCH -C h100                      # target H100 nodes
#SBATCH --nodes=1                    # number of node
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node (here = number of GPUs per node)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 4 for H100 nodes)
#SBATCH --cpus-per-task=12            # number of CPUs per task (here 1/4 of the node)
#SBATCH --time=1:00:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=mace-train-%J
#SBATCH --error=mace-train-%J


srun 3bpa.sh
