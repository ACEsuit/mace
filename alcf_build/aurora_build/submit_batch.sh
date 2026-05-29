#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -A <PROJECT>
#PBS -q workq

cd $PBS_O_WORKDIR

module load frameworks
source .venv/bin/activate

export MACE_CACHE_DIR=/lus/flare/projects/ChemGraph/thang/mace_models
export XDG_CACHE_HOME=$MACE_CACHE_DIR

# 12 tiles = 6 GPUs x 2 tiles per Aurora node
NUM_TILES=12

# CPU binding: 4 cores per tile, split across 2 sockets
# Socket 0 cores (tiles 0-5): 4-7, 8-11, 12-15, 16-19, 20-23, 24-27
# Socket 1 cores (tiles 6-11): 56-59, 60-63, 64-67, 68-71, 72-75, 76-79
CPU_BIND="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"

mpiexec -n ${NUM_TILES} --ppn ${NUM_TILES} \
    --cpu-bind verbose,${CPU_BIND} \
    python alcf_build/aurora_build/run_mace_xpu_batch.py "$@"
