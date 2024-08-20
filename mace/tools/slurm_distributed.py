###########################################################################################
# Slurm environment setup for distributed training.
# This code is refactored from rsarm's contribution at:
# https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os

import hostlist


class DistributedEnvironment:
    def __init__(self):

        try:
            self._setup_distr_env()
        except:
            # handle case where not running under Slurm
            self._setup_local_distr_env()
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])

    def _setup_distr_env(self):
        hostname = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0]
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33333")
        os.environ["WORLD_SIZE"] = os.environ.get(
            "SLURM_NTASKS",
            str(
                int(os.environ["SLURM_NTASKS_PER_NODE"])
                * int(os.environ["SLURM_NNODES"])
            ),
        )
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    def _setup_local_distr_env(self):
        hostname = "localhost"
        os.environ["MASTER_ADDR"] = hostname
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33333")
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        # os.environ["WORLD_SIZE"] = os.environ.get(
        #     "SLURM_NTASKS",
        #     str(
        #         int(os.environ["W"])
        #         * 1
        #     ),
        # )
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
