###########################################################################################
# Slurm environment setup for distributed training.
# This code is refactored from rsarm's contribution at:
# https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os

try:
    import hostlist
except ImportError:
    hostlist = None  # Only needed on SLURM systems


class DistributedEnvironment:
    def __init__(self):
        self._setup_distr_env()
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])

    def _setup_distr_env(self):
        if "SLURM_JOB_NODELIST" in os.environ:
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
        else:
            # Assume local manual run with torchrun
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "33333")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("RANK", "0")

    def __repr__(self):
        return (
            f"DistributedEnvironment(master_addr={self.master_addr}, master_port={self.master_port}, "
            f"world_size={self.world_size}, local_rank={self.local_rank}, rank={self.rank})"
        )
