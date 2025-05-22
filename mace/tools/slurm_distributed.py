###########################################################################################
# Slurm environment setup for distributed training.
# This code is refactored from rsarm's contribution at:
# https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os
import socket

import hostlist

# class DistributedEnvironment:
#     def __init__(self):
#         self._setup_distr_env()
#         self.master_addr = os.environ["MASTER_ADDR"]
#         self.master_port = os.environ["MASTER_PORT"]
#         self.world_size = int(os.environ["WORLD_SIZE"])
#         self.local_rank = int(os.environ["LOCAL_RANK"])
#         self.rank = int(os.environ["RANK"])

#     def _setup_distr_env(self):
#         hostname = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0]
#         os.environ["MASTER_ADDR"] = hostname
#         os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "33333")
#         os.environ["WORLD_SIZE"] = os.environ.get(
#             "SLURM_NTASKS",
#             str(
#                 int(os.environ["SLURM_NTASKS_PER_NODE"])
#                 * int(os.environ["SLURM_NNODES"])
#             ),
#         )
#         os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
#         os.environ["RANK"] = os.environ["SLURM_PROCID"]

class DistributedEnvironment:
    def __init__(self):
        self._setup_distr_env()
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])

    def _setup_distr_env(self):
        # Set MASTER_ADDR from the first host in PE_HOSTFILE
        pe_hostfile = os.environ.get("PE_HOSTFILE", "")
        if os.path.exists(pe_hostfile):
            with open(pe_hostfile, "r") as f:
                first_host = f.readline().split()[0]
                os.environ["MASTER_ADDR"] = first_host
        else:
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")

        # Default port if not set
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

        # Derive world size (number of slots)
        if "WORLD_SIZE" not in os.environ:
            ns = os.environ.get("NSLOTS", "1")
            os.environ["WORLD_SIZE"] = ns

        # Use MPI variables if present, otherwise fallback
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0"))
        os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0"))
