import os
import socket

from mpi4py import MPI


class MPIDistributedEnvironment:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = int(self.comm.Get_rank())
        self.local_rank = int(
            os.environ.get("PMI_LOCAL_RANK", 0)
        )  # self._get_local_rank()
        self.master_addr = self._get_master_addr()
        self._setup_distr_env()

    def _setup_distr_env(self):
        # Set environment variables
        os.environ["MASTER_ADDR"] = str(self.master_addr)
        os.environ["MASTER_PORT"] = "65533"  # You can change this port as needed
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)  # str(self._get_local_rank())

    def _get_master_addr(self):
        with open(os.environ["PBS_NODEFILE"], "r", encoding="utf-8") as file:
            nodes = file.read().splitlines()
        unique_nodes = list(set(nodes))  # Remove duplicates
        unique_nodes.sort()  # Optional: sort to maintain consistency
        return unique_nodes[0]  # Return the first node

    def _get_local_rank(self):
        # This method assumes that all processes on the same node have consecutive ranks
        # This might need adjustment based on your specific environment
        host_name = socket.gethostname()
        all_host_names = self.comm.gather(host_name, root=0)
        if self.rank == 0:
            local_rank_dict = {}
            for name in all_host_names:
                if name not in local_rank_dict:
                    local_rank_dict[name] = 0
                else:
                    local_rank_dict[name] += 1
            local_ranks = [local_rank_dict[name] for name in all_host_names]
        else:
            local_ranks = None
        local_rank = self.comm.scatter(local_ranks, root=0)
        return local_rank
