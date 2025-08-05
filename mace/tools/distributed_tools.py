import os

import torch


def init_distributed(args):
    """
    Returns (rank, local_rank, world_size) and initialises the process-group.
    Works for: slurm | torchrun | mpi | none
    """
    if not args.distributed:
        return 0, 0, 1  # single-GPU / debug run

    # ------------------------------------------------------------------ slurm
    if args.launcher == "slurm":
        from mace.tools.slurm_distributed import DistributedEnvironment

        env = DistributedEnvironment()
        rank, local_rank, world_size = env.rank, env.local_rank, env.world_size

    # ---------------------------------------------------------------- torchrun
    elif args.launcher == "torchrun":
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # -------------------------------------------------------------------- mpi
    elif args.launcher == "mpi":
        # OpenMPI & Intel-MPI export these:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

        # local-rank isn’t standardised; compute it from local node-size
        local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))
        local_rank = rank % local_size

        # tell PyTorch where the rendez-vous server is
        os.environ.setdefault("MASTER_ADDR", os.environ["MASTER_ADDR"])
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "33333"))
        # torchrun style vars so later code keeps working
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

    else:  # "none"
        return 0, 0, 1

    if not torch.distributed.is_initialized():
        if args.device == "cuda":
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        elif args.device == "xpu":
            torch.distributed.init_process_group(
                backend="ccl",
                init_method="env://",
            )
    return rank, local_rank, world_size
