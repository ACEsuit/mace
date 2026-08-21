import os

import torch


def xpu_device_index(local_rank: int) -> int:
    """
    Map a local rank onto a visible XPU device index.

    Under ZE_AFFINITY_MASK each rank sees a single tile, so local_rank can
    exceed the number of visible devices; torch then rejects the index with
    "value cannot be converted to type int without overflow".
    """
    try:
        n_visible = torch.xpu.device_count()
    except (AttributeError, RuntimeError):
        return 0
    return local_rank if local_rank < n_visible else 0


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
        # OpenMPI exports OMPI_*; Intel MPI / PALS export PMI_*/PALS_*.
        if "OMPI_COMM_WORLD_RANK" in os.environ:
            # OpenMPI & Intel-MPI export these:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

            # local-rank isn’t standardised; compute it from local node-size
            local_size = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))
            local_rank = rank % local_size
        else:
            rank = int(os.environ.get("PMI_RANK", os.environ.get("PALS_RANKID", 0)))
            # PMI_SIZE = global world size; PALS_NTASKS = same on PALS;
            # PALS_LOCAL_SIZE is per-NODE size, NOT a valid world_size.
            world_size = int(
                os.environ.get(
                    "PMI_SIZE",
                    os.environ.get("PALS_NTASKS", os.environ.get("WORLD_SIZE", 1)),
                )
            )
            local_rank = int(
                os.environ.get(
                    "PALS_LOCAL_RANKID", os.environ.get("MPI_LOCALRANKID", rank)
                )
            )

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
                backend="xccl",
                init_method="env://",
            )
        else:
            # CPU (tests, debugging): gloo. Previously no process group was
            # created here at all, so --distributed on CPU could never work.
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="env://",
            )
    return rank, local_rank, world_size
