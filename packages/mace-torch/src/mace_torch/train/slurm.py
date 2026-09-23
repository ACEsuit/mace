"""Where a process sits in a SLURM job, read from the variables SLURM sets.

The process group is initialized from ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT``, which is what ``torchrun``
exports. Under ``srun`` none of those exist, so they are derived here from
SLURM's own: the rank from ``SLURM_PROCID``, the world size from
``SLURM_NTASKS``, the local rank from ``SLURM_LOCALID``, and the rendezvous
address from the first host of ``SLURM_JOB_NODELIST``.

The node list is written in SLURM's compressed form, ``node[01-03,07]``, and is
expanded here rather than through an optional package: a missing optional
dependency would surface on the cluster, in the first job, as an import error
about something the user never asked for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["SlurmEnvironment", "expand_nodelist", "slurm_environment"]

#: The rendezvous port when the job does not set one, as legacy uses.
DEFAULT_PORT = "33333"


@dataclass(frozen=True)
class SlurmEnvironment:
    """One process's place in the job.

    Attributes:
        rank: Its index among all processes.
        world_size: How many processes there are.
        local_rank: Its index among the processes on its node.
        master_addr: The host rank zero runs on.
        master_port: The port the rendezvous listens on.
    """

    rank: int
    world_size: int
    local_rank: int
    master_addr: str
    master_port: str

    def exported(self) -> dict[str, str]:
        """The variables the process group reads, as ``torchrun`` names them."""
        return {
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": self.master_port,
        }


def expand_nodelist(nodelist: str) -> list[str]:
    """SLURM's compressed host list, written out.

    ``a[1-3],b07,c[08-09,11]`` is ``a1 a2 a3 b07 c08 c09 c11``: ranges keep the
    zero padding of their first bound, and brackets group comma-separated
    ranges.

    Raises:
        ValueError: On a list whose brackets do not close.
    """
    groups: list[str] = []
    depth, current = 0, ""
    for character in nodelist:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"the node list {nodelist!r} closes a bracket it never opened"
                )
        if character == "," and depth == 0:
            groups.append(current)
            current = ""
        else:
            current += character
    if depth != 0:
        raise ValueError(f"the node list {nodelist!r} has an unclosed bracket")
    groups.append(current)

    hosts: list[str] = []
    for group in filter(None, groups):
        match = re.fullmatch(r"([^\[]*)\[([^\]]*)\](.*)", group)
        if match is None:
            hosts.append(group)
            continue
        prefix, ranges, suffix = match.groups()
        for part in ranges.split(","):
            if "-" in part:
                first, last = part.split("-", 1)
                width = len(first)
                for number in range(int(first), int(last) + 1):
                    hosts.append(f"{prefix}{number:0{width}d}{suffix}")
            else:
                hosts.append(f"{prefix}{part}{suffix}")
    return hosts


def slurm_environment(environ: Mapping[str, str]) -> SlurmEnvironment:
    """Read a process's place in the job from SLURM's variables.

    Args:
        environ: The process environment, passed in so it can be checked
            without being modified.

    Raises:
        KeyError: Naming the SLURM variable that is missing, since a job not
            launched with ``srun`` has none of them.
    """
    missing = [
        name
        for name in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_JOB_NODELIST")
        if name not in environ
    ]
    if missing:
        raise KeyError(
            f"the SLURM launcher needs {missing}, which srun sets. The process "
            f"was not started by srun, or not inside an allocation."
        )
    if "SLURM_NTASKS" in environ:
        world_size = int(environ["SLURM_NTASKS"])
    else:
        world_size = int(environ["SLURM_NTASKS_PER_NODE"]) * int(
            environ["SLURM_NNODES"]
        )
    return SlurmEnvironment(
        rank=int(environ["SLURM_PROCID"]),
        world_size=world_size,
        local_rank=int(environ["SLURM_LOCALID"]),
        master_addr=expand_nodelist(environ["SLURM_JOB_NODELIST"])[0],
        master_port=environ.get("MASTER_PORT", DEFAULT_PORT),
    )
