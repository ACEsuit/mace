"""Periodic CUDA cell lists shared with the optional native deployment runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from mace_core.config.base import ConfigSection
from mace_core.edges import NeighborListCapabilities
from mace_core.neighbors import NeighborBackendNotAvailableError, NeighborList


class NativeCudaSettings(ConfigSection):
    library: Path


def _load(library: Path) -> None:
    if hasattr(torch.ops.mace_neighbors, "periodic"):
        return
    try:
        torch.ops.load_library(str(library.resolve(strict=True)))
    except (OSError, RuntimeError) as error:
        raise NeighborBackendNotAvailableError(
            f"cannot load native CUDA neighbour library {library}: {error}"
        ) from error

    @torch.library.register_fake("mace_neighbors::periodic")
    def fake(positions, cell, cutoff, capacity, padding_shift):
        edges = capacity or torch.library.get_ctx().new_dynamic_size()
        return (
            positions.new_empty((2, edges), dtype=torch.int64),
            positions.new_empty((edges, 3), dtype=torch.int64),
            positions.new_empty((), dtype=torch.int64),
        )


class NativeCudaBackend:
    settings_section = NativeCudaSettings

    def __init__(self, settings: NativeCudaSettings):
        _load(settings.library)

    def capabilities(self) -> NeighborListCapabilities:
        return NeighborListCapabilities("native_cuda", devices=frozenset({"cuda"}))

    def search(self, positions, cutoff, geometry):
        if not all(geometry.pbc):
            raise ValueError("native_cuda neighbours require full periodicity")
        cell = torch.tensor(geometry.cell, dtype=torch.float64, device=positions.device)
        if len(positions):
            index, shifts, _ = torch.ops.mace_neighbors.periodic(
                positions.detach().contiguous(),
                geometry.cell.ravel().tolist(),
                cutoff,
                0,
                0,
            )
        else:
            index = torch.empty((2, 0), dtype=torch.int64, device=positions.device)
            shifts = torch.empty((0, 3), dtype=torch.int64, device=positions.device)
        return NeighborList(index, shifts, shifts.to(torch.float64) @ cell, cell)
