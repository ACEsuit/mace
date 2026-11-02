"""Test-only providers: device tensors, batch search and a per-system skin cache."""

import itertools
import math
import os

import numpy as np
import pytest
import torch
from mace_core.edges import NeighborListCapabilities
from mace_core.neighbors import NeighborList


def require_cuda():
    if not torch.cuda.is_available():
        if "gpu" in os.environ.get("MACE_REQUIRE_CAPS", "").split(","):
            pytest.fail("gpu is required but CUDA is unavailable")
        pytest.skip("requires CUDA")


class DeviceProvider:
    """Small exhaustive search for conformance, not a shipped accelerated list."""

    def capabilities(self):
        return NeighborListCapabilities(
            "device", frozenset({"cpu", "cuda"}), frozenset({"sorted_by_sender"})
        )

    def search(self, positions, cutoff, geometry):
        box = torch.tensor(
            geometry.search_cell, dtype=torch.float64, device=positions.device
        )
        reciprocal = torch.linalg.inv(box)
        fractional = positions @ reciprocal
        ranges = []
        for axis in range(3):
            if geometry.pbc[axis]:
                spread = (
                    float(fractional[:, axis].max() - fractional[:, axis].min())
                    if len(positions)
                    else 0.0
                )
                reach = math.ceil(
                    cutoff * float(torch.linalg.vector_norm(reciprocal[:, axis]))
                    + spread
                )
                ranges.append(range(-reach, reach + 1))
            else:
                ranges.append(range(1))
        images = torch.tensor(
            list(itertools.product(*ranges)), dtype=torch.int64, device=positions.device
        )
        difference = positions[None, :, :] - positions[:, None, :]
        vectors = (
            difference[None, :, :, :]
            + (images.to(torch.float64) @ box)[:, None, None, :]
        )
        image, sender, receiver = torch.where(
            vectors.square().sum(-1) < cutoff * cutoff
        )
        shifts = images[image]
        keep = (sender != receiver) | shifts.ne(0).any(1)
        sender, receiver, shifts = sender[keep], receiver[keep], shifts[keep]
        order = torch.argsort(sender, stable=True)
        cell = torch.tensor(geometry.cell, dtype=torch.float64, device=positions.device)
        return NeighborList(
            torch.stack((sender[order], receiver[order])),
            shifts[order],
            shifts[order].to(torch.float64) @ cell,
            cell,
        )


class BatchProvider(DeviceProvider):
    def __init__(self):
        self.batch_calls = 0

    def search_batch(self, positions, cutoff, geometries):
        self.batch_calls += 1
        return [
            self.search(p, cutoff, g)
            for p, g in zip(positions, geometries, strict=True)
        ]


class StatefulProvider(DeviceProvider):
    def __init__(self):
        self.reference = None
        self.rebuilds = 0

    def capabilities(self):
        return NeighborListCapabilities(
            "stateful",
            frozenset({"cpu", "cuda"}),
            frozenset({"sorted_by_sender"}),
            stateful=True,
        )

    def search(self, positions, cutoff, geometry):
        changed = self.reference is None or self.reference.shape != positions.shape
        changed = changed or cutoff != getattr(self, "cutoff", None)
        changed = changed or not np.array_equal(
            geometry.cell, getattr(self, "cell", np.empty((0, 3)))
        )
        changed = changed or geometry.pbc != getattr(self, "pbc", None)
        if changed or (
            len(positions)
            and float(torch.linalg.vector_norm(positions - self.reference, dim=1).max())
            > 0.1
        ):
            self.reference = positions.clone()
            self.cutoff, self.cell, self.pbc = (
                cutoff,
                geometry.cell.copy(),
                geometry.pbc,
            )
            self.candidates = super().search(positions, cutoff + 0.2, geometry)
            self.rebuilds += 1
        candidate = self.candidates
        sender, receiver = candidate.edge_index
        vectors = positions[receiver] - positions[sender] + candidate.shifts
        keep = vectors.square().sum(1) < cutoff * cutoff
        return NeighborList(
            candidate.edge_index[:, keep],
            candidate.unit_shifts[keep],
            candidate.shifts[keep],
            candidate.cell,
        )


def register(path, monkeypatch, name, target, group="mace.neighbor_backends.torch"):
    info = path / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(f"[{group}]\n{name} = {target}\n")
    monkeypatch.syspath_prepend(str(path))
