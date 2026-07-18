"""Large-N GPU crossover study for exact FFF and padded multipole PME."""

import gc
import math
import os
import statistics
from dataclasses import dataclass
from typing import Callable, Optional

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_pme_reciprocal_space,
)
from nvalchemiops.torch.math.gto import NormMode, inv_cl

SOURCE_SIGMA = 1.0
RECEIVER_SIGMA = 1.0
MESH_SPACING = 0.5
LATTICE_SPACING = 2.0
PADDING = 12.0
PME_ALPHA = 1.0e6
EXACT_MAX_ATOMS = 4096


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    peak_gib: float


def _mesh_size(box_length: float) -> int:
    requested = math.ceil(box_length / MESH_SPACING)
    return max(16, 8 * math.ceil(requested / 8))


def _build_system(num_atoms: int, device: torch.device, dtype: torch.dtype):
    side = math.ceil(num_atoms ** (1.0 / 3.0))
    axis = torch.arange(side, device=device, dtype=dtype) * LATTICE_SPACING
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    positions = xyz.reshape(-1, 3)[:num_atoms].contiguous()

    generator = torch.Generator(device=device).manual_seed(17 + num_atoms)
    positions = positions + 0.08 * torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    )
    positions = positions - positions.amin(dim=0) + PADDING
    extent = positions.amax(dim=0) - positions.amin(dim=0)
    box_length = float(extent.max().item() + 2.0 * PADDING)
    cell = torch.eye(3, device=device, dtype=dtype) * box_length

    charges = torch.randn(num_atoms, device=device, dtype=dtype, generator=generator)
    charges = charges - charges.mean()
    dipoles = 0.1 * torch.randn(
        (num_atoms, 3), device=device, dtype=dtype, generator=generator
    )
    moments = torch.cat([charges[:, None], dipoles[:, [1, 2, 0]]], dim=-1).contiguous()
    batch = torch.zeros(num_atoms, device=device, dtype=torch.long)
    pbc = torch.zeros((1, 3), device=device, dtype=torch.bool)
    volume = torch.linalg.det(cell).abs().reshape(1)
    return positions, moments, cell, batch, pbc, volume, box_length


def _models(device: torch.device, dtype: torch.dtype):
    features = GTOElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=SOURCE_SIGMA,
        feature_max_l=1,
        feature_smearing_widths=[RECEIVER_SIGMA],
        include_self_interaction=False,
        kspace_cutoff=4.0,
        integral_normalization="receiver",
    ).to(device=device, dtype=dtype)
    energy = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=SOURCE_SIGMA,
        kspace_cutoff=4.0,
        include_self_interaction=False,
    ).to(device=device, dtype=dtype)
    return features, energy


def _receiver_scale(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scales = [
        inv_cl(RECEIVER_SIGMA, ell, NormMode.RECEIVER)
        / inv_cl(RECEIVER_SIGMA, ell, NormMode.MULTIPOLES)
        for ell in range(2)
    ]
    return torch.tensor(
        [scales[0], scales[1], scales[1], scales[1]],
        device=device,
        dtype=dtype,
    )


def _exact_features(descriptor, positions, moments, batch):
    pos = positions.detach().requires_grad_(True)
    src = moments.detach().requires_grad_(True)
    return descriptor.realspace_features(
        source_feats=src.unsqueeze(-2), node_positions=pos, batch=batch
    )[0]


def _exact_energy(energy_fn, positions, moments, batch):
    pos = positions.detach().requires_grad_(True)
    src = moments.detach().requires_grad_(True)
    return energy_fn.realspace_energy(source_feats=src, positions=pos, batch=batch)


def _pme_node_energy(positions, moments, cell, mesh_size):
    pos = positions.detach().requires_grad_(True)
    src = moments.detach().requires_grad_(True)
    node_energy = multipole_pme_reciprocal_space(
        pos,
        src,
        cell,
        sigma=SOURCE_SIGMA,
        alpha=PME_ALPHA,
        mesh_dimensions=(mesh_size, mesh_size, mesh_size),
    )
    return pos, src, node_energy


def _pme_features(
    descriptor,
    positions,
    moments,
    cell,
    batch,
    pbc,
    volume,
    mesh_size,
    receiver_scale,
):
    pos, src, node_energy = _pme_node_energy(positions, moments, cell, mesh_size)
    features = torch.autograd.grad(node_energy.sum(), src, create_graph=True)[0]
    features = features * receiver_scale
    return features + descriptor.non_periodic_correction_terms(
        source_feats=src,
        node_positions=pos,
        batch=batch,
        volumes=volume,
        pbc=pbc,
    )


def _pme_energy(energy_fn, positions, moments, cell, batch, volume, mesh_size):
    pos, src, node_energy = _pme_node_energy(positions, moments, cell, mesh_size)
    correction = energy_fn.monopole_dipole_correction(src, pos, volume, batch)
    return node_energy.sum().reshape(1) + correction


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _time_cuda(fn: Callable, warmup: int, repeats: int) -> Timing:
    _clear_cuda()
    for _ in range(warmup):
        output = fn()
        torch.cuda.synchronize()
        del output
    _clear_cuda()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
        del output
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    return Timing(
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        peak_gib=peak_gib,
    )


def _try_timing(name: str, fn: Callable, repeats: int) -> Optional[Timing]:
    try:
        timing = _time_cuda(fn, warmup=2, repeats=repeats)
        print(
            "FFF_LARGE_TIMING "
            f"operation={name} median_ms={timing.median_ms:.3f} "
            f"min_ms={timing.min_ms:.3f} max_ms={timing.max_ms:.3f} "
            f"peak_GiB={timing.peak_gib:.3f}",
            flush=True,
        )
        return timing
    except torch.cuda.OutOfMemoryError as exc:
        print(f"FFF_LARGE_TIMING operation={name} OOM={exc}", flush=True)
        _clear_cuda()
        return None


def _run_case(num_atoms: int, device: torch.device, dtype: torch.dtype) -> None:
    _clear_cuda()
    positions, moments, cell, batch, pbc, volume, box_length = _build_system(
        num_atoms, device, dtype
    )
    descriptor, energy_fn = _models(device, dtype)
    receiver_scale = _receiver_scale(device, dtype)
    mesh_size = _mesh_size(box_length)
    repeats = 10 if num_atoms <= 2048 else 5
    print(
        "FFF_LARGE_CASE "
        f"N={num_atoms} box_A={box_length:.2f} mesh={mesh_size}^3 "
        f"repeats={repeats}",
        flush=True,
    )

    exact_feature_timing = None
    exact_energy_timing = None
    if num_atoms <= EXACT_MAX_ATOMS:
        exact_feature_timing = _try_timing(
            "exact_feature",
            lambda: _exact_features(descriptor, positions, moments, batch),
            repeats,
        )
        exact_energy_timing = _try_timing(
            "exact_energy",
            lambda: _exact_energy(energy_fn, positions, moments, batch),
            repeats,
        )

    pme_feature_timing = _try_timing(
        "pme_feature",
        lambda: _pme_features(
            descriptor,
            positions,
            moments,
            cell,
            batch,
            pbc,
            volume,
            mesh_size,
            receiver_scale,
        ),
        repeats,
    )
    pme_energy_timing = _try_timing(
        "pme_energy",
        lambda: _pme_energy(
            energy_fn, positions, moments, cell, batch, volume, mesh_size
        ),
        repeats,
    )

    if all(
        timing is not None
        for timing in (
            exact_feature_timing,
            exact_energy_timing,
            pme_feature_timing,
            pme_energy_timing,
        )
    ):
        exact_r1 = 2 * exact_feature_timing.median_ms + exact_energy_timing.median_ms
        pme_r1 = 2 * pme_feature_timing.median_ms + pme_energy_timing.median_ms
        print(
            "FFF_LARGE_SUMMARY "
            f"N={num_atoms} exact_r1_ms={exact_r1:.3f} "
            f"pme_r1_ms={pme_r1:.3f} speedup={exact_r1 / pme_r1:.3f}",
            flush=True,
        )
    _clear_cuda()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    print(
        "FFF_LARGE_DEVICE "
        f"name={torch.cuda.get_device_name(0)} dtype={dtype} "
        f"padding_A={PADDING} mesh_spacing_A={MESH_SPACING}",
        flush=True,
    )
    for num_atoms in (1024, 2048, 3072, 4096, 8192, 16384):
        _run_case(num_atoms, device, dtype)


if __name__ == "__main__":
    main()
