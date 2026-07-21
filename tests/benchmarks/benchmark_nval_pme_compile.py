"""Measure fixed-cell caching and torch.compile overhead for multipole PME."""

import gc
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    _resolve_cell_inv_t,
    _resolve_pme_k_squared,
    _resolve_pme_moduli,
    multipole_pme_reciprocal_space,
)
from nvalchemiops.torch.math.gto import NormMode, inv_cl

SOURCE_SIGMA = 1.0
RECEIVER_SIGMA = 1.0
MESH_SPACING = 0.5
LATTICE_SPACING = 2.0
PADDING = 12.0
PME_ALPHA = 1.0e6
SPLINE_ORDER = 4


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    peak_gib: float


@dataclass
class PmeCache:
    cell_inv_t: torch.Tensor
    volume: torch.Tensor
    moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    k_squared: torch.Tensor


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
    return positions, moments, cell, batch, pbc, box_length


def _models(device: torch.device, dtype: torch.dtype):
    descriptor = GTOElectrostaticFeatures(
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
    scales = [
        inv_cl(RECEIVER_SIGMA, ell, NormMode.RECEIVER)
        / inv_cl(RECEIVER_SIGMA, ell, NormMode.MULTIPOLES)
        for ell in range(2)
    ]
    receiver_scale = torch.tensor(
        [scales[0], scales[1], scales[1], scales[1]],
        device=device,
        dtype=dtype,
    )
    return descriptor, energy, receiver_scale


def _prepare_cache(
    cell: torch.Tensor, mesh_dimensions: tuple[int, int, int], dtype: torch.dtype
) -> PmeCache:
    return PmeCache(
        cell_inv_t=_resolve_cell_inv_t(cell, None),
        volume=torch.abs(torch.det(cell.to(torch.float64))),
        moduli=_resolve_pme_moduli(
            mesh_dimensions, SPLINE_ORDER, dtype, cell.device, None
        ),
        k_squared=_resolve_pme_k_squared(cell, mesh_dimensions, dtype, None),
    )


def _make_node_energy(
    cell: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    cache: Optional[PmeCache],
) -> Callable:
    if cache is None:

        def node_energy(pos: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            return multipole_pme_reciprocal_space(
                pos,
                src,
                cell,
                sigma=SOURCE_SIGMA,
                alpha=PME_ALPHA,
                mesh_dimensions=mesh_dimensions,
                spline_order=SPLINE_ORDER,
            )

    else:
        cell_inv_t = cache.cell_inv_t
        volume = cache.volume
        moduli = cache.moduli
        k_squared = cache.k_squared

        def node_energy(pos: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            return multipole_pme_reciprocal_space(
                pos,
                src,
                cell,
                sigma=SOURCE_SIGMA,
                alpha=PME_ALPHA,
                mesh_dimensions=mesh_dimensions,
                spline_order=SPLINE_ORDER,
                cell_inv_t=cell_inv_t,
                volume=volume,
                moduli=moduli,
                k_squared=k_squared,
            )

    return node_energy


def _feature_fn(
    node_energy_fn: Callable,
    descriptor,
    receiver_scale: torch.Tensor,
    positions: torch.Tensor,
    moments: torch.Tensor,
    batch: torch.Tensor,
    pbc: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    pos = positions.detach().requires_grad_(True)
    src = moments.detach().requires_grad_(True)
    node_energy = node_energy_fn(pos, src)
    features = torch.autograd.grad(node_energy.sum(), src, create_graph=True)[0]
    features = features * receiver_scale
    return features + descriptor.non_periodic_correction_terms(
        source_feats=src,
        node_positions=pos,
        batch=batch,
        volumes=volume.reshape(1),
        pbc=pbc,
    )


def _energy_fn(
    node_energy_fn: Callable,
    energy_model,
    positions: torch.Tensor,
    moments: torch.Tensor,
    batch: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    pos = positions.detach().requires_grad_(True)
    src = moments.detach().requires_grad_(True)
    node_energy = node_energy_fn(pos, src)
    correction = energy_model.monopole_dipole_correction(
        src, pos, volume.reshape(1), batch
    )
    return node_energy.sum().reshape(1) + correction


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _time_cuda(fn: Callable, repeats: int) -> Timing:
    for _ in range(5):
        output = fn()
        torch.cuda.synchronize()
        del output
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
    return Timing(
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
    )


def _snapshot(fn: Callable) -> torch.Tensor:
    output = fn()
    torch.cuda.synchronize()
    snapshot = output.detach().clone()
    del output
    return snapshot


def _print_timing(num_atoms: int, variant: str, operation: str, timing: Timing):
    print(
        "PME_COMPILE_TIMING "
        f"N={num_atoms} variant={variant} operation={operation} "
        f"median_ms={timing.median_ms:.3f} min_ms={timing.min_ms:.3f} "
        f"max_ms={timing.max_ms:.3f} peak_GiB={timing.peak_gib:.3f}",
        flush=True,
    )


def _compile_node_energy(
    num_atoms: int,
    node_energy_fn: Callable,
    mode: str,
    positions: torch.Tensor,
    moments: torch.Tensor,
) -> tuple[Optional[Callable], Optional[float], Optional[float]]:
    try:
        compiled = torch.compile(node_energy_fn, fullgraph=True, mode=mode)
        pos = positions.detach().requires_grad_(True)
        src = moments.detach().requires_grad_(True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        node_energy = compiled(pos, src)
        torch.cuda.synchronize()
        forward_compile_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        gradient = torch.autograd.grad(node_energy.sum(), src, create_graph=True)[0]
        torch.cuda.synchronize()
        backward_compile_ms = (time.perf_counter() - start) * 1000.0
        del node_energy, gradient, pos, src
        print(
            "PME_COMPILE_COST "
            f"N={num_atoms} mode={mode} "
            f"forward_first_ms={forward_compile_ms:.3f} "
            f"backward_first_ms={backward_compile_ms:.3f}",
            flush=True,
        )
        return compiled, forward_compile_ms, backward_compile_ms
    except Exception as exc:  # pylint: disable=broad-exception-caught
        detail = " ".join(str(exc).split())[:800]
        print(
            f"PME_COMPILE_FAILURE N={num_atoms} mode={mode} "
            f"type={type(exc).__name__} "
            f"detail={detail}",
            flush=True,
        )
        _clear_cuda()
        return None, None, None


def _run_case(num_atoms: int, device: torch.device, dtype: torch.dtype) -> None:
    _clear_cuda()
    positions, moments, cell, batch, pbc, box_length = _build_system(
        num_atoms, device, dtype
    )
    descriptor, energy_model, receiver_scale = _models(device, dtype)
    mesh_size = _mesh_size(box_length)
    mesh_dimensions = (mesh_size, mesh_size, mesh_size)
    cache = _prepare_cache(cell, mesh_dimensions, dtype)
    volume = cache.volume.reshape(1)
    repeats = 30 if num_atoms <= 256 else 20
    print(
        "PME_COMPILE_CASE "
        f"N={num_atoms} box_A={box_length:.2f} mesh={mesh_size}^3 "
        f"repeats={repeats}",
        flush=True,
    )

    eager_uncached_node = _make_node_energy(cell, mesh_dimensions, None)
    eager_cached_node = _make_node_energy(cell, mesh_dimensions, cache)
    variants: dict[str, Callable] = {
        "eager_uncached": eager_uncached_node,
        "eager_cached": eager_cached_node,
    }

    def make_feature(node_fn):
        return lambda: _feature_fn(
            node_fn,
            descriptor,
            receiver_scale,
            positions,
            moments,
            batch,
            pbc,
            volume,
        )

    def make_energy(node_fn):
        return lambda: _energy_fn(
            node_fn, energy_model, positions, moments, batch, volume
        )

    reference_feature = _snapshot(make_feature(eager_uncached_node))
    reference_energy = _snapshot(make_energy(eager_uncached_node))
    cached_feature = _snapshot(make_feature(eager_cached_node))
    cached_energy = _snapshot(make_energy(eager_cached_node))
    torch.testing.assert_close(
        cached_feature, reference_feature, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(cached_energy, reference_energy, rtol=1e-10, atol=1e-10)

    # Warp launches on its own stream, so CUDA-graph capture used by
    # reduce-overhead is unsupported. NVIDIA's benchmark also uses default mode.
    for mode in ("default",):
        compiled, _, _ = _compile_node_energy(
            num_atoms, eager_cached_node, mode, positions, moments
        )
        if compiled is not None:
            variant = f"compiled_{mode.replace('-', '_')}"
            compiled_feature = _snapshot(make_feature(compiled))
            compiled_energy = _snapshot(make_energy(compiled))
            feature_abs = float((compiled_feature - reference_feature).abs().max())
            energy_abs = float((compiled_energy - reference_energy).abs().max())
            torch.testing.assert_close(
                compiled_feature, reference_feature, rtol=1e-8, atol=1e-8
            )
            torch.testing.assert_close(
                compiled_energy, reference_energy, rtol=1e-8, atol=1e-8
            )
            print(
                "PME_COMPILE_PARITY "
                f"N={num_atoms} variant={variant} "
                f"feature_max_abs={feature_abs:.3e} energy_max_abs={energy_abs:.3e}",
                flush=True,
            )
            variants[variant] = compiled

    timings: dict[str, tuple[Timing, Timing]] = {}
    for variant, node_fn in variants.items():
        feature_timing = _time_cuda(make_feature(node_fn), repeats)
        energy_timing = _time_cuda(make_energy(node_fn), repeats)
        _print_timing(num_atoms, variant, "feature", feature_timing)
        _print_timing(num_atoms, variant, "energy", energy_timing)
        timings[variant] = (feature_timing, energy_timing)

    for variant, (feature_timing, energy_timing) in timings.items():
        r1_ms = 2.0 * feature_timing.median_ms + energy_timing.median_ms
        baseline = (
            2.0 * timings["eager_uncached"][0].median_ms
            + timings["eager_uncached"][1].median_ms
        )
        print(
            "PME_COMPILE_SUMMARY "
            f"N={num_atoms} variant={variant} r1_ms={r1_ms:.3f} "
            f"speedup_vs_uncached={baseline / r1_ms:.3f}",
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
        "PME_COMPILE_DEVICE "
        f"name={torch.cuda.get_device_name(0)} dtype={dtype} "
        f"torch={torch.__version__}",
        flush=True,
    )
    for num_atoms in (64, 256, 1024):
        _run_case(num_atoms, device, dtype)


if __name__ == "__main__":
    main()
