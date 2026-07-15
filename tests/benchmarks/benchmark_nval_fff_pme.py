"""GPU crossover study for exact FFF and padded multipole PME electrostatics."""

import gc
import math
import os
import statistics
from dataclasses import dataclass

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
PME_ALPHA = 1.0e6


@dataclass
class Case:
    num_atoms: int
    padding: float
    mesh_spacing: float = MESH_SPACING


def _mesh_size(box_length: float, mesh_spacing: float = MESH_SPACING) -> int:
    requested = math.ceil(box_length / mesh_spacing)
    return max(16, 8 * math.ceil(requested / 8))


def _build_system(case: Case, device: torch.device, dtype: torch.dtype):
    side = math.ceil(case.num_atoms ** (1.0 / 3.0))
    axis = torch.arange(side, device=device, dtype=dtype) * LATTICE_SPACING
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    positions = xyz.reshape(-1, 3)[: case.num_atoms].contiguous()

    generator = torch.Generator(device=device).manual_seed(17 + case.num_atoms)
    positions = positions + 0.08 * torch.randn(
        positions.shape, device=device, dtype=dtype, generator=generator
    )
    positions = positions - positions.amin(dim=0) + case.padding
    extent = positions.amax(dim=0) - positions.amin(dim=0)
    box_length = float(extent.max().item() + 2.0 * case.padding)
    cell = torch.eye(3, device=device, dtype=dtype) * box_length

    charges = torch.randn(
        case.num_atoms, device=device, dtype=dtype, generator=generator
    )
    charges = charges - charges.mean()
    dipoles = 0.1 * torch.randn(
        (case.num_atoms, 3), device=device, dtype=dtype, generator=generator
    )
    moments = torch.cat([charges[:, None], dipoles[:, [1, 2, 0]]], dim=-1).contiguous()
    batch = torch.zeros(case.num_atoms, device=device, dtype=torch.long)
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


def _padded_pme_node_energy(positions, moments, cell, mesh_size):
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


def _padded_pme_features(
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
    pos, src, node_energy = _padded_pme_node_energy(positions, moments, cell, mesh_size)
    features = (
        torch.autograd.grad(node_energy.sum(), src, create_graph=True)[0]
        * receiver_scale
    )
    features = features + descriptor.non_periodic_correction_terms(
        source_feats=src,
        node_positions=pos,
        batch=batch,
        volumes=volume,
        pbc=pbc,
    )
    return features


def _padded_pme_energy(
    energy_fn,
    positions,
    moments,
    cell,
    batch,
    volume,
    mesh_size,
):
    pos, src, node_energy = _padded_pme_node_energy(positions, moments, cell, mesh_size)
    correction = energy_fn.monopole_dipole_correction(src, pos, volume, batch)
    return node_energy.sum().reshape(1) + correction


def _time_cuda(fn, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        outputs = fn()
        del outputs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
        del outputs
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    return statistics.median(timings), peak_gib


def _run_case(case: Case, device: torch.device, dtype: torch.dtype) -> None:
    torch.cuda.empty_cache()
    gc.collect()
    system = _build_system(case, device, dtype)
    positions, moments, cell, batch, pbc, volume, box_length = system
    descriptor, energy_fn = _models(device, dtype)
    scale = _receiver_scale(device, dtype)
    mesh_size = _mesh_size(box_length, case.mesh_spacing)

    exact_features_fn = lambda: _exact_features(descriptor, positions, moments, batch)
    exact_energy_fn = lambda: _exact_energy(energy_fn, positions, moments, batch)
    pme_features_fn = lambda: _padded_pme_features(
        descriptor,
        positions,
        moments,
        cell,
        batch,
        pbc,
        volume,
        mesh_size,
        scale,
    )
    pme_energy_fn = lambda: _padded_pme_energy(
        energy_fn,
        positions,
        moments,
        cell,
        batch,
        volume,
        mesh_size,
    )

    try:
        exact_features = exact_features_fn()
        exact_energy = exact_energy_fn()
        pme_features = pme_features_fn()
        pme_energy = pme_energy_fn()
        torch.cuda.synchronize()
        feature_rel_error = float(
            torch.linalg.vector_norm(pme_features - exact_features)
            / torch.linalg.vector_norm(exact_features)
        )
        energy_error_per_atom = float(
            (pme_energy - exact_energy).abs().max() / case.num_atoms
        )
        del exact_features, exact_energy, pme_features, pme_energy

        repeats = 5 if case.num_atoms <= 256 else 3 if case.num_atoms <= 512 else 2
        exact_feature_ms, exact_feature_peak = _time_cuda(
            exact_features_fn, warmup=1, repeats=repeats
        )
        pme_feature_ms, pme_feature_peak = _time_cuda(
            pme_features_fn, warmup=2, repeats=repeats
        )
        exact_energy_ms, exact_energy_peak = _time_cuda(
            exact_energy_fn, warmup=1, repeats=repeats
        )
        pme_energy_ms, pme_energy_peak = _time_cuda(
            pme_energy_fn, warmup=2, repeats=repeats
        )
        exact_polar_r1_ms = 2.0 * exact_feature_ms + exact_energy_ms
        pme_polar_r1_ms = 2.0 * pme_feature_ms + pme_energy_ms
        exact_polar_r2_ms = 4.0 * exact_feature_ms + exact_energy_ms
        pme_polar_r2_ms = 4.0 * pme_feature_ms + pme_energy_ms
        print(
            "FFF_PME_RESULT "
            f"N={case.num_atoms} padding_A={case.padding:.1f} "
            f"mesh_spacing_A={case.mesh_spacing:.3f} "
            f"box_A={box_length:.2f} mesh={mesh_size}^3 "
            f"feature_rel_error={feature_rel_error:.6e} "
            f"energy_error_eV_per_atom={energy_error_per_atom:.6e} "
            f"exact_feature_ms={exact_feature_ms:.3f} "
            f"pme_feature_ms={pme_feature_ms:.3f} "
            f"feature_speedup={exact_feature_ms / pme_feature_ms:.3f} "
            f"exact_energy_ms={exact_energy_ms:.3f} "
            f"pme_energy_ms={pme_energy_ms:.3f} "
            f"energy_speedup={exact_energy_ms / pme_energy_ms:.3f} "
            f"exact_polar_r1_ms={exact_polar_r1_ms:.3f} "
            f"pme_polar_r1_ms={pme_polar_r1_ms:.3f} "
            f"polar_r1_speedup={exact_polar_r1_ms / pme_polar_r1_ms:.3f} "
            f"exact_polar_r2_ms={exact_polar_r2_ms:.3f} "
            f"pme_polar_r2_ms={pme_polar_r2_ms:.3f} "
            f"polar_r2_speedup={exact_polar_r2_ms / pme_polar_r2_ms:.3f} "
            f"exact_feature_peak_GiB={exact_feature_peak:.3f} "
            f"pme_feature_peak_GiB={pme_feature_peak:.3f} "
            f"exact_energy_peak_GiB={exact_energy_peak:.3f} "
            f"pme_energy_peak_GiB={pme_energy_peak:.3f}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        print(
            "FFF_PME_RESULT "
            f"N={case.num_atoms} padding_A={case.padding:.1f} "
            f"mesh_spacing_A={case.mesh_spacing:.3f} "
            f"box_A={box_length:.2f} mesh={mesh_size}^3 OOM={exc}",
            flush=True,
        )
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    print(
        "FFF_PME_DEVICE "
        f"name={torch.cuda.get_device_name(0)} dtype={dtype} "
        f"mesh_spacing_A={MESH_SPACING}",
        flush=True,
    )

    print("FFF_PME_SECTION size_scaling", flush=True)
    for num_atoms in (64, 128, 256, 512, 1024):
        _run_case(Case(num_atoms=num_atoms, padding=12.0), device, dtype)

    print("FFF_PME_SECTION padding_sweep", flush=True)
    for padding in (4.0, 8.0, 12.0, 16.0, 24.0):
        _run_case(Case(num_atoms=256, padding=padding), device, dtype)

    print("FFF_PME_SECTION mesh_sweep", flush=True)
    for mesh_spacing in (1.0, 0.75, 0.5, 0.375, 0.25):
        _run_case(
            Case(num_atoms=256, padding=16.0, mesh_spacing=mesh_spacing),
            device,
            dtype,
        )


if __name__ == "__main__":
    main()
