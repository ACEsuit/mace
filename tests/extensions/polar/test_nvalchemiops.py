"""Parity tests for the optional PolarMACE NVIDIA electrostatics backend."""

import copy
import math
from unittest import mock

import pytest
import torch

from mace.calculators import MACECalculator
from mace.modules.polar_backends import (
    nvalchemiops_scf_energy,
    nvalchemiops_scf_features,
    prepare_nvalchemiops_scf_cache,
)

from .test_polar_models import (
    _build_minimal_batch,
    _build_minimal_model,
    _clone_batch,
)

pytestmark = pytest.mark.nvalchemiops


def _direct_inputs(device: torch.device):
    dtype = torch.float64
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [1.7, 0.9, 1.1], [2.8, 2.4, 1.5]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    source_feats = torch.tensor(
        [
            [0.4, 0.10, -0.05, 0.02],
            [-0.7, -0.03, 0.04, 0.08],
            [0.3, 0.02, 0.01, -0.06],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    cell = torch.tensor(
        [[[7.0, 0.2, 0.0], [0.0, 6.5, 0.1], [0.1, 0.0, 7.5]]],
        dtype=dtype,
        device=device,
    )
    batch = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
    return positions, source_feats, cell, batch


def _graph_longrange_values(
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
):
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.features import GTOElectrostaticFeatures
    from graph_longrange.kspace import compute_k_vectors_flat

    sigma = 1.1
    receiver_sigmas = [0.9, 1.4]
    k_cutoff = 4.0
    rcell = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        torch.tensor(k_cutoff, dtype=cell.dtype, device=cell.device), cell, rcell
    )
    volume = torch.linalg.det(cell).abs()
    pbc = torch.ones((cell.shape[0], 3), dtype=torch.bool, device=cell.device)

    descriptor = GTOElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=sigma,
        feature_max_l=1,
        feature_smearing_widths=receiver_sigmas,
        kspace_cutoff=k_cutoff,
        include_self_interaction=False,
        integral_normalization="receiver",
    ).to(device=cell.device, dtype=cell.dtype)
    geometry = descriptor.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    features = descriptor.forward_dynamic(
        cache=geometry,
        source_feats=source_feats.unsqueeze(-2),
        pbc=pbc,
    )

    energy_fn = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=sigma,
        kspace_cutoff=k_cutoff,
        include_self_interaction=False,
    ).to(device=cell.device, dtype=cell.dtype)
    energy = energy_fn(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        source_feats=source_feats,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    return features, energy


def _nvalchemiops_values(
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
):
    cache, batch_idx = prepare_nvalchemiops_scf_cache(
        cell,
        batch,
        density_smearing_width=1.1,
        feature_smearing_widths=(0.9, 1.4),
        kspace_cutoff=4.0,
        density_max_l=1,
        feature_max_l=1,
    )
    features = nvalchemiops_scf_features(
        cache,
        positions,
        source_feats,
        batch_idx=batch_idx,
        include_self_interaction=False,
    )
    energy = nvalchemiops_scf_energy(
        cache,
        positions,
        source_feats,
        batch_idx=batch_idx,
        batch=batch,
        num_graphs=int(cell.shape[0]),
        include_self_interaction=False,
    )
    return features, energy


def test_direct_features_energy_and_gradients_match_on_cpu():
    positions, source_feats, cell, batch = _direct_inputs(torch.device("cpu"))
    graph_features, graph_energy = _graph_longrange_values(
        positions, source_feats, cell, batch
    )
    graph_grads = torch.autograd.grad(
        graph_features.square().sum() + graph_energy.sum(),
        (positions, source_feats),
        create_graph=True,
    )
    graph_force_loss_grad = torch.autograd.grad(
        graph_grads[0].square().sum(), source_feats
    )[0]

    positions_nv = positions.detach().clone().requires_grad_(True)
    source_feats_nv = source_feats.detach().clone().requires_grad_(True)
    nv_features, nv_energy = _nvalchemiops_values(
        positions_nv, source_feats_nv, cell, batch
    )
    nv_grads = torch.autograd.grad(
        nv_features.square().sum() + nv_energy.sum(),
        (positions_nv, source_feats_nv),
        create_graph=True,
    )
    nv_force_loss_grad = torch.autograd.grad(
        nv_grads[0].square().sum(), source_feats_nv
    )[0]

    torch.testing.assert_close(nv_features, graph_features, rtol=1e-6, atol=5e-7)
    torch.testing.assert_close(nv_energy, graph_energy, rtol=5e-8, atol=5e-8)
    for actual, expected in zip(nv_grads, graph_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-5)
    torch.testing.assert_close(
        nv_force_loss_grad, graph_force_loss_grad, rtol=1e-5, atol=2e-3
    )


def test_batched_features_and_energy_match_on_cpu():
    positions, source_feats, cell, _ = _direct_inputs(torch.device("cpu"))
    positions = torch.cat(
        [positions.detach(), positions.detach() + torch.tensor([0.3, 0.1, 0.2])]
    ).requires_grad_(True)
    source_feats = torch.cat(
        [source_feats.detach(), 0.8 * source_feats.detach()]
    ).requires_grad_(True)
    second_cell = cell.detach().clone()
    second_cell[:, 0, 0] += 0.4
    second_cell[:, 1, 1] -= 0.2
    cells = torch.cat([cell, second_cell])
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    graph_features, graph_energy = _graph_longrange_values(
        positions, source_feats, cells, batch
    )
    nv_features, nv_energy = _nvalchemiops_values(positions, source_feats, cells, batch)

    torch.testing.assert_close(nv_features, graph_features, rtol=1e-6, atol=5e-7)
    torch.testing.assert_close(nv_energy, graph_energy, rtol=5e-8, atol=5e-8)


def _assert_full_model_parity(
    device: torch.device,
    dtype: torch.dtype,
    quadrupole_feature_corrections: bool = False,
) -> None:
    torch.manual_seed(7)
    graph_model = _build_minimal_model(
        device,
        dtype,
        quadrupole_feature_corrections=quadrupole_feature_corrections,
    ).eval()
    nval_model = copy.deepcopy(graph_model)
    nval_model.set_electrostatics_backend("nvalchemiops")

    batch = _build_minimal_batch(device, dtype)
    batch["unit_shifts"] = torch.zeros_like(batch["shifts"])
    cells = batch["cell"].view(-1, 3, 3)
    batch["volume"] = torch.linalg.det(cells).abs()

    graph_out = graph_model(
        _clone_batch(batch),
        training=False,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )
    nval_out = nval_model(
        _clone_batch(batch),
        training=False,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )

    for key in (
        "energy",
        "electrostatic_energy",
        "forces",
        "virials",
        "stress",
        "density_coefficients",
        "spin_charge_density",
    ):
        torch.testing.assert_close(nval_out[key], graph_out[key], rtol=3e-4, atol=2e-6)

    molecule_batch = _clone_batch(batch)
    molecule_batch["pbc"] = torch.zeros_like(molecule_batch["pbc"])
    graph_molecule = graph_model(
        _clone_batch(molecule_batch), training=False, compute_force=False
    )
    nval_molecule = nval_model(
        _clone_batch(molecule_batch), training=False, compute_force=False
    )
    torch.testing.assert_close(
        nval_molecule["energy"], graph_molecule["energy"], rtol=0.0, atol=0.0
    )


def _assert_slab_model_parity(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    graph_model = _build_minimal_model(device, dtype).eval()
    nval_model = copy.deepcopy(graph_model)
    nval_model.set_electrostatics_backend("nvalchemiops")

    batch = _build_minimal_batch(device, dtype)
    batch["unit_shifts"] = torch.zeros_like(batch["shifts"])
    cells = batch["cell"].view(-1, 3, 3)
    batch["volume"] = torch.linalg.det(cells).abs()
    slab_pbc = torch.ones_like(batch["pbc"], dtype=torch.bool)
    slab_pbc.view(-1, 3)[:, 2] = False
    batch["pbc"] = slab_pbc

    graph_out = graph_model(
        _clone_batch(batch),
        training=False,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )
    with mock.patch(
        "mace.modules.extensions.nvalchemiops_scf_features",
        wraps=nvalchemiops_scf_features,
    ) as feature_spy:
        nval_out = nval_model(
            _clone_batch(batch),
            training=False,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )

    assert feature_spy.called
    for key in (
        "energy",
        "electrostatic_energy",
        "forces",
        "virials",
        "stress",
        "density_coefficients",
        "spin_charge_density",
    ):
        torch.testing.assert_close(nval_out[key], graph_out[key], rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_polarmace_backend_matches_and_falls_back_on_cpu(dtype: torch.dtype):
    _assert_full_model_parity(torch.device("cpu"), dtype)


def test_quadrupole_corrections_use_periodic_backend_and_nonperiodic_fallback():
    _assert_full_model_parity(
        torch.device("cpu"),
        torch.float64,
        quadrupole_feature_corrections=True,
    )


def test_full_polarmace_slab_uses_nvalchemiops_on_cpu():
    _assert_slab_model_parity(torch.device("cpu"), torch.float64)


def test_calculator_rejects_compile_with_nvalchemiops():
    model = _build_minimal_model(torch.device("cpu"), torch.float64).eval()
    with pytest.raises(ValueError, match="requires compile_mode=None"):
        MACECalculator(
            models=model,
            model_type="PolarMACE",
            device="cpu",
            electrostatics_backend="nvalchemiops",
            compile_mode="default",
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_polarmace_backend_matches_on_gpu(dtype: torch.dtype):
    _assert_full_model_parity(torch.device("cuda"), dtype)


@pytest.mark.gpu
def test_full_polarmace_slab_uses_nvalchemiops_on_gpu():
    _assert_slab_model_parity(torch.device("cuda"), torch.float32)
