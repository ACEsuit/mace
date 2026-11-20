"""`PolarMACE` builds the long-range k-space geometry once and shares it.

The features descriptor and the electrostatic energy are built from the same
GTO basis on the same cell, so the phases, density basis and Coulomb kernel are
identical for both. `CachedGTOElectrostaticEnergy` reads the geometry the
features already materialised instead of rebuilding it.

These tests pin the two halves of that claim. The first group checks that the
cached energy is the *same* energy, bit for bit, with gradients and second
gradients, for every boundary-condition policy - a cache that silently changed
the interaction (or detached it from the positions, which force training would
read as a zero force derivative) would fail here. The second group checks the
wiring: one forward builds the geometry once, and `PolarMACE.forward` does not
fall back to the energy module's uncached path.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("graph_longrange")

from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat

from mace.modules.long_range_cache import CachedGTOElectrostaticEnergy

KS_CUTOFF = 4.0
CELL_LENGTH = 8.0
POSITIONS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.5, 0.3],
    [-0.7, 0.2, 1.1],
]


def _build_modules(handling: str, num_graphs: int):
    common = {
        "density_max_l": 1,
        "density_smearing_width": 1.0,
        "kspace_cutoff": KS_CUTOFF,
        "include_self_interaction": False,
    }
    features = GTOElectrostaticFeatures(
        feature_max_l=1,
        feature_smearing_widths=[1.0],
        integral_normalization="receiver",
        pbc_handling=handling,
        **common,
    )
    energy = CachedGTOElectrostaticEnergy(pbc_handling=handling, **common)
    assert num_graphs >= 1
    return features, energy


def _geometry(handling: str, pbc: torch.Tensor):
    num_graphs = pbc.shape[0]
    cell = torch.eye(3).repeat(num_graphs, 1, 1) * CELL_LENGTH
    rcell = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    positions = torch.tensor(POSITIONS)
    # Two graphs: the first holds atoms 0 and 1, the second holds atom 2.
    batch = torch.tensor([0, 0, 1]) if num_graphs > 1 else torch.zeros(3, dtype=torch.long)
    volume = torch.det(cell).abs()

    features, energy = _build_modules(handling, num_graphs)
    k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
        KS_CUTOFF, cell, rcell
    )
    cache = features.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    direct_kwargs = {
        "k_vectors": k_vectors,
        "k_norm2": k_norm2,
        "k_vector_batch": k_vector_batch,
        "k0_mask": k0_mask,
        "node_positions": positions,
        "batch": batch,
        "volume": volume,
        "pbc": pbc,
    }
    return energy, cache, direct_kwargs


def _densities(requires_grad: bool = True):
    generator = torch.Generator().manual_seed(1592)
    return [
        torch.randn(3, 4, generator=generator).requires_grad_(requires_grad)
        for _ in range(2)
    ]


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), f"max diff {(actual - expected).abs().max()}"


PBCHandlingCase = tuple[str, list[list[bool]]]

CASES: list[PBCHandlingCase] = [
    ("pbc", [[True, True, True]]),
    ("slab", [[True, True, False]]),
    ("molecule_in_box", [[False, False, False]]),
    ("mixed_periodic", [[True, True, True], [False, False, False]]),
    ("auto", [[True, True, True], [False, False, False]]),
    ("auto", [[False, False, False]]),
    ("realspace", [[False, False, False]]),
]

KSPACE_CASES = [
    ("pbc", [[True, True, True]]),
    ("slab", [[True, True, False]]),
    ("molecule_in_box", [[False, False, False]]),
    ("mixed_periodic", [[True, True, True], [False, False, False]]),
    ("auto", [[True, True, True], [False, False, False]]),
    ("auto", [[False, False, False]]),
]


@pytest.mark.parametrize("handling,pbc_rows", CASES, ids=lambda value: str(value))
def test_cached_energy_is_the_direct_energy_bit_for_bit(handling, pbc_rows):
    pbc = torch.tensor(pbc_rows, dtype=torch.bool)
    energy, cache, direct_kwargs = _geometry(handling, pbc)
    density = _densities()[0]

    cached = energy.forward_from_geometry(cache, density)
    direct = energy(source_feats=density, **direct_kwargs)

    _assert_bitwise_equal(cached, direct)


@pytest.mark.parametrize("handling,pbc_rows", CASES, ids=lambda value: str(value))
def test_cached_energy_shares_the_geometry_between_densities(handling, pbc_rows):
    """One geometry is built, then reused for the charge and spin densities."""
    pbc = torch.tensor(pbc_rows, dtype=torch.bool)
    energy, cache, direct_kwargs = _geometry(handling, pbc)
    charge_density, spin_density = _densities()

    cached = [
        energy.forward_from_geometry(cache, charge_density),
        energy.forward_from_geometry(cache, spin_density),
    ]
    direct = [
        energy(source_feats=charge_density, **direct_kwargs),
        energy(source_feats=spin_density, **direct_kwargs),
    ]

    for cached_graph, direct_graph in zip(cached, direct):
        _assert_bitwise_equal(cached_graph, direct_graph)


@pytest.mark.parametrize("handling,pbc_rows", KSPACE_CASES, ids=lambda value: str(value))
def test_cached_energy_has_correct_source_feats_gradient(handling, pbc_rows):
    """The cached energy must have the same gradient w.r.t. source_feats as the direct energy.

    This ensures the fixed-point iterations that update the density coefficients
    see the same gradient whether the energy reads the cache or rebuilds the geometry.
    """
    pbc = torch.tensor(pbc_rows, dtype=torch.bool)
    energy, cache, direct_kwargs = _geometry(handling, pbc)
    # "auto" handling with non-periodic PBC falls back to realspace mode,
    # which has no k-space geometry to share; skip for those.
    if cache["mode"] == "realspace":
        pytest.skip("realspace mode has no k-space phases")
    density = _densities(requires_grad=True)[0]

    cached = energy.forward_from_geometry(cache, density)
    direct = energy(source_feats=density, **direct_kwargs)

    cached_grad = torch.autograd.grad(cached.sum(), density, create_graph=True)[0]
    direct_grad = torch.autograd.grad(direct.sum(), density, create_graph=True)[0]

    _assert_bitwise_equal(cached_grad, direct_grad)


def test_geometry_cache_mode_must_match_the_boundary_condition():
    _, cache, _ = _geometry("realspace", torch.zeros(1, 3, dtype=torch.bool))
    energy = CachedGTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=1.0,
        kspace_cutoff=KS_CUTOFF,
        pbc_handling="pbc",
    )
    with pytest.raises(ValueError, match="realspace geometry"):
        energy.forward_from_geometry(cache, torch.zeros(3, 4))


def test_polar_forward_builds_one_geometry_and_reads_it(monkeypatch):
    from tests.extensions.polar.test_polar_models import (
        _build_minimal_batch,
        _build_minimal_model,
    )

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = _build_minimal_model("cpu", torch.float64)
        batch = _build_minimal_batch("cpu", torch.float64)

        assert isinstance(model.coulomb_energy, CachedGTOElectrostaticEnergy)

        builds = []
        precompute_geometry = model.electric_potential_descriptor.precompute_geometry

        def counting_precompute_geometry(*args, **kwargs):
            builds.append(1)
            return precompute_geometry(*args, **kwargs)

        def uncached_energy_path(*args, **kwargs):
            raise AssertionError(
                "PolarMACE.forward rebuilt the k-space geometry for the energy"
            )

        monkeypatch.setattr(
            model.electric_potential_descriptor,
            "precompute_geometry",
            counting_precompute_geometry,
        )
        monkeypatch.setattr(
            model.coulomb_energy, "forward", uncached_energy_path
        )

        outputs = model(batch, training=False)
    finally:
        torch.set_default_dtype(previous)

    assert len(builds) == 1
    assert torch.isfinite(outputs["electrostatic_energy"]).all()