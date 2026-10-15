"""Stability options for PolarMACE's predicted density: the regularised Fukui
normalisation and the l=1 charge-dipole cap. Both are off by default.
"""

from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE

from tests.helpers import GRAPH_LONGRANGE_AVAILABLE

pytestmark = [
    pytest.mark.polar,
    pytest.mark.skipif(
        not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
    ),
]

DTYPE = torch.float64
SMEARING = 1.5  # what the mace_polar_1 recipe uses


@pytest.fixture(autouse=True)
def _float64_default():
    """The GTO blocks bake constants at the default dtype, so it has to be float64
    when they are built -- and has to be put back, or later tests inherit it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    yield
    torch.set_default_dtype(previous)


def _build_model(**overrides) -> PolarMACE:
    num_elements = 2
    kwargs = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=num_elements,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=torch.zeros(num_elements, dtype=DTYPE),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=SMEARING,
        field_feature_max_l=1,
        field_feature_widths=[1.0],
        field_feature_norms=[1.0, 1.0],
        num_recursion_steps=2,
        field_si=False,
        include_electrostatic_self_interaction=True,
        add_local_electron_energy=True,
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config={
            "type": "AgnosticEmbeddedOneBodyVariableUpdate",
            "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "MLPNonLinearity",
        },
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
    )
    kwargs.update(overrides)
    return PolarMACE(**kwargs).to(dtype=DTYPE)


def _two_graph_batch(multiplicities) -> dict:
    """Two copies of a 2-atom molecule in one batch, with given multiplicities.

    Mixed multiplicities in a single batch are the point: a projection that masks
    per-batch rather than per-graph, or that broadcasts the mask wrongly, passes a
    single-graph test and fails this one.
    """
    n_graphs = len(multiplicities)
    positions, batch, src, dst, node_attrs = [], [], [], [], []
    for g in range(n_graphs):
        base = 2 * g
        positions += [[0.0, 0.0, 0.0], [1.2, 0.1, -0.2]]
        batch += [g, g]
        src += [base, base + 1]
        dst += [base + 1, base]
        node_attrs += [[1.0, 0.0], [0.0, 1.0]]

    n = 2 * n_graphs
    cell_len = 10.0
    cell = torch.eye(3, dtype=DTYPE).unsqueeze(0).repeat(n_graphs, 1, 1) * cell_len
    rcell = (
        torch.eye(3, dtype=DTYPE).unsqueeze(0).repeat(n_graphs, 1, 1)
        * (2.0 * math.pi / cell_len)
    )
    return {
        "positions": torch.tensor(positions, dtype=DTYPE),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "shifts": torch.zeros((len(src), 3), dtype=DTYPE),
        "node_attrs": torch.tensor(node_attrs, dtype=DTYPE),
        "batch": torch.tensor(batch, dtype=torch.long),
        "ptr": torch.arange(0, n + 1, 2, dtype=torch.long),
        "cell": cell.reshape(-1, 9),
        "rcell": rcell.reshape(-1, 9),
        "volume": torch.full((n_graphs,), cell_len**3, dtype=DTYPE),
        "pbc": torch.zeros((n_graphs, 3), dtype=torch.bool),
        "external_field": torch.zeros((n_graphs, 3), dtype=DTYPE),
        "fermi_level": torch.zeros((n_graphs,), dtype=DTYPE),
        "total_charge": torch.zeros((n_graphs,), dtype=DTYPE),
        "total_spin": torch.tensor(multiplicities, dtype=DTYPE),
        "density_coefficients": torch.zeros((n, 1), dtype=DTYPE),
    }


# --------------------------------------------------------------------------- #
# The l=1 charge-dipole cap.
# --------------------------------------------------------------------------- #


def _density(charge_dipoles, spin_dipole=(0.3, -0.2, 0.1), monopoles=(0.4, -0.1)):
    """(n, 2, 4) alpha/beta coefficients with the given charge dipoles."""
    n = len(charge_dipoles)
    charge = torch.zeros(n, 4, dtype=DTYPE)
    charge[:, 0] = monopoles[0] + monopoles[1]
    charge[:, 1:4] = torch.tensor(charge_dipoles, dtype=DTYPE)
    spin = torch.zeros(n, 4, dtype=DTYPE)
    spin[:, 0] = monopoles[0] - monopoles[1]
    spin[:, 1:4] = torch.tensor(spin_dipole, dtype=DTYPE)
    return 0.5 * torch.stack((charge + spin, charge - spin), dim=1)


def test_dipole_cap_is_exactly_the_identity_below_the_onset():
    model = _build_model(dipole_cap=[2.0, 3.0])
    density = _density([[0.3, -0.1, 0.2], [1.2, 0.9, -0.5], [0.0, 0.0, 0.0]])
    assert torch.equal(model._cap_charge_dipole(density), density)


def test_dipole_cap_bounds_the_charge_dipole_and_leaves_the_rest_alone():
    model = _build_model(dipole_cap=[2.0, 3.0])
    density = _density([[25.0, 2.0, -6.0], [2.5, 0.0, 0.0], [1e6, -1e6, 3.0]])
    capped = model._cap_charge_dipole(density)
    charge = capped[:, 0] + capped[:, 1]
    spin = capped[:, 0] - capped[:, 1]
    norms = torch.linalg.norm(charge[:, 1:4], dim=-1)
    assert torch.all(norms <= 3.0) and torch.all(norms > 2.0)
    before = density[:, 0] + density[:, 1]
    # Direction preserved, monopoles and the whole spin density untouched.
    cos = torch.nn.functional.cosine_similarity(charge[:, 1:4], before[:, 1:4], dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-12)
    assert torch.allclose(charge[:, 0], before[:, 0], rtol=0.0, atol=1e-14)
    assert torch.allclose(spin, density[:, 0] - density[:, 1], rtol=0.0, atol=1e-14)


def test_dipole_cap_bounds_large_float32_dipoles():
    """Squaring a 1e20 dipole overflows float32; the cap must still land below the ceiling."""
    model = _build_model(dipole_cap=[2.0, 3.0])
    density = _density([[1e20, 0.0, 0.0], [0.3, -0.1, 0.2]]).float().requires_grad_(True)
    capped = model._cap_charge_dipole(density)
    assert capped.dtype == torch.float32
    norms = torch.linalg.norm((capped[:, 0] + capped[:, 1])[:, 1:4], dim=-1)
    assert 2.9 < norms[0] <= 3.0
    assert torch.equal(capped[1], density[1])
    capped.sum().backward()
    assert torch.all(torch.isfinite(density.grad))


def test_dipole_cap_is_smooth_through_the_onset():
    """Value, slope and curvature match across the onset, so forces stay continuous."""
    model = _build_model(dipole_cap=[2.0, 3.0])

    def capped_norm(r):
        density = _density([[r, 0.0, 0.0]])
        out = model._cap_charge_dipole(density)
        return float((out[:, 0] + out[:, 1])[0, 1])

    h = 1e-4
    below = [capped_norm(2.0 - k * h) for k in (0, 1, 2)]
    above = [capped_norm(2.0 + k * h) for k in (0, 1, 2)]
    slope_below = (below[0] - below[1]) / h
    slope_above = (above[1] - above[0]) / h
    assert abs(slope_below - 1.0) < 1e-6 and abs(slope_above - 1.0) < 1e-3
    curvature_above = (above[2] - 2 * above[1] + above[0]) / h**2
    assert abs(curvature_above) < 1e-2


def test_dipole_cap_second_derivative_is_finite_at_a_zero_dipole():
    """The force loss differentiates the forces; a zero dipole must not make that NaN."""
    model = _build_model(dipole_cap=[2.0, 3.0])
    density = _density([[0.3, 0.1, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).requires_grad_(True)
    (grad,) = torch.autograd.grad((model._cap_charge_dipole(density) ** 2).sum(), density, create_graph=True)
    (grad**2).sum().backward()
    assert torch.all(torch.isfinite(grad)) and torch.all(torch.isfinite(density.grad))


def test_dipole_cap_bounds_a_model_driven_to_huge_dipoles():
    """Adversarial: blow up the source maps; only the cap keeps the dipole bounded."""
    torch.manual_seed(0)
    free = _build_model()
    capped = _build_model(dipole_cap=[2.0, 3.0])
    capped.load_state_dict(free.state_dict())
    for model in (free, capped):
        with torch.no_grad():
            for p in model.lr_source_maps.parameters():
                p.mul_(1e3)
    data = _two_graph_batch([1.0, 3.0])
    p_free = torch.linalg.norm(free(dict(data), compute_force=False)["density_coefficients"][:, 1:4], dim=-1)
    out = capped(dict(data), compute_force=False)
    p_capped = torch.linalg.norm(out["density_coefficients"][:, 1:4], dim=-1)
    assert p_free.max() > 3.0
    assert torch.all(p_capped <= 3.0 + 1e-9)
    assert torch.all(torch.isfinite(out["energy"]))


def test_dipole_cap_rejects_a_bad_interval():
    with pytest.raises(ValueError):
        _build_model(dipole_cap=[3.0, 2.0])


# --------------------------------------------------------------------------- #
# Regularised signed Fukui normalisation.
# --------------------------------------------------------------------------- #


def _fukui_batch():
    """Three graphs: well-conditioned, nearly cancelling, exactly cancelling, plus a lone atom."""
    f = torch.tensor(
        [[0.5, 0.2], [0.3, 0.3], [0.2, 0.5],          # graph 0: sums 1.0, 1.0
         [1.0, 0.8], [-0.99, -0.8], [0.0, 0.001],     # graph 1: sums 0.01, 0.001
         [0.7, -0.4], [-0.7, 0.4],                    # graph 2: sums exactly 0
         [-0.3, 2.0]],                                # graph 3: a lone atom
        dtype=DTYPE,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3])
    return f, batch


def test_fukui_regularisation_keeps_every_graph_summing_to_one_and_bounds_the_weights():
    model = _build_model(fukui_regularisation=0.05)
    f, batch = _fukui_batch()
    w = model._normalise_fukui(f, batch, 4)
    sums = torch.zeros(4, 2, dtype=DTYPE).index_add_(0, batch, w)
    assert torch.allclose(sums, torch.ones_like(sums), rtol=0.0, atol=1e-12)
    assert w.abs().max() <= 1.0 / (2 * 0.05) + 1.0  # ~1/(2r) plus the even share
    unregularised = _build_model()._normalise_fukui(f, batch, 4)
    assert unregularised.abs().max() > 50.0  # the singular division it replaces


def test_fukui_regularisation_barely_moves_a_well_conditioned_graph():
    f, batch = _fukui_batch()
    w = _build_model(fukui_regularisation=0.05)._normalise_fukui(f, batch, 4)
    plain = _build_model()._normalise_fukui(f, batch, 4)
    # graph 0 has no cancellation: relative change ~ r^2
    assert torch.allclose(w[:3], plain[:3], rtol=0.0, atol=5e-3)
    # the lone atom gets exactly one, as before
    assert torch.allclose(w[8], torch.ones(2, dtype=DTYPE), rtol=0.0, atol=1e-12)


def test_fukui_regularisation_is_smooth_through_an_exact_cancellation():
    """No sign flip at sum = 0: the weights move continuously as the sum crosses zero."""
    batch = torch.tensor([0, 0, 0])
    ts = torch.linspace(-0.4, 0.0, 800, dtype=DTYPE)  # the sum 0.2 + t crosses zero
    def max_step(model):
        w = torch.stack([model._normalise_fukui(torch.tensor([[1.0], [-1.0 + t], [0.2]], dtype=DTYPE).repeat(1, 2), batch, 1) for t in ts])
        return (w[1:] - w[:-1]).abs().max()
    assert max_step(_build_model(fukui_regularisation=0.05)) < 0.05
    assert max_step(_build_model()) > 10.0  # 1/s flips sign across s = 0
    model = _build_model(fukui_regularisation=0.05)
    f = torch.tensor([[1.0, 1.0], [-1.19, -1.19], [0.2, 0.2]], dtype=DTYPE, requires_grad=True)
    (g,) = torch.autograd.grad((model._normalise_fukui(f, batch, 1) ** 2).sum(), f, create_graph=True)
    (g**2).sum().backward()
    assert torch.all(torch.isfinite(g)) and torch.all(torch.isfinite(f.grad))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fukui_regularisation_is_finite_to_second_order_on_all_zero_sources(dtype):
    """A graph whose sources are all exactly zero gets uniform weights, and the force
    loss's double backward through it stays finite."""
    model = _build_model(fukui_regularisation=0.1)
    batch = torch.tensor([0, 0, 0, 1, 1])
    f = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.6, 0.3], [0.2, 0.5]],
        dtype=dtype,
        requires_grad=True,
    )
    w = model._normalise_fukui(f, batch, 2)
    assert torch.allclose(w[:3], torch.full_like(w[:3], 1.0 / 3.0), rtol=0.0, atol=1e-6)
    (g,) = torch.autograd.grad((w * torch.arange(1.0, 11.0, dtype=dtype).view(5, 2)).sum() ** 2, f, create_graph=True)
    (g**2).sum().backward()
    assert torch.all(torch.isfinite(w)) and torch.all(torch.isfinite(g))
    assert torch.all(torch.isfinite(f.grad))


def test_fukui_regularisation_is_finite_on_large_float32_sources():
    """Squaring sources of 1e20 overflows float32; the weights are scale invariant, so
    they must match those of the same sources at unit scale."""
    model = _build_model(fukui_regularisation=0.1)
    f, batch = _fukui_batch()
    big = (f * 1e20).float().requires_grad_(True)
    w = model._normalise_fukui(big, batch, 4)
    assert w.dtype == torch.float32
    assert torch.allclose(w.double(), model._normalise_fukui(f, batch, 4), rtol=0.0, atol=1e-5)
    (w**2).sum().backward()
    assert torch.all(torch.isfinite(big.grad))


def test_fukui_regularisation_keeps_the_total_charge_on_every_graph():
    torch.manual_seed(0)
    model = _build_model(fukui_regularisation=0.05)
    data = _two_graph_batch([1.0, 3.0])
    data["total_charge"] = torch.tensor([1.0, -2.0], dtype=DTYPE)
    out = model(dict(data), compute_force=True)
    per_graph = torch.zeros(2, dtype=DTYPE).index_add_(0, data["batch"], out["density_coefficients"][:, 0])
    assert torch.allclose(per_graph, data["total_charge"], rtol=0.0, atol=1e-10)


def test_fukui_regularisation_is_off_by_default_and_rejects_a_bad_value():
    assert _build_model().fukui_regularisation == 0.0
    with pytest.raises(ValueError):
        _build_model(fukui_regularisation=1.5)


def test_fukui_regularisation_forces_stay_continuous_when_one_source_crosses_zero():
    """A single source crossing zero, with the sum well away from zero, must not kink the
    weights: their derivative feeds the forces. A plain sum|f| has a cusp there."""
    model = _build_model(fukui_regularisation=0.1)
    batch = torch.tensor([0, 0, 0])

    def slope(t0):
        t = torch.tensor(t0, dtype=DTYPE, requires_grad=True)
        f = torch.stack([torch.tensor(1.0, dtype=DTYPE), torch.tensor(-0.9, dtype=DTYPE), t]).unsqueeze(1).repeat(1, 2)
        (g,) = torch.autograd.grad(model._normalise_fukui(f, batch, 1)[0, 0], t)
        return float(g)

    left, right = slope(-1e-7), slope(1e-7)
    assert abs(left - right) < 1e-2  # sum|f| gives a jump of about 3.5 here


def test_stability_flags_reach_a_foundation_fine_tune():
    """--fukui_regularisation and --dipole_cap must apply when fine-tuning a foundation
    model, including an explicit 0 / 'none', and a foundation's own settings must survive
    when the flags are unset."""
    import argparse

    from mace.tools.model_script_utils import _build_model as build_from_args
    from mace.tools.scripts_utils import extract_config_mace_model

    plain = extract_config_mace_model(_build_model())
    args = argparse.Namespace(model="PolarMACE", fukui_regularisation=0.1, dipole_cap="[2.0,3.0]")
    tuned = build_from_args(args, None, dict(plain), ["Default"])
    assert tuned.fukui_regularisation == 0.1 and tuned.dipole_cap == [2.0, 3.0]

    stable = extract_config_mace_model(_build_model(fukui_regularisation=0.1, dipole_cap=[2.0, 3.0]))
    omitted = argparse.Namespace(model="PolarMACE", fukui_regularisation=None, dipole_cap=None)
    kept = build_from_args(omitted, None, dict(stable), ["Default"])
    assert kept.fukui_regularisation == 0.1 and kept.dipole_cap == [2.0, 3.0]

    # an explicit 0 / 'none' switches an inherited setting off -- an unregularised control run
    off = argparse.Namespace(model="PolarMACE", fukui_regularisation=0.0, dipole_cap="none")
    control = build_from_args(off, None, dict(stable), ["Default"])
    assert control.fukui_regularisation == 0.0 and control.dipole_cap is None
