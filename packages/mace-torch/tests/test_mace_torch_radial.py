"""The radial bases, cutoff, pair repulsion and distance transforms.

The numbers here are the closed forms, committed as decimal literals; they were
produced by the legacy characterization suite (`tests/unit/test_radial.py`) and
are what remains as the module-level pin once the legacy tree is retired. The
live comparison against legacy is `tests/parity/test_radial_parity.py`.

Where the behaviour is exactly representable (a cutoff that returns zero, a
repulsion that switches off past the covalent radii) the assertion is exact
equality, because the padded-batch contract depends on exactly zero.

Every test runs at float32 and float64 (see `conftest.dtype`); a test whose
method needs fp64, such as `gradcheck`, is marked `fp64_only` and runs once.
"""

import itertools

import ase.data
import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_torch.nn.radial import (
    AgnesiTransform,
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
    ZBLBasis,
    polynomial_envelope,
)

# ===========================================================================
# BesselBasis -- f_n(r) = sqrt(2/r_max) * sin(n*pi*r/r_max) / r
# ===========================================================================

BESSEL_R_MAX, BESSEL_N = 5.0, 4

#: r -> the four basis values, sqrt(2/5) * sin(n*pi*r/5) / r for n = 1..4.
BESSEL_REFERENCE = {
    0.9: [
        0.37654068966260074,
        0.6358476387398435,
        0.6971871458425328,
        0.5414615143319496,
    ],
    1.7: [
        0.3260147103245917,
        0.314117569020157,
        -0.02336012437387206,
        -0.33662522050932886,
    ],
    3.0: [
        0.20050031833584858,
        -0.12391601148672815,
        -0.1239160114867282,
        0.20050031833584855,
    ],
}


def test_bessel_basis_values():
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    distances = torch.tensor([[r] for r in BESSEL_REFERENCE])
    assert_close(basis(distances), list(BESSEL_REFERENCE.values()), "bessel")


def test_bessel_buffers_and_trainability():
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    assert_close(
        basis.frequencies,
        np.pi / BESSEL_R_MAX * np.arange(1, BESSEL_N + 1),
        "bessel frequencies",
    )
    assert_close(basis.prefactor, np.sqrt(2.0 / BESSEL_R_MAX), "prefactor")
    assert not basis.frequencies.requires_grad
    assert BesselBasis(r_max=BESSEL_R_MAX, trainable=True).frequencies.requires_grad


@pytest.mark.parametrize("n_edges", [0, 1, 7])
def test_bessel_shape_and_dtype_contract(n_edges, dtype):
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    out = basis(torch.full((n_edges, 1), 1.3))
    assert out.shape == (n_edges, BESSEL_N)
    assert out.dtype == dtype


def test_bessel_is_finite_and_smooth_at_zero_length():
    """`sin(w r) / r` is 0/0 at r = 0 with limit `w`, and its slope there is zero.
    A zero-length padding edge must embed finitely to second order, since force
    training differentiates twice through the basis."""
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    lengths = torch.tensor([[0.0], [1.3]], requires_grad=True)
    out = basis(lengths)
    assert_close(out[0], basis.prefactor * basis.frequencies, "limit at r = 0")
    (first,) = torch.autograd.grad(out.sum(), lengths, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), lengths)
    assert first[0].item() == 0.0
    assert torch.isfinite(second).all()
    # the guard does not touch a nonzero length
    plain = basis.prefactor * (torch.sin(basis.frequencies * lengths[1]) / lengths[1])
    assert torch.equal(out[1], plain)


# ===========================================================================
# ChebyshevBasis -- T_n(x) on the raw distance, no rescaling by r_max
# ===========================================================================

#: x -> T_1..T_4 evaluated by the standard recurrence.
CHEBYCHEV_REFERENCE = {
    0.3: [0.3, -0.8200000000000001, -0.7919999999999999, 0.3448],
    -0.5: [-0.5, -0.5, 1.0, -0.5],
    0.9: [0.9, 0.6200000000000001, 0.2160000000000002, -0.2312000000000003],
}


def test_chebychev_basis_values():
    basis = ChebyshevBasis(num_basis=4)
    x = torch.tensor([[v] for v in CHEBYCHEV_REFERENCE])
    assert_close(basis(x), list(CHEBYCHEV_REFERENCE.values()), "chebychev")


def test_chebychev_diverges_outside_the_unit_interval():
    """Characterization, not endorsement: the polynomials are evaluated on the
    raw distance, so at r > 1 they take the cosh branch and grow without bound.
    A port that mapped r into [-1, 1] would silently change every model trained
    with `--radial_type chebyshev`."""
    basis = ChebyshevBasis(num_basis=3)
    # T_n(1.7): 1.7, 2*1.7^2-1 = 4.78, 4*1.7^3-3*1.7 = 14.552
    assert_close(basis(torch.tensor([[1.7]])), [[1.7, 4.78, 14.552]], "beyond 1")


@pytest.mark.parametrize("include_constant", [True, False])
def test_chebyshev_basis_orders_with_and_without_the_constant(include_constant):
    """Both legacy classes read `T_1..T_n`; with the constant it is `T_0..T_{n-1}`.
    Same width either way."""
    basis = ChebyshevBasis(num_basis=4, include_constant=include_constant)
    x = torch.tensor([[v] for v in CHEBYCHEV_REFERENCE])
    out = basis(x)
    assert out.shape == (3, 4)
    if include_constant:
        # T_0..T_3: the constant column, then the first three literals
        expected = [[1.0, *values[:3]] for values in CHEBYCHEV_REFERENCE.values()]
        assert_close(out, expected, "T_0..T_3")
    else:
        assert_close(out, list(CHEBYCHEV_REFERENCE.values()), "T_1..T_4")


# ===========================================================================
# GaussianBasis -- exp(-0.5 * ((r - c_k) / w)^2), c_k linspace(0, r_max)
# ===========================================================================

GAUSSIAN_R_MAX, GAUSSIAN_N = 5.0, 6

GAUSSIAN_REFERENCE = {
    0.9: [
        0.6669768108584744,
        0.9950124791926823,
        0.5460744266397094,
        0.11025052530448522,
        0.008188701014374074,
        0.00022374579372062055,
    ],
    3.0: [
        0.011108996538242306,
        0.1353352832366127,
        0.6065306597126334,
        1.0,
        0.6065306597126334,
        0.1353352832366127,
    ],
}


def test_gaussian_basis_values():
    basis = GaussianBasis(r_max=GAUSSIAN_R_MAX, num_basis=GAUSSIAN_N)
    distances = torch.tensor([[r] for r in GAUSSIAN_REFERENCE])
    assert_close(basis(distances), list(GAUSSIAN_REFERENCE.values()), "gaussian")
    # the width is folded into a single coefficient at construction
    assert_close(
        basis.exponent_coefficient,
        -0.5 / (GAUSSIAN_R_MAX / (GAUSSIAN_N - 1)) ** 2,
        "gaussian coefficient",
    )
    assert not basis.centers.requires_grad
    assert GaussianBasis(r_max=GAUSSIAN_R_MAX, trainable=True).centers.requires_grad


def test_gaussian_basis_is_not_zero_beyond_r_max():
    """Only the envelope makes a long edge contribute nothing, which is why the
    padding trick depends on `PolynomialCutoff` and not on the basis."""
    basis = GaussianBasis(r_max=GAUSSIAN_R_MAX, num_basis=GAUSSIAN_N)
    assert (basis(torch.tensor([[2 * GAUSSIAN_R_MAX]])) > 0.0).all()


# ===========================================================================
# PolynomialCutoff
# ===========================================================================

CUTOFF_R_MAX, CUTOFF_ORDER = 3.0, 6

#: r -> u(r) for p = 6, r_max = 3.
CUTOFF_REFERENCE = {
    0.0: 1.0,
    1.0: 0.9803383630544124,
    1.5: 0.85546875,
    2.0: 0.5317786922725198,
    2.9: 0.001828263342478209,
}


def test_polynomial_cutoff_values():
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, polynomial_order=CUTOFF_ORDER)
    distances = torch.tensor([[r] for r in CUTOFF_REFERENCE])
    assert_close(cutoff(distances), [[v] for v in CUTOFF_REFERENCE.values()], "cutoff")


def test_polynomial_cutoff_is_exactly_zero_at_and_beyond_r_max(dtype):
    """Exact equality, in both dtypes. At 2*r_max the polynomial is negative and
    the `(x < r_max)` mask produces -0.0, which compares equal to 0.0 -- assert
    the comparison, never the repr."""
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, polynomial_order=CUTOFF_ORDER)
    for distance in (CUTOFF_R_MAX, 2 * CUTOFF_R_MAX, 100 * CUTOFF_R_MAX):
        value = cutoff(torch.tensor(distance, dtype=dtype))
        assert value.item() == 0.0, (distance, value.item())
    inside = cutoff(torch.tensor(CUTOFF_R_MAX * 0.999, dtype=dtype))
    assert inside.item() > 0.0


def test_polynomial_cutoff_derivative_vanishes_at_and_beyond_r_max():
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, polynomial_order=CUTOFF_ORDER)
    distances = torch.tensor(
        [CUTOFF_R_MAX, CUTOFF_R_MAX + 0.5, 2 * CUTOFF_R_MAX], requires_grad=True
    )
    (gradient,) = torch.autograd.grad(cutoff(distances).sum(), distances)
    assert torch.equal(gradient, torch.zeros_like(gradient))


@fp64_only
def test_polynomial_cutoff_meets_r_max_with_a_vanishing_slope():
    """The envelope has a triple root at r_max, so its finite-difference slope
    just inside falls off like h^2 as the offset h halves."""
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, polynomial_order=CUTOFF_ORDER)

    def central_difference(distance, step=1e-7):
        left = cutoff(torch.tensor(distance - step))
        right = cutoff(torch.tensor(distance + step))
        return abs(float(right - left) / (2 * step))

    offsets = (0.05, 0.025, 0.0125, 0.00625)
    slopes = [central_difference(CUTOFF_R_MAX - h) for h in offsets]
    for coarse, fine in itertools.pairwise(slopes):
        ratio = coarse / fine
        assert 3.5 < ratio < 4.5, f"observed order {np.log2(ratio):.2f}, expected 2"
    assert slopes[-1] < 0.01 * central_difference(CUTOFF_R_MAX / 2)


@pytest.mark.parametrize("order", [2, 5, 6, 8])
def test_polynomial_cutoff_is_one_at_zero_for_every_order(order):
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, polynomial_order=order)
    assert cutoff(torch.tensor(0.0)).item() == 1.0
    assert cutoff(torch.tensor(CUTOFF_R_MAX)).item() == 0.0
    sampled = cutoff(torch.linspace(0.0, CUTOFF_R_MAX, 200))
    assert bool((torch.diff(sampled) <= 0).all())


def test_polynomial_envelope_broadcasts_a_per_edge_r_max():
    """The ZBL term hands the envelope one radius per edge: the result is what
    a cutoff module built with that radius gives, edge by edge."""
    distances = torch.tensor([[1.0], [1.0]])
    per_edge_r_max = torch.tensor([[1.5], [0.5]])
    out = polynomial_envelope(distances, per_edge_r_max, CUTOFF_ORDER)
    inside = PolynomialCutoff(r_max=1.5, polynomial_order=CUTOFF_ORDER)
    assert torch.equal(out[0], inside(distances[0]))
    assert out[1].item() == 0.0


# ===========================================================================
# ZBLBasis -- the published Ziegler-Biersack-Littmark screened Coulomb pair
# ===========================================================================
#
#   V(r) = (1/2) * (14.3996 * Z_u * Z_v / r) * phi(r/a) * envelope(r)
#   a    = 0.4543 * 0.529 / (Z_u^0.3 + Z_v^0.3)
#   phi(x) = 0.1818 e^{-3.2x} + 0.5099 e^{-0.9423x}
#          + 0.2802 e^{-0.4028x} + 0.02817 e^{-0.2016x}
#
# The envelope's r_max is the pair's covalent radii sum, not the model cutoff.

#: (Z_u, Z_v, r) -> per-node energy in eV, evaluated from the formula above.
ZBL_REFERENCE = {
    (1, 6, 0.9): 0.04838438085989024,
    (8, 8, 1.2): 0.00924872728685459,
}


@pytest.mark.parametrize("pair", sorted(ZBL_REFERENCE))
def test_zbl_matches_the_published_formula(pair):
    z_u, z_v, distance = pair
    zbl = ZBLBasis(polynomial_order=6)
    node_atomic_numbers = torch.tensor([z_u, z_v])
    lengths = torch.tensor([[distance], [distance]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    energies = zbl(lengths, node_atomic_numbers, edge_index)
    # one directed edge lands on each node, each carrying half the pair energy
    assert energies.shape == (2,)
    assert_close(energies, [ZBL_REFERENCE[pair]] * 2, f"zbl {pair}")
    assert_close(energies.sum(), 2 * ZBL_REFERENCE[pair], "zbl total")


def test_zbl_is_exactly_zero_beyond_the_pair_covalent_radii():
    zbl = ZBLBasis(polynomial_order=6)
    node_atomic_numbers = torch.tensor([1, 6])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    pair_r_max = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    assert pair_r_max == pytest.approx(1.07)
    for distance in (pair_r_max, 1.2, 3.0):
        energies = zbl(
            torch.tensor([[distance], [distance]]), node_atomic_numbers, edge_index
        )
        assert torch.equal(energies, torch.zeros(2)), distance
    inside = zbl(
        torch.tensor([[pair_r_max * 0.99], [pair_r_max * 0.99]]),
        node_atomic_numbers,
        edge_index,
    )
    assert (inside > 0.0).all()


def test_zbl_energy_is_scattered_onto_the_receiver():
    zbl = ZBLBasis(polynomial_order=6)
    # three H atoms, all edges pointing at node 0
    node_atomic_numbers = torch.tensor([1, 1, 1])
    edge_index = torch.tensor([[1, 2], [0, 0]])
    lengths = torch.tensor([[0.6], [0.6]])
    energies = zbl(lengths, node_atomic_numbers, edge_index)
    assert energies.shape == (3,)
    one_edge = zbl(lengths[:1], node_atomic_numbers[:2], edge_index[:, :1])
    assert one_edge[0] > 0.0
    assert_close(energies[0], 2 * one_edge[0], "receiver sum")
    assert torch.equal(energies[1:], torch.zeros(2))


def test_zbl_buffers_and_trainability():
    zbl = ZBLBasis(polynomial_order=6)
    assert_close(zbl.screening_coefficients, [0.1818, 0.5099, 0.2802, 0.02817], "zbl c")
    assert zbl.screening_length_exponent.item() == pytest.approx(0.300)
    assert zbl.screening_length_prefactor.item() == pytest.approx(0.4543)
    assert not zbl.screening_length_exponent.requires_grad
    assert not zbl.screening_length_prefactor.requires_grad
    trainable = ZBLBasis(polynomial_order=6, trainable=True)
    assert trainable.screening_length_exponent.requires_grad
    assert trainable.screening_length_prefactor.requires_grad


# ===========================================================================
# Distance transforms
# ===========================================================================

#: r -> transformed r, for an H-C edge (r_0 = 0.535 Ang = half the radii sum).
#: T(r) = 1 / (1 + a * (r/r_0)^q / (1 + (r/r_0)^(q-p)))
AGNESI_REFERENCE = {0.9: 0.3974261261086945, 1.7: 0.2451457879436009}

#: The same edge through the tanh clamp, whose r_0 is the *full* radii sum
#: (1.07 Ang), giving p_0 = 0.8025 and p_1 = 1.4267.
SOFT_REFERENCE = {0.9: 0.8083566107332905, 1.7: 1.699505533439394}


def _hc_edge(distances):
    lengths = torch.tensor([[float(d)] for d in distances])
    node_atomic_numbers = torch.tensor([1, 6])
    edge_index = torch.stack(
        [
            torch.zeros(len(distances), dtype=torch.long),
            torch.ones(len(distances), dtype=torch.long),
        ]
    )
    return lengths, node_atomic_numbers, edge_index


def test_agnesi_transform_values():
    out = AgnesiTransform()(*_hc_edge(AGNESI_REFERENCE))
    assert_close(out, [[v] for v in AGNESI_REFERENCE.values()], "agnesi")


def test_agnesi_transform_is_monotone_decreasing():
    distances = np.linspace(0.2, 6.0, 200)
    out = AgnesiTransform()(*_hc_edge(distances)).squeeze(-1)
    assert bool((torch.diff(out) < 0).all())
    assert float(out[0]) < 1.0


def test_soft_transform_values():
    out = SoftTransform()(*_hc_edge(SOFT_REFERENCE))
    assert_close(out, [[v] for v in SOFT_REFERENCE.values()], "soft")


def test_soft_transform_clamps_below_p0_and_is_the_identity_above():
    transform = SoftTransform()
    r_0 = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    p_0 = 0.75 * r_0
    short, long = 0.15, 4.0
    out = transform(*_hc_edge([short, long])).squeeze(-1)
    assert float(out[0]) == pytest.approx(p_0, abs=1e-3)
    assert_close(out[1], long, "identity above p_1")


def test_soft_transform_is_monotone_only_above_the_clamp():
    """A real wrinkle of the legacy formula: below p_0 the transform dips about
    5e-4 under p_0 before recovering. Harmless, but a port asserting global
    monotonicity would be asserting something false."""
    transform = SoftTransform()
    r_0 = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    p_0 = 0.75 * r_0

    above = np.linspace(p_0, 6.0, 200)
    assert bool((torch.diff(transform(*_hc_edge(above)).squeeze(-1)) > 0).all())

    below = np.linspace(0.05, p_0, 200)
    values = transform(*_hc_edge(below)).squeeze(-1)
    assert not bool((torch.diff(values) > 0).all())
    assert float(values.min()) < p_0


def test_transforms_are_trainable_on_request():
    agnesi = AgnesiTransform(trainable=True)
    assert agnesi.amplitude.requires_grad
    assert agnesi.exponent_q.requires_grad
    assert agnesi.exponent_p.requires_grad
    assert SoftTransform(trainable=True).steepness.requires_grad
    assert not SoftTransform().steepness.requires_grad


# ===========================================================================
# Radial MLP
# ===========================================================================


def test_radial_mlp_structure_and_shapes(dtype):
    mlp = RadialMLP([8, 16, 4])
    kinds = [type(module).__name__ for module in mlp.layers]
    # no normalisation or activation after the last layer: the output is unbounded
    assert kinds == ["Linear", "LayerNorm", "SiLU", "Linear"]
    assert mlp.channels == [8, 16, 4]
    out = mlp(torch.zeros(5, 8))
    assert out.shape == (5, 4)
    assert out.dtype == dtype


def test_radial_mlp_single_layer_has_no_activation():
    mlp = RadialMLP([3, 2])
    assert [type(module).__name__ for module in mlp.layers] == ["Linear"]
    assert mlp(torch.ones(1, 3)).shape == (1, 2)


# ===========================================================================
# Differentiability: force training runs grad(grad(E)) through every one of these
# ===========================================================================

GRADCHECK_R_MAX = 3.0


def _basis_times_cutoff(kind: str):
    cutoff = PolynomialCutoff(r_max=GRADCHECK_R_MAX, polynomial_order=6)
    basis = {
        "bessel": lambda: BesselBasis(r_max=GRADCHECK_R_MAX, num_basis=4),
        "gaussian": lambda: GaussianBasis(r_max=GRADCHECK_R_MAX, num_basis=5),
        "chebyshev": lambda: ChebyshevBasis(num_basis=4),
    }[kind]()
    return lambda lengths: basis(lengths) * cutoff(lengths)


@fp64_only
@pytest.mark.parametrize("kind", ["bessel", "gaussian", "chebyshev"])
def test_basis_times_cutoff_passes_gradcheck_and_gradgradcheck(kind):
    function = _basis_times_cutoff(kind)
    # strictly inside the cutoff and away from zero, where every term is smooth
    lengths = torch.tensor([[0.7], [1.4], [2.6]], requires_grad=True)
    assert torch.autograd.gradcheck(function, (lengths,))
    assert torch.autograd.gradgradcheck(function, (lengths,))


@fp64_only
@pytest.mark.parametrize("transform", [AgnesiTransform, SoftTransform])
def test_distance_transforms_pass_gradcheck_and_gradgradcheck(transform):
    module = transform()
    _, node_atomic_numbers, edge_index = _hc_edge([0.0, 0.0, 0.0])
    lengths = torch.tensor([[0.7], [1.1], [2.6]], requires_grad=True)

    def function(lengths_):
        return module(lengths_, node_atomic_numbers, edge_index)

    assert torch.autograd.gradcheck(function, (lengths,))
    assert torch.autograd.gradgradcheck(function, (lengths,))


@fp64_only
def test_zbl_passes_gradcheck_and_gradgradcheck():
    zbl = ZBLBasis(polynomial_order=6)
    node_atomic_numbers = torch.tensor([1, 6, 8])
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
    lengths = torch.tensor([[0.6], [0.6], [0.5], [0.5]], requires_grad=True)

    def function(lengths_):
        return zbl(lengths_, node_atomic_numbers, edge_index)

    assert torch.autograd.gradcheck(function, (lengths,))
    assert torch.autograd.gradgradcheck(function, (lengths,))
