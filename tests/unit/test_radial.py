"""Characterization of `mace/modules/radial.py` and the radial-embedding
assembly in `mace/modules/blocks.py`.

Everything in this file is pure mathematics: a distance in, a number out, no
learned parameters and no graph beyond the element pair an edge connects. It
ports into the rewrite nearly verbatim, so these values are the spec the port
is checked against rather than a snapshot of what the code happens to do.

Two conventions, both deliberate:

* **Reference values are committed as decimal literals AND re-derived from
  the closed form.** The literals are what a port is measured against; the
  derivation is what says the literals are the formula rather than a
  photograph of the implementation. A change to either side alone fails.
* **Tolerances are imported, never written here.** The one table lives in
  `tests/golden/harness.py`; the row these tests use is `closed_form_fp64`,
  which is not a golden row -- nothing here crosses a machine or a device,
  only the order of the fp64 arithmetic differs.

Where the behaviour is exactly representable -- a cutoff that returns zero, a
repulsion that switches off past the covalent radii -- the assertion is exact
equality with no row at all. That is not pedantry: `build_fake_padding_graph`
(`mace/data/padding_tools.py:81-106`) gives every padding edge a self-loop
with a shift of `2 * r_max` precisely so the envelope annihilates it, and
"very small" would not do.
"""

import ase.data
import numpy as np
import pytest
import torch

from mace.modules.blocks import RadialEmbeddingBlock
from mace.modules.radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
    ZBLBasis,
)
from tests.golden.harness import tolerance

CLOSED_FORM = tolerance("closed_form_fp64")


@pytest.fixture(name="fp64")
def fixture_fp64():
    """These modules read `torch.get_default_dtype()` at construction, and the
    suite runs in float32. Nothing below is a float32 claim."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def assert_close(actual, expected, what=""):
    """The one comparison used for a closed-form check, at the imported row."""
    actual = np.asarray(
        actual.detach().numpy() if torch.is_tensor(actual) else actual, dtype=float
    )
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    deviation = np.abs(actual - expected)
    bound = CLOSED_FORM.atol + CLOSED_FORM.rtol * np.abs(expected)
    worst = int(np.argmax(deviation - bound)) if deviation.size else 0
    assert np.all(deviation <= bound), (
        f"{what}: worst deviation {deviation.ravel()[worst]:.3e} at index "
        f"{worst} exceeds the '{CLOSED_FORM.name}' row"
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


def test_bessel_reference_values_are_the_closed_form():
    n = np.arange(1, BESSEL_N + 1)
    for distance, expected in BESSEL_REFERENCE.items():
        closed_form = (
            np.sqrt(2.0 / BESSEL_R_MAX)
            * np.sin(n * np.pi * distance / BESSEL_R_MAX)
            / distance
        )
        assert_close(closed_form, expected, f"bessel closed form at r={distance}")


def test_bessel_basis_values(fp64):  # pylint: disable=unused-argument
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    distances = torch.tensor([[r] for r in BESSEL_REFERENCE])
    assert_close(
        basis(distances),
        list(BESSEL_REFERENCE.values()),
        "bessel",
    )


def test_bessel_buffers_and_trainability(fp64):  # pylint: disable=unused-argument
    basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
    assert_close(
        basis.bessel_weights,
        np.pi / BESSEL_R_MAX * np.arange(1, BESSEL_N + 1),
        "bessel weights",
    )
    assert_close(basis.prefactor, np.sqrt(2.0 / BESSEL_R_MAX), "prefactor")
    assert not basis.bessel_weights.requires_grad
    assert BesselBasis(r_max=BESSEL_R_MAX, trainable=True).bessel_weights.requires_grad
    assert "trainable=False" in repr(basis)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_edges", [0, 1, 7])
def test_bessel_shape_and_dtype_contract(dtype, n_edges):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        basis = BesselBasis(r_max=BESSEL_R_MAX, num_basis=BESSEL_N)
        out = basis(torch.full((n_edges, 1), 1.3, dtype=dtype))
        assert out.shape == (n_edges, BESSEL_N)
        assert out.dtype == dtype
    finally:
        torch.set_default_dtype(previous)


# ===========================================================================
# ChebychevBasis -- T_n(x), and it does NOT rescale by r_max
# ===========================================================================

#: x -> T_1..T_4 evaluated by the standard recurrence.
CHEBYCHEV_REFERENCE = {
    0.3: [0.3, -0.8200000000000001, -0.7919999999999999, 0.3448],
    -0.5: [-0.5, -0.5, 1.0, -0.5],
    0.9: [0.9, 0.6200000000000001, 0.2160000000000002, -0.2312000000000003],
}


def test_chebychev_reference_values_are_the_recurrence():
    for x, expected in CHEBYCHEV_REFERENCE.items():
        closed_form = [x, 2 * x**2 - 1, 4 * x**3 - 3 * x, 8 * x**4 - 8 * x**2 + 1]
        assert_close(closed_form, expected, f"chebychev closed form at x={x}")


def test_chebychev_basis_values(fp64):  # pylint: disable=unused-argument
    basis = ChebychevBasis(r_max=BESSEL_R_MAX, num_basis=4)
    x = torch.tensor([[v] for v in CHEBYCHEV_REFERENCE])
    assert_close(basis(x), list(CHEBYCHEV_REFERENCE.values()), "chebychev")


def test_chebychev_ignores_r_max_and_diverges_outside_the_unit_interval(
    fp64,
):  # pylint: disable=unused-argument
    """Characterization, not endorsement. `r_max` is stored and never used:
    the polynomials are evaluated on the raw distance, so at r > 1 they take
    the cosh branch and grow without bound instead of oscillating. Two bases
    built with different `r_max` return the same numbers. A port that
    "fixed" this by mapping r into [-1, 1] would silently change every model
    trained with `--radial_type chebyshev`.
    """
    near = ChebychevBasis(r_max=3.0, num_basis=3)
    far = ChebychevBasis(r_max=30.0, num_basis=3)
    distances = torch.tensor([[1.7]])
    assert torch.equal(near(distances), far(distances))
    # T_n(1.7): 1.7, 2*1.7^2-1 = 4.78, 4*1.7^3-3*1.7 = 14.552
    assert_close(near(distances), [[1.7, 4.78, 14.552]], "chebychev beyond 1")


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


def test_gaussian_reference_values_are_the_closed_form():
    centres = np.linspace(0.0, GAUSSIAN_R_MAX, GAUSSIAN_N)
    width = GAUSSIAN_R_MAX / (GAUSSIAN_N - 1)
    for distance, expected in GAUSSIAN_REFERENCE.items():
        closed_form = np.exp(-0.5 * ((distance - centres) / width) ** 2)
        assert_close(closed_form, expected, f"gaussian closed form at r={distance}")


def test_gaussian_basis_values(fp64):  # pylint: disable=unused-argument
    basis = GaussianBasis(r_max=GAUSSIAN_R_MAX, num_basis=GAUSSIAN_N)
    distances = torch.tensor([[r] for r in GAUSSIAN_REFERENCE])
    assert_close(basis(distances), list(GAUSSIAN_REFERENCE.values()), "gaussian")
    # the width is folded into a single coefficient at construction
    assert_close(
        basis.coeff, -0.5 / (GAUSSIAN_R_MAX / (GAUSSIAN_N - 1)) ** 2, "gaussian coeff"
    )
    assert not basis.gaussian_weights.requires_grad
    assert GaussianBasis(
        r_max=GAUSSIAN_R_MAX, trainable=True
    ).gaussian_weights.requires_grad


def test_gaussian_basis_is_not_zero_beyond_r_max(fp64):  # pylint: disable=unused-argument
    """Unlike the cutoff, a Gaussian never reaches zero. Only the envelope
    makes a long edge contribute nothing, which is why the padding trick
    depends on `PolynomialCutoff` and not on the basis."""
    basis = GaussianBasis(r_max=GAUSSIAN_R_MAX, num_basis=GAUSSIAN_N)
    assert (basis(torch.tensor([[2 * GAUSSIAN_R_MAX]])) > 0.0).all()


# ===========================================================================
# PolynomialCutoff
# ===========================================================================

CUTOFF_R_MAX, CUTOFF_P = 3.0, 6

CUTOFF_REFERENCE = {
    0.0: 1.0,
    1.0: 0.9803383630544124,
    1.5: 0.85546875,
    2.0: 0.5317786922725198,
    2.9: 0.001828263342478209,
}


def _envelope(distance, r_max=CUTOFF_R_MAX, p=CUTOFF_P):
    u = distance / r_max
    return (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * u**p
        + p * (p + 2.0) * u ** (p + 1)
        - (p * (p + 1.0) / 2.0) * u ** (p + 2)
    )


def test_cutoff_reference_values_are_the_closed_form():
    for distance, expected in CUTOFF_REFERENCE.items():
        assert_close(_envelope(distance), expected, f"cutoff closed form at {distance}")


def test_polynomial_cutoff_values(fp64):  # pylint: disable=unused-argument
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, p=CUTOFF_P)
    distances = torch.tensor([[r] for r in CUTOFF_REFERENCE])
    assert_close(
        cutoff(distances), [[v] for v in CUTOFF_REFERENCE.values()], "cutoff"
    )
    assert cutoff.p.dtype == torch.int32
    assert f"p={cutoff.p}" in repr(cutoff)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_polynomial_cutoff_is_exactly_zero_at_and_beyond_r_max(dtype):
    """Exact equality, in both dtypes, at both distances the padded-batch
    contract relies on. At 2*r_max the polynomial evaluates to a negative
    number and the `(x < r_max)` mask produces -0.0, which compares equal to
    0.0 -- assert the comparison, never the repr."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, p=CUTOFF_P)
        for distance in (CUTOFF_R_MAX, 2 * CUTOFF_R_MAX, 100 * CUTOFF_R_MAX):
            value = cutoff(torch.tensor(distance, dtype=dtype))
            assert value.item() == 0.0, (distance, value.item())
        # ... and it is not zero just inside
        inside = cutoff(torch.tensor(CUTOFF_R_MAX * 0.999, dtype=dtype))
        assert inside.item() > 0.0
    finally:
        torch.set_default_dtype(previous)


def test_polynomial_cutoff_derivative_vanishes_at_and_beyond_r_max(
    fp64,
):  # pylint: disable=unused-argument
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, p=CUTOFF_P)
    distances = torch.tensor(
        [CUTOFF_R_MAX, CUTOFF_R_MAX + 0.5, 2 * CUTOFF_R_MAX], requires_grad=True
    )
    (gradient,) = torch.autograd.grad(cutoff(distances).sum(), distances)
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_polynomial_cutoff_meets_r_max_with_a_vanishing_slope(
    fp64,
):  # pylint: disable=unused-argument
    """Continuity across the cutoff, measured rather than asserted: the
    envelope has a triple root at r_max, so its finite-difference slope just
    inside must fall off like h^2 as the offset h halves. A first-order kink
    would halve instead, and a discontinuity would not shrink at all."""
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, p=CUTOFF_P)

    def central_difference(distance, step=1e-7):
        left = cutoff(torch.tensor(distance - step))
        right = cutoff(torch.tensor(distance + step))
        return abs(float(right - left) / (2 * step))

    # offsets small enough to be in the asymptotic regime: at h = 0.4 (13% of
    # r_max) the higher-order terms still dominate and the observed ratio is
    # only 2.76, which is a property of the sampling, not of the envelope
    offsets = (0.05, 0.025, 0.0125, 0.00625)
    slopes = [central_difference(CUTOFF_R_MAX - h) for h in offsets]
    for coarse, fine in zip(slopes, slopes[1:]):
        ratio = coarse / fine
        assert 3.5 < ratio < 4.5, f"observed order {np.log2(ratio):.2f}, expected 2"
    # and the last one is already tiny compared with the slope mid-range
    assert slopes[-1] < 0.01 * central_difference(CUTOFF_R_MAX / 2)


@pytest.mark.parametrize("p", [2, 5, 6, 8])
def test_polynomial_cutoff_is_one_at_zero_for_every_p(p, fp64):  # pylint: disable=unused-argument
    cutoff = PolynomialCutoff(r_max=CUTOFF_R_MAX, p=p)
    assert cutoff(torch.tensor(0.0)).item() == 1.0
    assert cutoff(torch.tensor(CUTOFF_R_MAX)).item() == 0.0
    # monotone decreasing in between, for every p the CLI accepts
    sampled = cutoff(torch.linspace(0.0, CUTOFF_R_MAX, 200))
    assert bool((torch.diff(sampled) <= 0).all())


# ===========================================================================
# ZBLBasis -- the published Ziegler-Biersack-Littmark screened Coulomb pair
# ===========================================================================
#
#   V(r) = (1/2) * (14.3996 * Z_u * Z_v / r) * phi(r/a) * envelope(r)
#   a    = 0.4543 * 0.529 / (Z_u^0.3 + Z_v^0.3)
#   phi(x) = 0.1818 e^{-3.2x} + 0.5099 e^{-0.9423x}
#          + 0.2802 e^{-0.4028x} + 0.02817 e^{-0.2016x}
#
# The envelope is a PolynomialCutoff whose r_max is the *pair's* covalent
# radii sum, not the model cutoff -- so the term switches off at 1.07 Ang for
# H-C and at 1.32 Ang for O-O. The 1/2 is because the sum below runs over
# directed edges, i.e. every pair twice.


def zbl_pair_energy(z_u, z_v, distance, p=6):
    """The formula above, written out independently of the implementation."""
    a = 0.4543 * 0.529 / (z_u**0.300 + z_v**0.300)
    x = distance / a
    phi = (
        0.1818 * np.exp(-3.2 * x)
        + 0.5099 * np.exp(-0.9423 * x)
        + 0.2802 * np.exp(-0.4028 * x)
        + 0.02817 * np.exp(-0.2016 * x)
    )
    pair_r_max = ase.data.covalent_radii[z_u] + ase.data.covalent_radii[z_v]
    envelope = _envelope(distance, r_max=pair_r_max, p=p) * (distance < pair_r_max)
    return 0.5 * (14.3996 * z_u * z_v) / distance * phi * envelope


#: (Z_u, Z_v, r) -> per-node energy, hand-evaluated from the formula above.
#: H-C at 0.9 Ang: a = 0.4543*0.529/(1 + 6^0.3) = 0.09147 Ang, x = 9.839,
#: phi = 6.148e-5, prefactor 14.3996*6/0.9 = 95.997 eV, envelope 0.1300.
ZBL_REFERENCE = {
    (1, 6, 0.9): 0.04838438085989024,
    (8, 8, 1.2): 0.00924872728685459,
}


def test_zbl_reference_values_are_the_published_formula():
    for (z_u, z_v, distance), expected in ZBL_REFERENCE.items():
        assert_close(
            zbl_pair_energy(z_u, z_v, distance), expected, f"zbl {z_u}-{z_v}"
        )


@pytest.mark.parametrize("pair", sorted(ZBL_REFERENCE))
def test_zbl_matches_the_published_formula(pair, fp64):  # pylint: disable=unused-argument
    z_u, z_v, distance = pair
    zbl = ZBLBasis(p=6)
    species = sorted({z_u, z_v})
    node_attrs = torch.eye(len(species))[
        [species.index(z_u), species.index(z_v)]
    ].to(torch.get_default_dtype())
    lengths = torch.tensor([[distance], [distance]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    energies = zbl(
        lengths, node_attrs, edge_index, torch.tensor(species, dtype=torch.long)
    )
    # one directed edge lands on each node, each carrying half the pair energy
    assert energies.shape == (2,)
    assert_close(energies, [ZBL_REFERENCE[pair]] * 2, f"zbl {pair}")
    assert_close(
        energies.sum(), 2 * ZBL_REFERENCE[pair], "zbl total"
    )


def test_zbl_is_exactly_zero_beyond_the_pair_covalent_radii(
    fp64,
):  # pylint: disable=unused-argument
    """The envelope's r_max is the pair's covalent radii sum (1.07 Ang for
    H-C), not the model cutoff -- so a perfectly ordinary bond length gets
    exactly no repulsion."""
    zbl = ZBLBasis(p=6)
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    atomic_numbers = torch.tensor([1, 6])
    pair_r_max = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    assert pair_r_max == pytest.approx(1.07)
    for distance in (pair_r_max, 1.2, 3.0):
        energies = zbl(
            torch.tensor([[distance], [distance]]),
            node_attrs,
            edge_index,
            atomic_numbers,
        )
        assert torch.equal(energies, torch.zeros(2)), distance
    # ... and it is not zero just inside
    inside = zbl(
        torch.tensor([[pair_r_max * 0.99], [pair_r_max * 0.99]]),
        node_attrs,
        edge_index,
        atomic_numbers,
    )
    assert (inside > 0.0).all()


def test_zbl_energy_is_scattered_onto_the_receiver(fp64):  # pylint: disable=unused-argument
    """Per-node, not per-edge: the return is indexed by receiver, so a node
    with two neighbours carries both halves."""
    zbl = ZBLBasis(p=6)
    # three H atoms, all edges pointing at node 0
    node_attrs = torch.ones(3, 1)
    edge_index = torch.tensor([[1, 2], [0, 0]])
    lengths = torch.tensor([[0.6], [0.6]])
    energies = zbl(lengths, node_attrs, edge_index, torch.tensor([1]))
    assert energies.shape == (3,)
    assert_close(energies[0], 2 * zbl_pair_energy(1, 1, 0.6), "receiver sum")
    assert torch.equal(energies[1:], torch.zeros(2))


def test_zbl_buffers_and_trainability(fp64):  # pylint: disable=unused-argument
    zbl = ZBLBasis(p=6)
    assert_close(zbl.c, [0.1818, 0.5099, 0.2802, 0.02817], "zbl c")
    assert zbl.a_exp.item() == pytest.approx(0.300)
    assert zbl.a_prefactor.item() == pytest.approx(0.4543)
    assert not zbl.a_exp.requires_grad and not zbl.a_prefactor.requires_grad
    trainable = ZBLBasis(p=6, trainable=True)
    assert trainable.a_exp.requires_grad and trainable.a_prefactor.requires_grad


def test_zbl_accepts_and_ignores_r_max(fp64, caplog):  # pylint: disable=unused-argument
    """`r_max` is a deprecated argument kept for old checkpoints: it warns and
    changes nothing, because the envelope's cutoff comes from the covalent
    radii of each pair."""
    with caplog.at_level("WARNING"):
        deprecated = ZBLBasis(p=6, r_max=5.0)
    assert "r_max is deprecated" in caplog.text
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    lengths = torch.tensor([[0.9], [0.9]])
    atomic_numbers = torch.tensor([1, 6])
    assert torch.equal(
        deprecated(lengths, node_attrs, edge_index, atomic_numbers),
        ZBLBasis(p=6)(lengths, node_attrs, edge_index, atomic_numbers),
    )


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
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.stack(
        [
            torch.zeros(len(distances), dtype=torch.long),
            torch.ones(len(distances), dtype=torch.long),
        ]
    )
    return lengths, node_attrs, edge_index, torch.tensor([1, 6])


def test_agnesi_reference_values_are_the_closed_form():
    q, p, a = 0.9183, 4.5791, 1.0805
    r_0 = 0.5 * (ase.data.covalent_radii[1] + ase.data.covalent_radii[6])
    for distance, expected in AGNESI_REFERENCE.items():
        y = distance / r_0
        assert_close(1.0 / (1 + a * y**q / (1 + y ** (q - p))), expected, "agnesi")


def test_agnesi_transform_values(fp64):  # pylint: disable=unused-argument
    transform = AgnesiTransform()
    out = transform(*_hc_edge(AGNESI_REFERENCE))
    assert_close(out, [[v] for v in AGNESI_REFERENCE.values()], "agnesi")


def test_agnesi_transform_is_monotone_decreasing(fp64):  # pylint: disable=unused-argument
    """It compresses distance: larger r maps to smaller transformed r, over
    the whole range a cutoff can span."""
    distances = np.linspace(0.2, 6.0, 200)
    out = AgnesiTransform()(*_hc_edge(distances)).squeeze(-1)
    assert bool((torch.diff(out) < 0).all())
    assert float(out[0]) < 1.0


def test_soft_reference_values_are_the_closed_form():
    r_0 = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    p_0, p_1 = 0.75 * r_0, (4.0 / 3.0) * r_0
    midpoint, alpha = 0.5 * (p_0 + p_1), 4.0 / (p_1 - p_0)
    for distance, expected in SOFT_REFERENCE.items():
        switch = 0.5 * (1.0 + np.tanh(alpha * (distance - midpoint)))
        assert_close(p_0 + (distance - p_0) * switch, expected, "soft")


def test_soft_transform_values(fp64):  # pylint: disable=unused-argument
    out = SoftTransform()(*_hc_edge(SOFT_REFERENCE))
    assert_close(out, [[v] for v in SOFT_REFERENCE.values()], "soft")


def test_soft_transform_clamps_below_p0_and_is_the_identity_above(
    fp64,
):  # pylint: disable=unused-argument
    """It flattens short distances onto p_0 = 0.75 * r_0 and leaves long ones
    alone. Note the docstring on the class says it clamps at `p1`; the code
    clamps at `p_0`, and the code is what is pinned here."""
    transform = SoftTransform()
    r_0 = ase.data.covalent_radii[1] + ase.data.covalent_radii[6]
    p_0 = 0.75 * r_0
    short, long = 0.15, 4.0
    out = transform(*_hc_edge([short, long])).squeeze(-1)
    assert float(out[0]) == pytest.approx(p_0, abs=1e-3)
    assert float(out[1]) == pytest.approx(long, abs=1e-6)


def test_soft_transform_is_monotone_only_above_the_clamp(
    fp64,
):  # pylint: disable=unused-argument
    """Characterization of a real wrinkle: T is *not* globally monotone. Below
    p_0 the (x - p_0) factor is negative while the switch is still opening, so
    T dips about 5e-4 below p_0 near r = 0.74 Ang before recovering. Harmless
    -- nothing is that short in practice -- but a port that asserts global
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


def test_transforms_are_trainable_on_request(fp64):  # pylint: disable=unused-argument
    agnesi = AgnesiTransform(trainable=True)
    assert agnesi.a.requires_grad and agnesi.q.requires_grad and agnesi.p.requires_grad
    assert SoftTransform(trainable=True).alpha.requires_grad
    assert not SoftTransform().alpha.requires_grad


# ===========================================================================
# RadialMLP
# ===========================================================================


def test_every_module_has_a_repr_naming_its_parameters(fp64):  # pylint: disable=unused-argument
    """A printed model is how a user checks which radial setup a checkpoint
    was built with, so the reprs are part of the surface."""
    assert "num_basis=4" in repr(BesselBasis(r_max=5.0, num_basis=4))
    assert "num_basis=4" in repr(ChebychevBasis(r_max=5.0, num_basis=4))
    assert "r_max=3.0" in repr(PolynomialCutoff(r_max=3.0)).replace("tensor(", "")
    assert "0.1818" in repr(ZBLBasis(p=6))
    assert "a=1.0805" in repr(AgnesiTransform())
    assert "alpha=4.0000" in repr(SoftTransform())


def test_radial_mlp_structure_and_shapes(fp64):  # pylint: disable=unused-argument
    mlp = RadialMLP([8, 16, 4])
    kinds = [type(module).__name__ for module in mlp.net]
    # Linear -> LayerNorm -> SiLU -> Linear: no normalisation or activation
    # after the last layer, so the output is unbounded
    assert kinds == ["Linear", "LayerNorm", "SiLU", "Linear"]
    assert mlp.hs == [8, 16, 4]
    out = mlp(torch.zeros(5, 8))
    assert out.shape == (5, 4)
    assert out.dtype == torch.float64


def test_radial_mlp_single_layer_has_no_activation(fp64):  # pylint: disable=unused-argument
    mlp = RadialMLP([3, 2])
    assert [type(module).__name__ for module in mlp.net] == ["Linear"]
    assert mlp(torch.ones(1, 3)).shape == (1, 2)


# ===========================================================================
# RadialEmbeddingBlock: --apply_cutoff and the order of the three steps
# ===========================================================================
#
# forward() computes the cutoff from the RAW edge lengths, then applies the
# distance transform, then the basis. Both facts are load-bearing and neither
# is visible from the outside unless it is asserted: with `--distance_transform
# Agnesi` an envelope computed from the transformed lengths differs by 0.87 at
# r = 0.9 Ang, i.e. it is not a rounding-level difference but a different
# model.

EMBEDDING_R_MAX = 3.0


def _embedding_inputs():
    lengths = torch.tensor([[0.9], [1.7], [2.5]])
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]])
    return lengths, node_attrs, edge_index, torch.tensor([1, 6])


@pytest.mark.parametrize("radial_type", ["bessel", "gaussian", "chebyshev"])
@pytest.mark.parametrize("distance_transform", ["None", "Agnesi", "Soft"])
def test_apply_cutoff_true_returns_the_product_and_no_envelope(
    radial_type, distance_transform, fp64
):  # pylint: disable=unused-argument
    block = RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=6,
        radial_type=radial_type,
        distance_transform=distance_transform,
    )
    assert block.apply_cutoff is True  # the CLI default
    radial, envelope = block(*_embedding_inputs())
    assert envelope is None
    assert radial.shape == (3, 4)
    assert block.out_dim == 4


@pytest.mark.parametrize("radial_type", ["bessel", "gaussian", "chebyshev"])
@pytest.mark.parametrize("distance_transform", ["None", "Agnesi", "Soft"])
def test_apply_cutoff_false_defers_the_envelope_to_the_consumer(
    radial_type, distance_transform, fp64
):  # pylint: disable=unused-argument
    """`--apply_cutoff False` returns the bare basis and the envelope beside
    it, un-multiplied. The product of the two is bit-for-bit the default
    branch's output -- that identity is the whole contract, since the
    consumer applies the envelope later."""
    common = dict(
        r_max=EMBEDDING_R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=6,
        radial_type=radial_type,
        distance_transform=distance_transform,
    )
    applied, _ = RadialEmbeddingBlock(**common, apply_cutoff=True)(
        *_embedding_inputs()
    )
    deferred, envelope = RadialEmbeddingBlock(**common, apply_cutoff=False)(
        *_embedding_inputs()
    )
    assert envelope is not None
    assert envelope.shape == (3, 1)
    assert torch.equal(applied, deferred * envelope)
    assert not torch.equal(applied, deferred)  # the envelope is not all ones


@pytest.mark.parametrize("distance_transform", ["Agnesi", "Soft"])
def test_the_cutoff_is_computed_before_the_distance_transform(
    distance_transform, fp64
):  # pylint: disable=unused-argument
    """Ordering, pinned discriminatingly: the returned envelope is the one
    computed from the raw lengths, and it is *not* the one the transformed
    lengths would give."""
    lengths, node_attrs, edge_index, atomic_numbers = _embedding_inputs()
    block = RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=6,
        distance_transform=distance_transform,
        apply_cutoff=False,
    )
    _, envelope = block(lengths, node_attrs, edge_index, atomic_numbers)

    cutoff = PolynomialCutoff(r_max=EMBEDDING_R_MAX, p=6)
    transform = {"Agnesi": AgnesiTransform, "Soft": SoftTransform}[
        distance_transform
    ]()
    transformed = transform(lengths, node_attrs, edge_index, atomic_numbers)

    assert torch.equal(envelope, cutoff(lengths))
    assert not torch.allclose(envelope, cutoff(transformed))


def test_the_basis_sees_the_transformed_lengths(fp64):  # pylint: disable=unused-argument
    lengths, node_attrs, edge_index, atomic_numbers = _embedding_inputs()
    block = RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=6,
        distance_transform="Agnesi",
        apply_cutoff=False,
    )
    radial, _ = block(lengths, node_attrs, edge_index, atomic_numbers)
    transformed = AgnesiTransform()(lengths, node_attrs, edge_index, atomic_numbers)
    reference = BesselBasis(r_max=EMBEDDING_R_MAX, num_basis=4)
    assert torch.equal(radial, reference(transformed))
    assert not torch.allclose(radial, reference(lengths))


def test_a_padding_edge_contributes_exactly_nothing(fp64):  # pylint: disable=unused-argument
    """The property `build_fake_padding_graph` depends on: a self-loop edge
    with a shift of 2*r_max has length 2*r_max, and the embedded value there
    is exactly zero -- in the default branch by multiplication, and in the
    deferred branch through an envelope that is exactly zero."""
    block = RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX, num_bessel=4, num_polynomial_cutoff=6
    )
    padded = torch.tensor([[2 * EMBEDDING_R_MAX]])
    node_attrs = torch.tensor([[1.0, 0.0]])
    edge_index = torch.tensor([[0], [0]])
    atomic_numbers = torch.tensor([1, 6])
    radial, _ = block(padded, node_attrs, edge_index, atomic_numbers)
    assert torch.equal(radial, torch.zeros_like(radial))

    deferred = RadialEmbeddingBlock(
        r_max=EMBEDDING_R_MAX,
        num_bessel=4,
        num_polynomial_cutoff=6,
        apply_cutoff=False,
    )
    _, envelope = deferred(padded, node_attrs, edge_index, atomic_numbers)
    assert torch.equal(envelope, torch.zeros_like(envelope))


if __name__ == "__main__":
    pytest.main([__file__])
