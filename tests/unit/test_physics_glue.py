"""The differentiable physics glue, characterized against finite differences.

**These docstrings are normative.** CORE-3 copies the three sign conventions
below verbatim into the rewrite's units module, and every one of them is
asserted by a test in this file rather than assumed:

* **forces** = -dE/d(positions), in eV/Ang.
* **stress** = (1/V) dE/d(strain), in eV/Ang^3, with V = |det(cell)|.
* **virials** = -stress * V = -dE/d(strain), in eV.

All three come out of one function, `compute_forces_virials`
(mace/modules/utils.py:52-80): it takes the raw autograd gradients of the
energy with respect to the positions and to the injected strain, divides the
strain gradient by |det(cell)| to form the stress, and negates *both* raw
gradients on the way out. So the returned stress is the only one of the three
that is **not** negated, and the virial is the negative of the same quantity
the stress is built from. That asymmetry is the single most likely thing for
a port to normalise away, which is why the relation is pinned twice here --
once against a finite-difference derivative and once against a synthetic
energy whose gradient is known exactly and by hand.

Everything runs on the committed fp64 anchors on CPU, with no capability
marker: this is the specification the new derivative code is written against,
so it may not be a test that can skip.

Tolerances come from the one table in `tests/golden/harness.py`. The
finite-difference comparisons use the fp64 CPU row (1e-6 absolute), and they
clear it by four to five orders of magnitude -- measured on this tree, the
worst deviation is 2.2e-11 eV/Ang for the forces and 2.5e-13 eV/Ang^3 for the
stress. The budget at h = 1e-4 Ang is dominated by cancellation, not by the
5-point stencil's O(h^4) truncation: roughly eps * |E| / h ~ 1e-16 * 40 / 1e-4
= 4e-11, which is what is actually observed.
"""

import numpy as np
import pytest
import torch

from mace.modules.utils import (
    compute_forces,
    compute_forces_virials,
    get_edge_vectors_and_lengths,
    get_symmetric_displacement,
    prepare_graph,
)
from mace.tools import torch_tools
from tests.golden import harness
from tests.golden.anchors import ANCHORS, anchor_graph, load_anchor

TOL = harness.FP64_CPU_REFERENCE

#: 5-point central difference step. Large enough that cancellation at fp64
#: stays ~1e-11 eV/Ang, small enough that the O(h^4) truncation term is far
#: below that.
STEP = 1e-4

#: strain increment for the stress differences, in dimensionless strain.
STRAIN_STEP = 1e-5


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures()


def _five_point(values, step):
    """(f(-2h), f(-h), f(+h), f(+2h)) -> df/dx, error O(h^4)."""
    minus2, minus1, plus1, plus2 = values
    return (minus2 - 8.0 * minus1 + 8.0 * plus1 - plus2) / (12.0 * step)


def _energy(model, atoms):
    return float(
        model(anchor_graph(model, atoms), compute_force=False)["energy"].detach()
    )


# ---------------------------------------------------------------------------
# forces = -dE/dx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
def test_forces_are_minus_the_energy_gradient(anchor, fixtures):
    """Five-point central differences on every degree of freedom.

    The cluster fixture is used rather than a periodic one so that only the
    position derivative is in play; the strain derivative gets its own test.
    """
    atoms = fixtures["water_cluster"]
    model = load_anchor(anchor)
    with torch_tools.default_dtype("float64"):
        forces = (
            model(anchor_graph(model, atoms), compute_force=True)["forces"]
            .detach()
            .numpy()
        )
        reference_positions = atoms.get_positions().copy()
        gradient = np.zeros_like(reference_positions)
        for atom in range(len(atoms)):
            for axis in range(3):
                energies = []
                for multiple in (-2, -1, 1, 2):
                    moved = atoms.copy()
                    positions = reference_positions.copy()
                    positions[atom, axis] += multiple * STEP
                    moved.set_positions(positions)
                    energies.append(_energy(model, moved))
                gradient[atom, axis] = _five_point(energies, STEP)

    assert np.abs(forces).max() > 1e-3, "a vanishing force makes this vacuous"
    deviation = np.abs(forces - (-gradient)).max()
    assert deviation < TOL.atol, f"max |F + dE/dx| = {deviation:.3e}"
    # and the sign is asserted, not assumed: the other choice is wrong by
    # twice the force, which is orders of magnitude outside the row.
    assert np.abs(forces - gradient).max() > TOL.atol


def test_the_reported_force_is_the_gradient_of_the_energy_the_model_returns(fixtures):
    """`d(total_energy)/dx` is bit-for-bit the force, in both classes.

    ScaleShiftMACE differentiates the *interaction* energy
    (mace/modules/models.py:585) while plain MACE differentiates the total
    (`:403`), and the two are only interchangeable because the E0 branch has
    no autograd path to the positions at all. That is a property of the
    model, not an identity, so it is measured: if a future E0 term ever
    became position dependent, the ScaleShiftMACE forces would silently stop
    being the gradient of the energy it reports.

    **Bit-for-bit is a claim about the seed.** Differentiating `energy.sum()`
    seeds the shared backward with a stride-0 `expand` of a scalar instead of
    the materialized `grad_outputs=ones` that `compute_forces` passes
    (mace/modules/utils.py:36-46), and the broadcast path that takes lands on
    different last bits: measured on Viper-CPU (EPYC Genoa, torch 2.13+rocm7.1,
    MKL) the two disagree by 7.3e-17 for ScaleShiftMACE and 5.6e-17 for MACE,
    about a third of an ulp, while on macOS/Accelerate both are exactly zero.
    Matching the seed and `create_graph` makes this the same computation run
    twice rather than two computations of one quantity, which is the only kind
    of comparison bit-exactness belongs in; both platforms then give zero.

    The deviations are collected over the anchors and asserted once, because
    an assertion inside the loop stops at the first: when only ScaleShiftMACE
    was reported, plain MACE was not passing, it was unreached.
    """
    atoms = fixtures["water_cluster"]
    disagreed = []
    for anchor in ANCHORS:
        model = load_anchor(anchor)
        with torch_tools.default_dtype("float64"):
            graph = anchor_graph(model, atoms)
            out = model(graph, training=True, compute_force=True)
            gradient = torch.autograd.grad(
                outputs=[out["energy"]],
                inputs=[graph["positions"]],
                grad_outputs=[torch.ones_like(out["energy"])],
                retain_graph=True,
                create_graph=True,
            )[0]
        if not torch.equal(out["forces"], -gradient):
            deviation = float((out["forces"] + gradient).abs().max())
            disagreed.append(f"{anchor}: max |F + dE/dx| = {deviation:.3e}")
    assert not disagreed, "; ".join(disagreed)


def test_a_structure_with_no_edges_gets_zero_forces(fixtures):
    model = load_anchor("tiny_scaleshift")
    with torch_tools.default_dtype("float64"):
        graph = anchor_graph(model, fixtures["isolated_atom"])
        assert graph["edge_index"].shape[1] == 0
        forces = model(graph, compute_force=True)["forces"]
    assert torch.equal(forces, torch.zeros_like(forces))


def test_compute_forces_returns_zeros_when_the_energy_is_unconnected():
    """The `allow_unused` branch, which no fixture reaches.

    A single-atom structure still returns a *computed* zero (autograd walks
    an empty edge set and produces 0.0, which the trailing `-1 *` turns into
    -0.0). The None branch is only reached when the energy shares no graph
    with the positions at all, and it returns a fresh +0.0 -- so the two are
    distinguishable, and only this one is a stand-in for "there was nothing
    to differentiate".
    """
    positions = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    unrelated = torch.ones(1, dtype=torch.float64, requires_grad=True) * 2.0
    forces = compute_forces(unrelated, positions, training=False)
    assert forces.shape == positions.shape
    assert torch.equal(forces, torch.zeros_like(forces))
    assert not torch.signbit(forces).any(), "the None branch returns +0.0"


# ---------------------------------------------------------------------------
# stress = (1/V) dE/d(strain), virials = -stress * V
# ---------------------------------------------------------------------------


def _strained_energy(model, atoms, strain):
    """Energy of `atoms` under the linear deformation (I + strain).

    Both the cell and the positions are deformed, which is what the model's
    own strain injection does: `get_symmetric_displacement` adds
    `positions @ symmetric_displacement` to the positions and
    `cell @ symmetric_displacement` to the cell.
    """
    deformed = atoms.copy()
    deformation = np.eye(3) + strain
    deformed.set_cell(np.array(atoms.get_cell()) @ deformation.T, scale_atoms=False)
    deformed.set_positions(atoms.get_positions() @ deformation.T)
    return _energy(model, deformed)


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
def test_stress_is_the_strain_derivative_over_the_volume(anchor, fixtures):
    """All six independent components, by finite strains of +-1e-5.

    The strain applied for component (i, j) is symmetric by construction, so
    what is differentiated is exactly the symmetrised strain the model
    injects -- an antisymmetric perturbation is a rotation and must not
    change the energy at all (asserted separately below).
    """
    atoms = fixtures["triclinic_bulk"]
    model = load_anchor(anchor)
    volume = float(abs(np.linalg.det(np.array(atoms.get_cell()))))
    with torch_tools.default_dtype("float64"):
        stress = (
            model(anchor_graph(model, atoms), compute_stress=True)["stress"]
            .detach()
            .numpy()[0]
        )
        finite_difference = np.zeros((3, 3))
        for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
            direction = np.zeros((3, 3))
            if i == j:
                direction[i, i] = 1.0
            else:
                direction[i, j] = direction[j, i] = 0.5
            energies = [
                _strained_energy(model, atoms, multiple * STRAIN_STEP * direction)
                for multiple in (-2, -1, 1, 2)
            ]
            value = _five_point(energies, STRAIN_STEP) / volume
            finite_difference[i, j] = finite_difference[j, i] = value

    assert np.abs(stress).max() > 1e-5, "a vanishing stress makes this vacuous"
    deviation = np.abs(stress - finite_difference).max()
    assert deviation < TOL.atol, f"max |stress - dE/de / V| = {deviation:.3e}"
    # the opposite sign convention is wrong by twice the stress
    assert np.abs(stress + finite_difference).max() > TOL.atol


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
def test_virials_are_minus_the_stress_times_the_volume(anchor, fixtures):
    """The relation both come out of, asserted rather than assumed."""
    atoms = fixtures["triclinic_bulk"]
    model = load_anchor(anchor)
    volume = float(abs(np.linalg.det(np.array(atoms.get_cell()))))
    with torch_tools.default_dtype("float64"):
        out = model(anchor_graph(model, atoms), compute_stress=True, compute_virials=True)
    stress = out["stress"].detach().numpy()[0]
    virials = out["virials"].detach().numpy()[0]
    assert np.abs(virials).max() > 1e-3, "a vanishing virial makes this vacuous"
    assert np.abs(virials + stress * volume).max() < TOL.atol


def test_the_sign_of_every_derivative_against_a_hand_differentiable_energy():
    """The conventions with no model in the way.

    E = a . x  (summed over atoms) + c * tr(displacement) has
    dE/dx = a and dE/d(eps) = c * I by inspection, so the three returned
    quantities can be read off:

        forces  = -a
        stress  = +c * I / V
        virials = -c * I

    A model-based test can only ever compare two computed things; this one
    compares against arithmetic.
    """
    a = torch.tensor([[2.0, -3.0, 0.5]], dtype=torch.float64)
    c = 7.0
    volume = 8.0
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)

    def evaluate(cell):
        positions = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        displacement = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
        energy = (a * positions).sum() + c * torch.einsum(
            "bii->b", displacement
        ).sum()
        return compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=False,
            compute_stress=True,
        )

    cell = identity * 2.0  # det = 8
    forces, virials, stress = evaluate(cell)
    assert torch.allclose(forces, -a, atol=TOL.atol, rtol=TOL.rtol)
    assert torch.allclose(virials, -c * identity, atol=TOL.atol, rtol=TOL.rtol)
    expected_stress = c * identity / volume
    assert torch.allclose(stress, expected_stress, atol=TOL.atol, rtol=TOL.rtol)
    # the volume is |det(cell)|, so a left-handed cell gives the same stress
    _, _, mirrored = evaluate(-cell)
    assert torch.allclose(mirrored, expected_stress, atol=TOL.atol, rtol=TOL.rtol)


def test_an_unconnected_energy_gives_zero_forces_and_a_zero_virial():
    """The two `is None` guards in `compute_forces_virials` (`:75-78`).

    They are what a fully dissociated structure hits. Note the shapes they
    fall back to: the forces match the positions, but the virial is
    `torch.zeros((1, 3, 3))` regardless of how many graphs were in the batch
    -- and in the default dtype rather than the energy's. A batch of four
    structures that reaches this branch gets one virial back.
    """
    positions = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    displacement = torch.zeros((3, 3, 3), dtype=torch.float64, requires_grad=True)
    unrelated = torch.ones(1, dtype=torch.float64, requires_grad=True) * 5.0
    cell = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)

    forces, virials, stress = compute_forces_virials(
        energy=unrelated,
        positions=positions,
        displacement=displacement,
        cell=cell,
        training=False,
        compute_stress=True,
    )
    assert torch.equal(forces, torch.zeros_like(positions))
    assert virials.shape == (1, 3, 3)
    assert torch.count_nonzero(virials) == 0
    # the stress keeps the displacement's shape, because it was seeded from it
    assert stress.shape == (3, 3, 3)


def test_get_symmetric_displacement_invents_a_cell_when_there_is_none():
    """`cell=None` becomes a `(3 * n_graphs, 3)` block of zeros (`:92-98`).

    A zero cell makes every shift zero, so an aperiodic batch that reaches
    this path gets the right edge vectors and a meaningless volume -- which
    is why the neighbour list hands back a fabricated cell for aperiodic
    structures rather than letting this one be used for a stress.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    unit_shifts = torch.zeros((1, 3), dtype=torch.float64)
    edge_index = torch.tensor([[0], [1]])
    moved, shifts, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=None,
        edge_index=edge_index,
        num_graphs=2,
        batch=torch.tensor([0, 1]),
    )
    assert torch.equal(shifts, torch.zeros((1, 3), dtype=torch.float64))
    assert displacement.shape == (2, 3, 3)
    assert displacement.requires_grad
    assert torch.allclose(moved, positions, atol=TOL.atol, rtol=TOL.rtol)


def test_a_blown_up_stress_component_is_silently_zeroed():
    """`torch.where(|stress| < 1e10, stress, 0)` (mace/modules/utils.py:74).

    A stress that overflows the threshold is replaced by zero, not clipped
    and not propagated, and nothing is logged. Characterization, emphatically
    not endorsement: downstream this reads as "this structure has no stress".
    The trigger in practice is a near-degenerate cell, so the test uses one.
    """
    def evaluate(cell):
        displacement = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
        positions = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
        energy = torch.einsum("bii->b", displacement) * 1.0 + positions.sum()
        return compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=False,
            compute_stress=True,
        )

    tiny = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    tiny[0, 2, 2] = 1e-11  # volume 1e-11 -> stress 1e11, over the threshold
    _, virials, stress = evaluate(tiny)
    assert torch.equal(
        torch.diagonal(stress, dim1=-2, dim2=-1),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    # the virial is *not* clamped: only the stress passes through the where.
    assert torch.allclose(
        torch.diagonal(virials, dim1=-2, dim2=-1),
        -torch.ones(1, 3, dtype=torch.float64),
        atol=TOL.atol,
        rtol=TOL.rtol,
    )

    # just under the threshold the same component survives untouched
    survivable = tiny.clone()
    survivable[0, 2, 2] = 1e-9  # volume 1e-9 -> stress 1e9
    _, _, kept = evaluate(survivable)
    assert float(kept[0, 0, 0]) == pytest.approx(1e9)


# ---------------------------------------------------------------------------
# prepare_graph: the strain handle, and what it mutates
# ---------------------------------------------------------------------------


def _graph_for(fixture_name, fixtures, model):
    with torch_tools.default_dtype("float64"):
        return anchor_graph(model, fixtures[fixture_name])


def test_prepare_graph_injects_a_differentiable_strain_handle(fixtures):
    """No `displacement`, no stress: this is where the strain enters at all.

    `get_symmetric_displacement` builds a zero 3x3 per graph and ties it to
    the graph with `displacement + positions.sum() * 0.0`, so it is a
    non-leaf tensor that carries a gradient path while contributing exactly
    nothing to the energy. Both halves matter: without the path there is no
    stress, and with any contribution the energy would be wrong.
    """
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    context = prepare_graph(graph, compute_stress=True)
    assert context.displacement is not None
    assert context.displacement.shape == (1, 3, 3)
    assert context.displacement.requires_grad
    assert torch.count_nonzero(context.displacement) == 0


def test_prepare_graph_without_stress_leaves_an_inert_displacement(fixtures):
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    context = prepare_graph(graph)
    assert context.displacement is not None
    assert not context.displacement.requires_grad


def test_prepare_graph_rewrites_the_callers_positions_and_shifts(fixtures):
    """A mutation the port has to keep: `data` is written back into.

    `prepare_graph` sets `requires_grad_` on the caller's positions tensor
    and, in the stress branch, replaces `data["positions"]` and
    `data["shifts"]` with the strained ones (mace/modules/utils.py:783).
    The edge vectors are then built from the *replacement*, so a port that
    treats the input dict as read-only computes the unstrained vectors and
    gets a zero stress with no error anywhere.
    """
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    positions_before = graph["positions"].detach().clone()
    shifts_before = graph["shifts"].detach().clone()
    assert not graph["positions"].requires_grad

    context = prepare_graph(graph, compute_stress=True)

    assert graph["positions"].requires_grad
    assert graph["positions"] is not positions_before
    # at zero strain the replacement is numerically the same structure ...
    assert torch.allclose(
        graph["positions"], positions_before, atol=TOL.atol, rtol=TOL.rtol
    )
    assert torch.allclose(graph["shifts"], shifts_before, atol=TOL.atol, rtol=TOL.rtol)
    # ... but it is a different tensor, and it is what the vectors came from
    vectors, _ = get_edge_vectors_and_lengths(
        positions=graph["positions"],
        edge_index=graph["edge_index"],
        shifts=graph["shifts"],
    )
    assert torch.equal(context.vectors, vectors)


def test_prepare_graph_leaves_the_shifts_alone_without_a_strain(fixtures):
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    shifts_before = graph["shifts"]
    prepare_graph(graph)
    assert graph["shifts"] is shifts_before


def test_an_antisymmetric_strain_is_a_rotation_and_changes_no_energy(fixtures):
    """Why the displacement is symmetrised before it is applied.

    `get_symmetric_displacement` uses 0.5 * (d + d^T). Feeding it a purely
    antisymmetric handle therefore applies nothing at all, which is the
    infinitesimal statement that the energy is rotation invariant.
    """
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    with torch_tools.default_dtype("float64"):
        plain = float(model(dict(graph), compute_force=False)["energy"].detach())
        antisymmetric = torch.zeros((1, 3, 3), dtype=torch.float64)
        antisymmetric[0, 0, 1] = 1e-3
        antisymmetric[0, 1, 0] = -1e-3
        antisymmetric.requires_grad_(True)
        rotated_graph = dict(graph)
        rotated_graph["displacement"] = antisymmetric
        rotated = float(
            model(rotated_graph, compute_force=False, compute_stress=True)[
                "energy"
            ].detach()
        )
    assert rotated == pytest.approx(plain, abs=TOL.atol)


# ---------------------------------------------------------------------------
# get_edge_vectors_and_lengths
# ---------------------------------------------------------------------------


def test_edge_vectors_are_the_periodic_images_they_claim_to_be(fixtures):
    """Every edge vector reproduced by hand from the cell and the unit shift.

    This is the assembly step -- `positions[receiver] - positions[sender] +
    shifts` -- checked against an independent construction from the integer
    unit shifts, together with the two properties that make the edge set a
    neighbour list at all: nothing longer than the cutoff, and every pair of
    atoms within the cutoff present exactly once.

    (The neighbour list *itself* has a dedicated brute-force oracle in P0-7,
    which is the general tool; what is pinned here is the vector arithmetic
    the physics glue does on top of it.)
    """
    model = load_anchor("tiny_scaleshift")
    atoms = fixtures["triclinic_bulk"]
    cutoff = float(model.r_max)
    graph = _graph_for("triclinic_bulk", fixtures, model)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=graph["positions"],
        edge_index=graph["edge_index"],
        shifts=graph["shifts"],
    )
    cell = graph["cell"].view(3, 3).numpy()
    positions = graph["positions"].detach().numpy()
    unit_shifts = graph["unit_shifts"].numpy()
    senders, receivers = graph["edge_index"].numpy()

    by_hand = (
        positions[receivers] - positions[senders] + unit_shifts @ cell
    )
    assert np.abs(vectors.detach().numpy() - by_hand).max() < TOL.atol
    assert np.abs(
        lengths.detach().numpy().ravel() - np.linalg.norm(by_hand, axis=1)
    ).max() < TOL.atol
    assert lengths.max() < cutoff

    # every image within the cutoff is present, exactly once
    found = {
        (int(s), int(r), *(int(n) for n in shift))
        for s, r, shift in zip(senders, receivers, unit_shifts)
    }
    assert len(found) == len(senders), "the edge list repeats an image"
    expected = set()
    reach = 2
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            for n0 in range(-reach, reach + 1):
                for n1 in range(-reach, reach + 1):
                    for n2 in range(-reach, reach + 1):
                        shift = np.array([n0, n1, n2]) @ cell
                        delta = positions[j] - positions[i] + shift
                        distance = float(np.linalg.norm(delta))
                        if 0.0 < distance < cutoff:
                            expected.add((i, j, n0, n1, n2))
    assert found == expected


def test_normalising_the_edge_vectors_divides_by_the_length_plus_eps():
    positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0], [1]])
    shifts = torch.zeros((1, 3), dtype=torch.float64)
    eps = 1e-9
    vectors, lengths = get_edge_vectors_and_lengths(
        positions, edge_index, shifts, normalize=True, eps=eps
    )
    assert float(lengths[0, 0]) == pytest.approx(5.0)
    # the eps is inside the denominator, so the normed vector is slightly
    # short -- by 2e-10 here, which no unit vector consumer notices and which
    # a port that divides by the bare length will not reproduce bit for bit.
    assert torch.allclose(
        vectors,
        torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float64) / (5.0 + eps),
        atol=TOL.atol,
        rtol=TOL.rtol,
    )
    assert float(torch.linalg.norm(vectors)) < 1.0


# ---------------------------------------------------------------------------
# get_outputs: which quantities exist under which flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,present",
    [
        ({}, {"forces"}),
        ({"compute_force": False}, set()),
        ({"compute_stress": True}, {"forces", "stress", "virials"}),
        ({"compute_virials": True}, {"forces", "stress", "virials"}),
        ({"compute_edge_forces": True}, {"forces", "edge_forces"}),
        ({"compute_hessian": True}, {"forces", "hessian"}),
    ],
)
def test_which_derivatives_a_flag_combination_produces(flags, present, fixtures):
    """The dispatcher's contract, as a table.

    Two entries are not what the flag names suggest, and both are load
    bearing: asking for a stress also returns a virial (they come out of one
    autograd call), and `compute_virials` alone hands back a stress too --
    `compute_stress` gates only the division by the volume, and the returned
    `stress` is a zeros tensor of the displacement's shape when it is off.
    """
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    with torch_tools.default_dtype("float64"):
        out = model(graph, **flags)
    for name in ("forces", "virials", "stress", "hessian", "edge_forces"):
        if name in present:
            assert out[name] is not None, f"{name} missing with {flags}"
        else:
            assert out[name] is None, f"{name} unexpectedly present with {flags}"


def test_asking_for_a_virial_without_a_stress_still_divides_by_nothing(fixtures):
    """`compute_virials=True, compute_stress=False` returns a zero stress.

    Not `None`, and not the real stress: `compute_forces_virials` seeds
    `stress = torch.zeros_like(displacement)` and only overwrites it inside
    `if compute_stress`. A consumer that trusts a non-None stress gets zeros.
    """
    model = load_anchor("tiny_scaleshift")
    graph = _graph_for("triclinic_bulk", fixtures, model)
    with torch_tools.default_dtype("float64"):
        out = model(graph, compute_virials=True, compute_stress=False)
    assert out["virials"] is not None
    assert torch.count_nonzero(out["virials"]) > 0
    assert torch.count_nonzero(out["stress"]) == 0
