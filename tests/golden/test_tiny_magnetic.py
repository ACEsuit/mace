"""The tiny magnetic anchor reproduces its committed reference.

This is the numerical reference the magnetic family did not have. The
behavioural suite next door (``tests/extensions/magnetic``, 39 tests) covers
rotation equivariance, inversion parity, parameter registration, calculator
backends and dtype scoping -- properties, all of them, and a property is not
a number a rewrite has to reproduce. What is pinned here is the number:
energy, per-atom energies, forces and ``magforces`` (``dE/dm``) on five
moment-carrying structures, with the moments recorded as inputs and compared
exactly.

Marked ``magnetic`` explicitly. ``tests/conftest.py`` derives that marker from
the directory for everything under ``tests/extensions/magnetic``; a file in
``tests/golden`` gets no directory-derived marker, so without the decorator
these would fail on the import of a checkpoint that cannot be unpickled
without ``sphericart`` rather than skipping. In the ci-extensions ``magnetic``
job they must not skip either, which is what ``require-caps: magnetic`` on
that job is for.
"""

import json

import numpy as np
import pytest
import torch

from mace.tools import torch_tools
from tests.golden import harness
from tests.golden import magnetic_surfaces as ms

pytestmark = pytest.mark.magnetic

REFERENCE_PATH = harness.REFERENCES_DIR / "tiny_magnetic_e3nn_cpu_fp64.json"
SIDECAR_PATH = harness.MODELS_DIR / "tiny_magnetic.build.json"

TOL = harness.FP64_CPU_REFERENCE


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return ms.magnetic_fixtures()


@pytest.fixture(name="anchor", scope="module")
def fixture_anchor():
    return ms.load_anchor()


@pytest.fixture(name="reference", scope="module")
def fixture_reference():
    return harness.load_reference(REFERENCE_PATH)


# ---------------------------------------------------------------------------
# The reference itself
# ---------------------------------------------------------------------------


def test_anchor_reproduces_its_reference(fixtures, anchor, reference):
    snapshot = harness.snapshot_outputs(
        ms.MagneticForward(anchor),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(snapshot, reference, row=TOL.name)


def test_reference_carries_dtype_units_and_provenance(reference):
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["units"]["energy"] == "eV"
    provenance = reference["provenance"]
    assert provenance["source"].endswith("tiny_magnetic.model")
    assert provenance["recipe"] == "tests/golden/build_magnetic_anchor.py"
    assert provenance["tolerance_row"] == TOL.name
    for entry in reference["fixtures"].values():
        for channel in entry["outputs"].values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS


def test_sidecar_records_how_the_anchor_was_built():
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    assert sidecar["model"] == "tiny_magnetic.model"
    assert sidecar["class"] == "MagneticScaleShiftMACE"
    assert sidecar["dtype"] == "float64"
    assert sidecar["seed"]
    assert sidecar["command"]
    assert "regenerate.py --target magnetic" in sidecar["regenerate_with"]
    assert sidecar["config"]["use_magmom_one_body"] is True
    assert sidecar["config"]["use_reduced_cg"] is False


def test_anchor_checkpoint_stays_small():
    """A committed checkpoint is a permanent cost in every clone."""
    size_mb = (harness.MODELS_DIR / "tiny_magnetic.model").stat().st_size / 1e6
    assert size_mb < 1.5, f"tiny_magnetic.model is {size_mb:.2f} MB"


# ---------------------------------------------------------------------------
# The moment is an input, and the reference is only meaningful at it
# ---------------------------------------------------------------------------


def test_every_fixture_records_its_moments_as_a_pinned_input(reference, fixtures):
    """``magforces`` is a derivative with respect to something.

    A reference that recorded ``dE/dm`` and not ``m`` would be a number nobody
    can reproduce, and -- worse -- two snapshots taken at different moments
    would compare clean. The harness compares the inputs block in both
    directions and at the exact row; this asserts the block is actually there
    and holds the fixture's own array, so the guarantee is not vacuous.
    """
    for name, entry in reference["fixtures"].items():
        recorded = entry["inputs"]["magmom"]
        assert recorded["unit"] == "muB"
        assert recorded["kind"] == harness.PER_ATOM_VECTOR
        on_file = np.asarray(fixtures[name].arrays[ms.MAGMOM_KEY], dtype=float)
        assert np.array_equal(np.asarray(recorded["value"]), on_file), name


def test_a_perturbed_moment_fails_the_comparison(fixtures, anchor, reference):
    """The input block earning its keep.

    2e-3 muB on a 3 muB moment is a different magnetic state and a different
    energy, and the point of comparing the inputs exactly rather than at the
    output row is that this fails on the *input* -- immediately, naming the
    channel -- instead of being argued about as drift. At the fp32 row the
    same nudge sits comfortably inside the output bound.
    """
    perturbed = {name: atoms.copy() for name, atoms in fixtures.items()}
    target = perturbed["mag_fe_dimer_fm"]
    moments = np.asarray(target.arrays[ms.MAGMOM_KEY], dtype=float).copy()
    moments[0, 2] += 2e-3
    target.arrays[ms.MAGMOM_KEY] = moments
    snapshot = harness.snapshot_outputs(ms.MagneticForward(anchor), perturbed)
    with pytest.raises(AssertionError, match="inputs/magmom"):
        harness.compare_to_reference(snapshot, reference, row=TOL.name)


def test_ase_initial_moments_are_refused_rather_than_silently_ignored(
    fixtures, anchor
):
    """The models read an array; ase's attribute is not it.

    A structure prepared with ``set_initial_magnetic_moments`` and handed to a
    magnetic model runs on whatever is in ``REF_magmom`` and ignores the
    attribute entirely. The harness refuses that structure instead of
    recording moments the evaluation never saw.
    """
    probe = {"mag_fe_dimer_fm": fixtures["mag_fe_dimer_fm"].copy()}
    probe["mag_fe_dimer_fm"].set_initial_magnetic_moments([1.0, -1.0])
    del probe["mag_fe_dimer_fm"].arrays[ms.MAGMOM_KEY]
    with pytest.raises(ValueError, match="ase initial magnetic moments"):
        harness.snapshot_outputs(ms.MagneticForward(anchor), probe)


def test_dedm_is_well_defined_at_a_zero_moment(anchor, fixtures):
    """The obvious worry about ``dE/dm``, measured instead of assumed.

    The forward takes ``torch.norm(magmom)`` before anything else
    (mace/modules/extensions.py:1813), and the gradient of a 2-norm at the
    origin is 0/0 -- so a nonmagnetic site looks like a row of nans waiting to
    happen, and nans are the worst possible content for a golden because every
    comparison against them is false and none of them is reported as a
    difference.

    It is fine, and for a reason worth recording rather than rediscovering:
    the norm enters only *squared* (``1 - 2 * clamp(|m| / m_max)**2``,
    :1818-1824), which is smooth at the origin, and torch's convention of
    returning a zero gradient for the norm there is exactly the correct value
    for the square. So the derivative is a real derivative and not a
    subgradient: it agrees with a central difference of the energy.

    Pinned because a rewrite that reaches for ``|m|`` un-squared -- an
    ordinary-looking way to write the same basis -- introduces a kink at the
    origin and this stops holding, quietly, on exactly the structures a
    magnetic model meets most often.
    """
    probe = fixtures["mag_fe_dimer_fm"].copy()
    moments = np.asarray(probe.arrays[ms.MAGMOM_KEY], dtype=float).copy()
    moments[1] = 0.0
    probe.arrays[ms.MAGMOM_KEY] = moments
    with torch_tools.default_dtype("float64"):
        out = anchor(
            ms.build_batch(anchor, probe), compute_force=True, compute_magforces=True
        )
    magforces = out["magforces"].detach().numpy()
    assert np.isfinite(magforces).all()

    def energy(m_z):
        shifted = probe.copy()
        nudged = moments.copy()
        nudged[1, 2] = m_z
        shifted.arrays[ms.MAGMOM_KEY] = nudged
        with torch_tools.default_dtype("float64"):
            return float(
                anchor(ms.build_batch(anchor, shifted), compute_force=False)["energy"][
                    0
                ]
            )

    step = 1e-5
    central = (energy(step) - energy(-step)) / (2.0 * step)
    # magforces is -dE/dm (mace/modules/utils.py:229-263)
    assert -float(magforces[1, 2]) == pytest.approx(central, rel=1e-6)


# ---------------------------------------------------------------------------
# What the reference is a reference *of*: the magnetic physics
# ---------------------------------------------------------------------------


def test_the_two_dimers_differ_only_in_their_spin_state(reference):
    """The fixture pair that makes the family falsifiable.

    ``mag_fe_dimer_fm`` and ``mag_fe_dimer_afm`` are the same two iron atoms
    in the same two places with the same moment magnitude; the second one's
    is reversed. Every difference between their two reference entries is
    therefore the spin state and nothing else, and a rewrite that dropped the
    moment from the message passing -- or read it from the wrong array, or
    lost its sign -- would make the two produce one number while every
    non-magnetic golden still passed.
    """
    fm = reference["fixtures"]["mag_fe_dimer_fm"]
    afm = reference["fixtures"]["mag_fe_dimer_afm"]
    fm_m = np.asarray(fm["inputs"]["magmom"]["value"])
    afm_m = np.asarray(afm["inputs"]["magmom"]["value"])
    assert np.array_equal(np.abs(fm_m), np.abs(afm_m))
    assert np.array_equal(fm_m[1], -afm_m[1])

    splitting = fm["outputs"]["energy"]["value"] - afm["outputs"]["energy"]["value"]
    # Measured on the committed anchor: 9.53e-2 eV, which is five orders of
    # magnitude above the row these references are asserted at. The bound is
    # written against that row rather than as a number of its own, so it says
    # "the splitting is real" and not "the splitting is this large".
    assert abs(splitting) > 1e3 * TOL.atol, (
        f"the FM and AFM dimers differ by only {splitting:.3e} eV; either the "
        f"moments stopped reaching the energy or the anchor became blind to "
        f"their relative orientation"
    )


def test_the_ferromagnetic_dimer_has_inequivalent_sites(reference):
    """The moment is treated as a polar vector, and that is visible here.

    Two chemically identical atoms, identical moments, and a geometry that
    maps onto itself under inversion through the midpoint. If the moment were
    an axial vector -- which is what a magnetic moment is -- inversion would
    leave it alone, the two sites would be equivalent, and ``dE/dm`` would be
    the same on both. It is not: the moment enters through solid harmonics of
    ``m`` itself, so terms odd in ``m . r`` survive and the two sites differ.

    The behavioural suite states the same thing from the other side --
    ``test_magnetic_mace_inversion_parity`` flips positions *and* moments
    together -- and this is its numerical consequence on a committed
    structure. It is pinned so that a rewrite choosing the axial convention
    fails loudly here rather than producing plausible numbers that mean
    something else.
    """
    entry = reference["fixtures"]["mag_fe_dimer_fm"]
    magforces = np.asarray(entry["outputs"]["magforces"]["value"])
    difference = np.abs(magforces[0] - magforces[1]).max()
    assert difference > 1e-3, (
        f"the two sites of the ferromagnetic dimer now agree to {difference:.3e} "
        f"eV/muB. If the moment has become an axial vector this is expected and "
        f"the whole reference has to be regenerated; if it has not, something "
        f"symmetrised that should not have."
    )


def test_the_one_body_magnetic_term_is_inside_the_reference(anchor, fixtures):
    """``--use_magmom_one_body``, pinned where it is the only thing left.

    The one-body term is a per-atom energy depending on |m| alone, through a
    Chebyshev basis and a per-species constant correction
    (mace/modules/extensions.py:1719-1737, applied at :1866-1888). On the
    isolated-atom fixture there are no edges, so the interaction and product
    blocks contribute nothing and this term is the entire magnetic content of
    the energy: zeroing its coefficients has to move that fixture's energy,
    and if it does not, the switch is not reaching the forward.
    """
    import copy  # noqa: PLC0415

    stripped = copy.deepcopy(anchor)
    with torch.no_grad():
        stripped.onebody_magmombasis_coeffs.zero_()

    atoms = fixtures["mag_fe_atom"]
    with torch_tools.default_dtype("float64"):
        full = float(
            anchor(ms.build_batch(anchor, atoms), compute_force=False)["energy"][0]
        )
        without = float(
            stripped(ms.build_batch(stripped, atoms), compute_force=False)["energy"][0]
        )
    assert abs(full - without) > 1e-3, (
        "zeroing the one-body magnetic coefficients left the isolated atom's "
        "energy unchanged, so nothing in this reference covers "
        "--use_magmom_one_body"
    )


def test_the_per_element_m_max_is_indexed_by_species(anchor, fixtures):
    """The lookup only two elements can distinguish.

    ``m_max`` is read as ``self.m_max[argmax(node_attrs)]``
    (mace/modules/extensions.py:1815-1817), so it is indexed by z-table
    position: entry 0 is oxygen and entry 1 is iron. Changing *only* the
    oxygen entry must move the Fe2O2 cluster and must leave every all-iron
    structure bit-identical, because their species index never reaches that
    slot. A transposed lookup fails both halves at once, and neither half is
    testable with a single-element fixture set -- which is why the cluster
    exists.
    """
    import copy  # noqa: PLC0415

    swapped = copy.deepcopy(anchor)
    with torch.no_grad():
        swapped.m_max[0] = 0.9

    def energy(model, atoms):
        with torch_tools.default_dtype("float64"):
            return float(
                model(ms.build_batch(model, atoms), compute_force=False)["energy"][0]
            )

    cluster = fixtures["mag_feo_cluster"]
    assert energy(anchor, cluster) != energy(swapped, cluster)
    for name in ("mag_fe_atom", "mag_fe_dimer_fm", "mag_fe3_canted"):
        assert energy(anchor, fixtures[name]) == energy(swapped, fixtures[name]), name


def test_no_fixture_saturates_the_moment_clamp(fixtures, anchor):
    """A saturated clamp is a structurally zero derivative in disguise.

    The radial magnetic basis reads ``1 - 2 * clamp(|m| / m_max, 0, 1)**2``
    (mace/modules/extensions.py:1818-1824). Above ``m_max`` the clamp is flat,
    so that whole path contributes exactly nothing to ``dE/dm`` -- a zero that
    looks like a computed one. The fixtures are chosen to stay inside; this
    asserts it against the anchor's own ``m_max`` rather than against the
    number written in the recipe.
    """
    z_table = {int(z): i for i, z in enumerate(anchor.atomic_numbers)}
    m_max = anchor.m_max.detach().numpy()
    ratios = []
    for name, atoms in fixtures.items():
        moments = np.asarray(atoms.arrays[ms.MAGMOM_KEY], dtype=float)
        ceiling = np.array(
            [m_max[z_table[int(z)]] for z in atoms.get_atomic_numbers()]
        )
        ratio = np.linalg.norm(moments, axis=-1) / ceiling
        assert (ratio < 1.0).all(), f"{name} saturates the clamp: {ratio}"
        assert (ratio > 0.0).all(), f"{name} carries a zero moment: {ratio}"
        ratios.append(ratio)
    spread = np.concatenate(ratios)
    assert spread.max() - spread.min() > 0.5, (
        "every fixture now samples the magnetic basis at nearly the same "
        "argument, so the reference pins one point of it rather than a range"
    )


# ---------------------------------------------------------------------------
# The other two surfaces, against the same reference
# ---------------------------------------------------------------------------


def test_magforces_reach_no_calculator_which_is_why_this_golden_is_model_route(
    fixtures, anchor
):
    """The reason this family's reference is not taken through a calculator.

    ``MagneticMACECalculator`` runs the forward with its default
    ``compute_magforces=True``, computes ``dE/dm`` and then keeps energy,
    free_energy, node_energy, forces and stress. If it ever starts exposing
    the magnetic forces, this fails and the calculator route becomes an
    option; until then, the ``golden_outputs`` hook is not a convenience.
    """
    from mace.calculators import MagneticMACECalculator  # noqa: PLC0415

    calc = MagneticMACECalculator(
        models=[anchor], device="cpu", default_dtype="float64",
        magmom_key=ms.MAGMOM_KEY,
    )
    probe = fixtures["mag_fe_dimer_fm"].copy()
    probe.calc = calc
    probe.get_potential_energy()
    assert "magforces" not in calc.results


def test_the_calculator_agrees_with_the_reference_on_what_it_does_expose(
    fixtures, anchor, reference
):
    """One reference, two surfaces.

    The calculator cannot serve the whole reference -- it drops the magnetic
    forces and, unlike ``MACECalculator``, never writes an ``energies`` key at
    all -- but it must not disagree with it on the part it carries.

    The second half is the collision. What ``MagneticMACECalculator`` calls
    ``node_energy`` is the model's ``node_energy`` with E0 subtracted
    (mace/calculators/mace.py:1388), while the model's own spelling of that
    word is the E0-inclusive quantity, which is the ``energies`` channel. Both
    are per-atom scalars in eV, so nothing but the numbers distinguishes them,
    and before the surface-scoped alias in model_keys.py a model-route
    reference and a calculator-route one landed on the same channel holding
    two different quantities. Here that difference is asserted to still be the
    E0 table.
    """
    from mace.calculators import MagneticMACECalculator  # noqa: PLC0415

    calc = MagneticMACECalculator(
        models=[anchor], device="cpu", default_dtype="float64",
        magmom_key=ms.MAGMOM_KEY,
    )
    snapshot = harness.snapshot_outputs(
        calc, fixtures, dtype="float64", device="cpu", backend="e3nn"
    )
    harness.compare_to_reference(
        snapshot, reference, row=TOL.name, channels=["energy", "forces"]
    )
    assert "energies" not in snapshot["fixtures"]["mag_fe_atom"]["outputs"]

    e0_table = {
        int(z): float(e)
        for z, e in zip(
            anchor.atomic_numbers,
            anchor.atomic_energies_fn.atomic_energies.flatten(),
        )
    }
    for name, atoms in fixtures.items():
        probe = atoms.copy()
        probe.calc = calc
        probe.get_potential_energy()
        inclusive = np.asarray(
            reference["fixtures"][name]["outputs"]["energies"]["value"]
        )
        expected_e0 = np.array(
            [e0_table[int(z)] for z in atoms.get_atomic_numbers()]
        )
        difference = inclusive - calc.results["node_energy"]
        assert np.abs(difference - expected_e0).max() < TOL.atol, (
            f"{name}: the calculator's node_energy and the model's no longer "
            f"differ by exactly the E0 table, so the two channels this "
            f"registry keeps apart have started meaning something else"
        )


def test_the_eval_cli_reproduces_the_reference_including_magforces(
    fixtures, reference, tmp_path
):
    """The third surface: ``mace_eval_configs --return_magforces``.

    This is the only way a user gets ``dE/dm`` out without writing python, and
    it is a different code path from both the forward and the calculator: it
    batches, splits per structure, writes onto the ase objects under a prefix
    and serialises to extxyz. Every step is somewhere the magnetic columns
    could go missing or be misaligned, and none of it is covered by pinning
    the forward.

    Compared at the same row as everything else. extxyz writes per-atom
    columns as ``%16.8f``, which is coarser than float64 and finer than the
    fp64 row by two orders of magnitude, so the file round trip is inside the
    bound rather than being given one of its own.
    """
    snapshot = harness.snapshot_outputs(
        ms.MagneticEvalCLI(tmp_path),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(
        snapshot,
        reference,
        row=TOL.name,
        channels=["energy", "forces", "magforces", "energies"],
    )


def test_compute_magforces_is_only_honoured_alongside_the_forces(anchor, fixtures):
    """``--compute_magforces``, and the coupling nothing else states.

    ``get_outputs`` reaches ``compute_forces_magforces`` only on the branch
    where ``compute_force`` is also true (mace/modules/utils.py:317-325);
    asking for magnetic forces without atomic forces falls through to the
    plain branch and returns ``None`` -- no error, no warning. The harness
    then leaves the channel out entirely and a reference that pins it fails
    with "channel vanished", which is the good outcome and the one this
    records.
    """
    atoms = fixtures["mag_fe_dimer_fm"]
    with torch_tools.default_dtype("float64"):
        both = anchor(
            ms.build_batch(anchor, atoms), compute_force=True, compute_magforces=True
        )
        neither = anchor(
            ms.build_batch(anchor, atoms), compute_force=False, compute_magforces=True
        )
        off = anchor(
            ms.build_batch(anchor, atoms), compute_force=True, compute_magforces=False
        )
    assert both["magforces"] is not None
    assert neither["magforces"] is None
    assert off["magforces"] is None

    snapshot = harness.snapshot_outputs(
        ms.MagneticForward(anchor, compute_magforces=False),
        {"mag_fe_dimer_fm": atoms},
    )
    assert "magforces" not in snapshot["fixtures"]["mag_fe_dimer_fm"]["outputs"]


# ---------------------------------------------------------------------------
# The symmetry --data_aug_magmom exists to induce
# ---------------------------------------------------------------------------


def test_a_joint_rotation_is_a_symmetry_and_a_spin_only_one_is_not(anchor, fixtures):
    """Why ``--data_aug_magmom`` is a training flag and not a property.

    Rotating positions and moments together leaves the energy invariant: that
    is architectural, and it holds here at the same row the reference uses.
    Rotating the moments alone does *not*, because the model couples the
    moment to the lattice -- and that is the whole reason the augmentation
    exists, since a non-relativistic magnet should be invariant under it and
    this architecture only becomes so by being trained on rotated copies.

    Pinned on the committed anchor because both halves are load-bearing and
    the second one is the surprising one: if a rewrite made the model
    spin-only invariant, ``--data_aug_magmom`` would silently become a no-op
    and nothing else in the suite would notice.
    """
    atoms = fixtures["mag_fe3_canted"]
    generator = torch.Generator().manual_seed(20260810)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotation = q.numpy()

    def energy(positions, moments):
        probe = atoms.copy()
        probe.positions = positions
        probe.arrays[ms.MAGMOM_KEY] = moments
        with torch_tools.default_dtype("float64"):
            return float(
                anchor(ms.build_batch(anchor, probe), compute_force=False)["energy"][0]
            )

    moments = np.asarray(atoms.arrays[ms.MAGMOM_KEY], dtype=float)
    plain = energy(atoms.positions, moments)
    joint = energy(atoms.positions @ rotation.T, moments @ rotation.T)
    spin_only = energy(atoms.positions, moments @ rotation.T)

    assert joint == pytest.approx(plain, abs=TOL.atol)
    assert abs(spin_only - plain) > 1e-3, (
        "rotating the moments alone no longer changes the energy, so this "
        "architecture has become spin-only invariant and --data_aug_magmom "
        "induces something it already has"
    )
