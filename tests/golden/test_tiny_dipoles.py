"""The tiny AtomicDipolesMACE anchor reproduces its committed reference.

The energy anchors next door pin an energy assembly. This one pins the other
half of the model zoo: a graph dipole, which is a scatter-sum of per-atom
vectors plus a fixed-charge baseline and shares no arithmetic with an energy.
RET-4 deletes the dipole classes and degrades the per-model parity tests down
to exactly this reference, so it runs per pull request, on CPU, with no
network and no optional dependency, and carries no capability marker -- an
anchor that could skip would be an anchor that could rot.
"""

import json

import numpy as np
import pytest
import torch

from tests.golden import harness, routes
from tests.golden.build_dipole_anchor import ANCHOR_CONFIG, build_model

MODEL_PATH = harness.MODELS_DIR / "tiny_dipoles.model"
SIDECAR_PATH = harness.MODELS_DIR / "tiny_dipoles.build.json"
REFERENCE_PATH = harness.REFERENCES_DIR / "tiny_dipoles_e3nn_cpu_fp64.json"

#: The reference is taken on the molecular fixtures only. A dipole is defined
#: up to an origin, and under periodic boundaries the choice of unit cell
#: moves it, so a golden taken on the slabs or the triclinic cell would pin a
#: number that is deterministic and physically meaningless -- the worst kind
#: to hand a rewrite as a target.
FIXTURE_TAGS = ("molecular",)
#: The manifest is shared; this anchor knows H/C/O only.
FIXTURE_ELEMENTS = (1, 6, 8)


def _load():
    return torch.load(MODEL_PATH, weights_only=False, map_location="cpu").to(
        torch.float64
    )


def _assert_within(row, got, want, what):
    """Compare at a named row of the one tolerance table.

    Spelled out rather than handed to ``numpy.testing.assert_allclose``
    because passing ``atol=``/``rtol=`` keywords anywhere under
    ``tests/golden/`` is what ``test_harness.py`` scans for, and rightly so: a
    literal in a keyword is how a second tolerance table starts. The bound is
    the same one ``harness.compare_to_reference`` applies.
    """
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    bound = row.atol + row.rtol * np.abs(want)
    excess = np.abs(got - want) - bound
    assert excess.max() <= 0.0, (
        f"{what}: outside the '{row.name}' row by {excess.max():.3g}\n"
        f"  got       {got.tolist()}\n"
        f"  reference {want.tolist()}"
    )


def _projection(out):
    return {
        "dipole": routes.as_numpy(out["dipole"][0]),
        "atomic_dipoles": routes.as_numpy(out["atomic_dipoles"]),
    }


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(tags=FIXTURE_TAGS, elements=FIXTURE_ELEMENTS)


def test_anchor_reproduces_its_reference(fixtures):
    model = _load()
    snapshot = harness.snapshot_outputs(
        routes.ForwardRoute(model, _projection),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    reference = harness.load_reference(REFERENCE_PATH)
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


def test_anchor_is_the_class_it_claims_to_be():
    model = _load()
    assert type(model).__name__ == "AtomicDipolesMACE"
    # Not an energy model, and the class asserts as much on construction
    # (mace/modules/models.py:664). If this ever gains an atomic_energies_fn
    # the anchor was rebuilt as something else.
    assert not hasattr(model, "atomic_energies_fn")
    assert not hasattr(model, "scale_shift")


def test_the_committed_anchor_carries_the_plain_e3nn_basis():
    """Why this checkpoint reproduces on a machine with cuequivariance and on
    one without.

    ``AtomicDipolesMACE`` accepts ``use_reduced_cg`` and ignores it, so
    ``EquivariantProductBasisBlock`` gets its ``None`` default and
    ``SymmetricContractionWrapper`` evaluates ``use_reduced_cg and
    CUET_AVAILABLE`` to a falsy value either way
    (mace/modules/wrapper_ops.py:428). The plain-MACE anchor has to pin the
    flag to False because its ``True`` default is silently degraded when
    cuequivariance is absent; here the reduced path is unreachable.

    That is a claim about somebody else's dispatch table, so it is measured:
    the recipe is re-run in this process and its Clebsch-Gordan buffers are
    compared bit for bit against the committed checkpoint's. The buffers are
    not random -- they are CG coefficients -- so this is a real comparison and
    not a re-seeding. It runs both with cuequivariance installed (the nightly
    full-scope job) and without (every PR), and only one of the two can pass
    if the reduced basis ever leaks in.
    """
    committed = _load()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        rebuilt = build_model().to(torch.float64)
    finally:
        torch.set_default_dtype(previous)

    committed_buffers = {
        name: tensor
        for name, tensor in committed.named_buffers()
        if "U_matrix" in name
    }
    assert committed_buffers, "the anchor has no symmetric-contraction basis"
    rebuilt_buffers = dict(rebuilt.named_buffers())
    for name, tensor in committed_buffers.items():
        other = rebuilt_buffers[name]
        assert other.shape == tensor.shape, (
            f"{name}: the recipe now builds a {tuple(other.shape)} basis and "
            f"the committed anchor carries {tuple(tensor.shape)}. The reduced "
            f"Clebsch-Gordan basis has a different extent, so this is what a "
            f"cuequivariance-dependent build looks like."
        )
        assert torch.equal(other, tensor), f"{name} differs from the committed anchor"

    for product in committed.products:
        contraction = product.symmetric_contractions
        assert type(contraction).__module__.startswith("mace."), (
            "the committed anchor holds a cuequivariance contraction object, "
            "which cannot be unpickled on a machine without the package"
        )


def test_the_fixed_charge_baseline_is_live():
    """The term the committed reference cannot pin, pinned algebraically.

    ``AtomicDipolesMACE`` adds ``compute_fixed_charge_dipole(data["charges"],
    ...)`` to the scatter-summed atomic dipoles
    (mace/modules/models.py:825-831). None of the committed fixtures carries a
    reference-charge array, so ``AtomicData`` fills the batch with zeros and
    the baseline is identically zero in the reference -- a rewrite could drop
    the whole term and the golden would not notice. Asserting the fixtures
    carry no charges is not enough; this checks the term is wired by feeding
    the graph a non-zero charge vector and requiring the dipole to move by
    exactly sum_i q_i r_i.

    Kept out of the reference on purpose: the fixtures are P0-1's and are
    edit-locked, and inventing a second charged fixture here would fork the
    structure set that every other golden shares.
    """
    model = _load()
    atoms = harness.load_fixtures(names=["water_cluster"])["water_cluster"]
    graph = routes.graph_batch(model, atoms)

    assert torch.count_nonzero(graph["charges"]) == 0, (
        "the fixture now carries reference charges, so the committed "
        "reference was taken at a non-zero baseline and this test's premise "
        "is gone"
    )

    from mace.tools import torch_tools  # noqa: PLC0415

    with torch_tools.default_dtype("float64"):
        neutral = model(routes.graph_batch(model, atoms))["dipole"][0].detach()

        charged = routes.graph_batch(model, atoms)
        charges = torch.tensor(
            [0.4, -0.2, -0.2, 0.3, -0.15, -0.15, 0.5, -0.25, -0.25],
            dtype=torch.float64,
        )
        charged["charges"] = charges
        moved = model(charged)["dipole"][0].detach()

    # sum_i q_i r_i is in e*Ang; this class divides by 1e-11 / c / e
    # (mace/modules/utils.py:622), which is the e*Ang -> Debye conversion, so
    # the baseline arrives in the unit the channel declares. The constants are
    # taken from scipy, which is where the model takes them, and not from
    # ase.units: the two disagree in the ninth significant figure (CODATA
    # vintages), which is above the fp64 row and would fail this on the
    # conversion rather than on the physics.
    from scipy.constants import c, e  # noqa: PLC0415

    e_ang_to_debye = 1.0 / (1e-11 / c / e)
    expected = (charges @ graph["positions"].detach()) * e_ang_to_debye
    _assert_within(
        harness.FP64_CPU_REFERENCE,
        (moved - neutral).numpy(),
        expected.numpy(),
        "the dipole did not move by the fixed-charge baseline",
    )
    assert np.abs(expected.numpy()).max() > 1.0, (
        "the probe charges produce a baseline too small to distinguish from "
        "the term being dropped"
    )


def test_the_two_fixed_charge_baselines_do_not_share_a_unit():
    """A divergence between the two dipole families, turned into a number.

    ``AtomicDipolesMACE`` builds its baseline with
    ``compute_fixed_charge_dipole``, which divides by ``1e-11 / c / e`` and so
    returns Debye. ``AtomicDielectricMACE`` -- the MACE-MDP class -- builds
    its own with ``compute_fixed_charge_dipole_polar``, where that division is
    present but commented out (mace/modules/utils.py:634-636), so its baseline
    is e*Ang. The two graph dipoles are therefore *not* the same quantity even
    though both land in a channel the schema declares as Debye, and the ratio
    is exactly the conversion factor.

    Nothing here judges which one is right; that is a physics question for
    ELEC-2/FM-2. What a golden can do is make sure the discrepancy cannot be
    tidied away by accident while both references still pass, which is what a
    rewrite unifying the two functions would otherwise do.
    """
    from scipy.constants import c, e  # noqa: PLC0415

    from mace.modules.utils import (  # noqa: PLC0415
        compute_fixed_charge_dipole,
        compute_fixed_charge_dipole_polar,
    )

    charges = torch.tensor([0.4, -0.2, -0.2], dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.3], [-0.3, 0.85, 0.0]], dtype=torch.float64
    )
    batch = torch.zeros(3, dtype=torch.long)

    debye = compute_fixed_charge_dipole(charges, positions, batch, 1)
    raw = compute_fixed_charge_dipole_polar(charges, positions, batch, 1)
    ratio = (debye / raw).numpy()
    _assert_within(
        harness.FP64_CPU_REFERENCE,
        ratio,
        np.full_like(ratio, 1.0 / (1e-11 / c / e)),
        "the two fixed-charge baselines no longer differ by the e*Ang->Debye "
        "conversion",
    )
    assert not np.allclose(ratio, 1.0), (
        "the two baselines now agree; if that is deliberate, one of the two "
        "dipole references is in a different unit than it was and has to be "
        "regenerated with the physics written down"
    )


def test_the_calculator_route_reaches_the_same_dipole(fixtures):
    """The other door, and the one that is nearly closed.

    ``MACECalculator(model_type="DipoleMACE")`` puts only ``dipole`` in its
    results -- ``atomic_dipoles`` is not in ``results_map``
    (mace/calculators/mace.py:719-738), so the per-atom term is reachable
    through the forward alone. The one channel both doors carry has to be the
    same number, or the reference pins a route nothing else uses.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415

    model = _load()
    calc = MACECalculator(
        models=[model], device="cpu", default_dtype="float64", model_type="DipoleMACE"
    )
    via_calculator = harness.snapshot_outputs(routes.CalculatorRoute(calc), fixtures)
    via_model = harness.snapshot_outputs(routes.ForwardRoute(model, _projection), fixtures)
    for name in fixtures:
        assert (
            via_calculator["fixtures"][name]["outputs"]["dipole"]
            == via_model["fixtures"][name]["outputs"]["dipole"]
        ), name
        assert set(via_calculator["fixtures"][name]["outputs"]) == {"dipole"}


def test_a_dipole_model_cannot_be_driven_through_an_ase_accessor(fixtures):
    """Why this file uses a route object instead of handing over the calculator.

    The harness's calculator branch starts at ``get_potential_energy``, which
    is the right entry point for every energy model and does not exist for
    this family. Pinned as a contract so the indirection in
    ``tests/golden/routes.py`` is a stated consequence rather than a habit: if
    a future ``MACECalculator`` learns to answer for a dipole-only model, this
    fails and the route object can go.
    """
    from ase.calculators.calculator import PropertyNotImplementedError  # noqa: PLC0415

    from mace.calculators import MACECalculator  # noqa: PLC0415

    calc = MACECalculator(
        models=[_load()],
        device="cpu",
        default_dtype="float64",
        model_type="DipoleMACE",
    )
    probe = fixtures["water_cluster"].copy()
    probe.calc = calc
    with pytest.raises(PropertyNotImplementedError):
        probe.get_potential_energy()


def test_reference_carries_dtype_units_and_provenance():
    reference = harness.load_reference(REFERENCE_PATH)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["units"]["energy"] == "eV"
    provenance = reference["provenance"]
    assert provenance["source"].endswith(MODEL_PATH.name)
    assert provenance["recipe"] == "tests/golden/build_dipole_anchor.py"
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    assert set(reference["fixtures"]) == set(
        harness.load_fixtures(tags=FIXTURE_TAGS, elements=FIXTURE_ELEMENTS)
    )
    for entry in reference["fixtures"].values():
        assert set(entry["outputs"]) == {"dipole", "atomic_dipoles"}
        for channel in entry["outputs"].values():
            assert channel["unit"] == "Debye"
            assert channel["kind"] in harness.KINDS


def test_sidecar_records_how_the_anchor_was_built():
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    assert sidecar["model"] == MODEL_PATH.name
    assert sidecar["class"] == "AtomicDipolesMACE"
    assert sidecar["dtype"] == "float64"
    assert sidecar["seed"]
    assert sidecar["command"]
    assert "regenerate.py" in sidecar["regenerate_with"]
    assert sidecar["config"] == ANCHOR_CONFIG
    # An energy model's sidecar records its E0 table; this class asserts there
    # is none, and recording it as null is the difference between "no E0s" and
    # "nobody wrote them down".
    assert sidecar["atomic_energies"] is None


def test_anchor_checkpoint_stays_small():
    size_mb = MODEL_PATH.stat().st_size / 1e6
    assert size_mb < 1.5, f"{MODEL_PATH.name} is {size_mb:.2f} MB"
