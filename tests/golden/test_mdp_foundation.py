"""The published MACE-MDP model reproduces its committed reference.

``mace_mdp`` loads an ``AtomicDielectricMACE``, which is the **only** class in
the tree that emits a ``polarizability`` (``mace/modules/models.py:1190``), so
this is the one place that quantity is pinned at all. FM-2 converts this
checkpoint; the numbers below are what the conversion has to reproduce.

The reference is taken through the model's ``forward``, not through the
calculator, for a reason that is a property of the family and not a
preference: ``dmu_dr`` and ``dalpha_dr`` -- the position derivatives of the
dipole and the polarizability, which are what an infrared or Raman intensity
is computed from -- appear in no calculator's ``results`` at all
(``results_map``, ``mace/calculators/mace.py:719-738``). A calculator-route
reference would pin four channels and silently leave the two that only this
family has. The calculator's own four are then asserted against the same file,
which is the fourth deliverable of this ticket: one number, two doors.

``network``-marked and nothing else: ``AtomicDielectricMACE`` lives in
``mace/modules/models.py`` and needs no optional dependency, so the only
capability at stake is the download.
"""

import numpy as np
import pytest
import torch

from tests.golden import harness, routes

REFERENCE_PATH = harness.REFERENCES_DIR / "mdp_foundation_cpu_fp64.json"

#: MACE-MDP is an organic dipole/polarizability model and a dipole under
#: periodic boundaries is origin-dependent, so the golden stays on the
#: molecular fixtures. Same choice, same reason, as the tiny dipole anchor.
#: The same named set the reference was generated over; see
#: targets/foundation_references.py for why it is named rather than taken
#: from the "molecular" tag, which the magnetic fixtures also carry.
from tests.golden.targets.foundation_references import MDP_FIXTURES


#: The four the calculator route also carries. `atomic_dipoles`, `dmu_dr` and
#: `dalpha_dr` are model-route only.
SHARED_WITH_CALCULATOR = ("charges", "dipole", "polarizability", "polarizability_sh")

pytestmark = pytest.mark.network


def _projection(out):
    projected = {
        "charges": routes.as_numpy(out["charges"]),
        "dipole": routes.as_numpy(out["dipole"][0]),
        "atomic_dipoles": routes.as_numpy(out["atomic_dipoles"]),
        "polarizability": routes.as_numpy(out["polarizability"][0]),
        "polarizability_sh": routes.as_numpy(out["polarizability_sh"][0]),
    }
    for key in ("dmu_dr", "dalpha_dr"):
        if out.get(key) is not None:
            projected[key] = routes.as_numpy(out[key])
    return projected


@pytest.fixture(name="mdp_model", scope="module")
def fixture_mdp_model():
    from mace.calculators.foundations_models import mace_mdp  # noqa: PLC0415

    # device and dtype spelled out even though mace_mdp already defaults to
    # float64: the default is one signature edit away from being float32, and
    # the device default is cuda whenever one is visible.
    return (
        mace_mdp(device="cpu", default_dtype="float64", return_raw_model=True)
        .to(torch.float64)
        .eval()
    )


@pytest.fixture(name="mdp_calc", scope="module")
def fixture_mdp_calc():
    from mace.calculators.foundations_models import mace_mdp  # noqa: PLC0415

    return mace_mdp(device="cpu", default_dtype="float64")


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=list(MDP_FIXTURES))


def test_mdp_foundation_reproduces_its_reference(mdp_model, fixtures):
    snapshot = harness.snapshot_outputs(
        routes.ForwardRoute(
            mdp_model,
            _projection,
            forward_kwargs={"compute_dielectric_derivatives": True},
        ),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    reference = harness.load_reference(REFERENCE_PATH)
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


def test_the_calculator_agrees_with_the_direct_model_snapshot(
    mdp_calc, mdp_model, fixtures
):
    """Deliverable 4: the ``mace_mdp`` calculator path, pinned to the same file.

    Two separate claims, because they need different strictness.

    Drift is caught against the committed reference, at the same tolerance row
    the model route uses. "The two routes agree with each other" would not do
    on its own: they would still agree after both drifted together.

    A route difference is caught between the two routes as evaluated here, in
    this process, and there bit-equality is the right bar: the calculator is
    the same forward with a dict rename in front of it, so anything other than
    equality means the graph reached the model differently on one of them.

    That second comparison cannot be made against the file. Bit-equality is a
    claim about the two routes, not about the machine, and the last bit of a
    float64 reduction moves between torch builds -- 2.11 and 2.13 disagree at
    1e-16 on this model. Holding the file to bit-equality would have been a
    claim about the build that recorded it.
    """
    reference = harness.load_reference(REFERENCE_PATH)
    snapshot = harness.snapshot_outputs(
        routes.CalculatorRoute(mdp_calc),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
        channels=list(SHARED_WITH_CALCULATOR),
    )
    harness.compare_to_reference(
        snapshot,
        reference,
        row=harness.FP64_CPU_REFERENCE.name,
        channels=list(SHARED_WITH_CALCULATOR),
    )
    forward = harness.snapshot_outputs(
        routes.ForwardRoute(mdp_model, _projection),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
        channels=list(SHARED_WITH_CALCULATOR),
    )
    for name in fixtures:
        got = snapshot["fixtures"][name]["outputs"]
        want = forward["fixtures"][name]["outputs"]
        for channel in SHARED_WITH_CALCULATOR:
            assert got[channel] == want[channel], (
                f"{name}/{channel}: the calculator route and the forward route "
                f"are not bit-identical, so one of them is feeding the model a "
                f"different graph"
            )


def test_the_calculator_surface_is_the_dipole_polarizability_one(mdp_calc, fixtures):
    """``model_type``, the property list and the two shapes the ticket names.

    The property list is checked by *difference* rather than by membership,
    and that is deliberate: ``MACECalculator`` has no
    ``implemented_properties`` of its own, so ``self.implemented_properties.
    extend(...)`` (mace/calculators/mace.py:212-233) mutates
    ``ase.calculators.calculator.Calculator.implemented_properties`` -- a list
    shared by every calculator in the process. After any MACE calculator has
    been built, a membership assertion here passes no matter what this one
    declares. Counting what *this* construction added is exact either way, and
    stays correct once the leak is fixed. See
    ``test_implemented_properties_leak_onto_the_shared_ase_list``.
    """
    from collections import Counter  # noqa: PLC0415

    from ase.calculators.calculator import Calculator  # noqa: PLC0415

    from mace.calculators.foundations_models import mace_mdp  # noqa: PLC0415

    assert mdp_calc.model_type == "DipolePolarizabilityMACE"

    before = Counter(Calculator.implemented_properties)
    fresh = mace_mdp(device="cpu", default_dtype="float64")
    added = Counter(fresh.implemented_properties) - before
    assert set(added) == {"dipole", "charges", "polarizability", "polarizability_sh"}

    probe = fixtures["water_cluster"].copy()
    fresh.calculate(probe)
    assert np.asarray(fresh.results["dipole"]).shape == (3,)
    assert np.asarray(fresh.results["polarizability"]).shape == (3, 3)


def test_implemented_properties_leak_onto_the_shared_ase_list(mdp_calc):
    """The defect the test above has to route around, pinned so it is dated.

    ``MACECalculator`` never gives itself an ``implemented_properties``
    attribute, so every instance extends ase's class-level list in place. Two
    consequences, both silent: a MACE calculator advertises properties that
    belong to a *different* MACE calculator built earlier in the same process,
    and every non-MACE ase calculator in that process inherits them too.

    Left as characterisation rather than fixed here, because a golden change
    that also edits ``mace/calculators/mace.py`` is exactly the mixed change
    the goldens exist to make impossible. When it is fixed, this test fails
    and the counter above keeps working unchanged.
    """
    from ase.calculators.calculator import Calculator  # noqa: PLC0415

    from mace.calculators import MACECalculator  # noqa: PLC0415

    assert "implemented_properties" not in MACECalculator.__dict__
    assert mdp_calc.implemented_properties is Calculator.implemented_properties


def test_mace_mdp_refuses_another_model_type():
    """``mace_mdp`` only answers for its own class, and says so up front."""
    from mace.calculators.foundations_models import mace_mdp  # noqa: PLC0415

    with pytest.raises(ValueError, match="DipolePolarizabilityMACE"):
        mace_mdp(device="cpu", default_dtype="float64", model_type="MACE")


def test_mace_mdp_warns_that_it_is_not_an_energy_model(mdp_calc):
    """And the golden must not assert an energy, because there is none.

    ``AtomicDielectricMACE.forward`` returns no ``energy`` key at all, so the
    calculator leaves none in its results and an ase accessor raises. That is
    the whole reason this file drives ``calculate()`` through
    ``tests/golden/routes.CalculatorRoute``.
    """
    from ase.calculators.calculator import PropertyNotImplementedError  # noqa: PLC0415

    from mace.calculators.foundations_models import mace_mdp  # noqa: PLC0415

    with pytest.warns(UserWarning, match="not suitable for energies or forces"):
        mace_mdp(device="cpu", default_dtype="float64")

    probe = harness.load_fixtures(names=["water_cluster"])["water_cluster"].copy()
    probe.calc = mdp_calc
    with pytest.raises(PropertyNotImplementedError):
        probe.get_potential_energy()
    assert "energy" not in mdp_calc.results
    assert "forces" not in mdp_calc.results


def test_the_reference_pins_the_polarizability_and_its_derivatives():
    """The two things only this family has, named so a shrink fails loudly."""
    reference = harness.load_reference(REFERENCE_PATH)
    for name, entry in reference["fixtures"].items():
        outputs = entry["outputs"]
        missing = sorted(
            {
                "charges",
                "dipole",
                "atomic_dipoles",
                "polarizability",
                "polarizability_sh",
                "dmu_dr",
                "dalpha_dr",
            }
            - set(outputs)
        )
        assert not missing, f"{name}: the reference no longer pins {missing}"
        n_atoms = entry["n_atoms"]
        assert outputs["polarizability"]["shape"] == [3, 3]
        assert outputs["polarizability_sh"]["shape"] == [6]
        # The measured extents of the position_gradient kind, which the
        # harness leaves free because they are properties of the
        # differentiated quantity: 3 for the dipole, 9 for the flattened 3x3
        # polarizability.
        assert outputs["dmu_dr"]["shape"] == [3, n_atoms, 3]
        assert outputs["dalpha_dr"]["shape"] == [9, n_atoms, 3]


def test_reference_carries_dtype_units_and_provenance():
    reference = harness.load_reference(REFERENCE_PATH)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["metadata"]["model_class"] == "AtomicDielectricMACE"
    provenance = reference["provenance"]
    assert provenance["source"] == "mace_mdp()"
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    assert set(reference["fixtures"]) == set(harness.load_fixtures(names=list(MDP_FIXTURES)))
    for entry in reference["fixtures"].values():
        for channel in entry["outputs"].values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS
            assert np.isfinite(np.asarray(channel["value"], dtype=float)).all()
