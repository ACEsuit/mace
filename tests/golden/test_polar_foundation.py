"""The published MACE-Polar model reproduces its committed reference.

ELEC-2 rebuilds ``PolarMACE`` on the new architecture and converts this
checkpoint onto it; this file is the number it has to land on. What is pinned
is the whole surface the calculator exposes -- the energy/force/stress side
*and* the electrostatics side (dipole, per-atom charges and spins, the three
energy decompositions, the density coefficients, the spin-resolved density and
the Fukui functions) -- because the electrostatics is the part that has no
analogue in any energy model and therefore no other reference.

**There is no polarizability here, and that is not an omission.**
``AtomicDielectricMACE`` is the only class in the tree that emits that key
(``mace/modules/models.py:1190``); ``PolarMACE`` emits a dipole and its
electrostatics and never a polarizability -- the word does not occur in the
class. A test that "checks the polarizability of the polar model" checks
nothing, so the polarizability golden is next door, on MACE-MDP.

Marked ``polar`` **and** ``network``: the model is downloaded, and the forward
needs ``graph_longrange``. Locally either one missing is a clean skip; the
ci-extensions ``polar`` job guarantees both (``require-caps: polar,network``),
so there it fails instead.
"""

import numpy as np
import pytest

from tests.golden import harness
from tests.golden.targets.foundation_references import POLAR_FIXTURES

REFERENCE_PATH = harness.REFERENCES_DIR / "polar_foundation_cpu_fp64.json"

#: Kept in step with tests/golden/regenerate.py::POLAR_MODEL. Not imported
#: from it: regenerate.py is the write path and importing it from a read path
#: would make a mistake there invisible here.
POLAR_MODEL = "polar-1-s"

pytestmark = [pytest.mark.polar, pytest.mark.network]


@pytest.fixture(name="polar_calc", scope="module")
def fixture_polar_calc():
    from mace.calculators.foundations_models import mace_polar  # noqa: PLC0415

    # dtype and device spelled out: mace_polar defaults to float32
    # (mace/calculators/foundations_models.py:343) and to cuda when one is
    # present, so a golden that took the defaults would be an fp32 GPU
    # snapshot asserted at the fp64 CPU row.
    return mace_polar(model=POLAR_MODEL, device="cpu", default_dtype="float64")


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    # The same named set the reference was generated over; see
    # targets/foundation_references.py for why it is named rather than
    # taken as 'everything'.
    return harness.load_fixtures(names=list(POLAR_FIXTURES))


def test_polar_foundation_reproduces_its_reference(polar_calc, fixtures):
    snapshot = harness.snapshot_outputs(
        polar_calc, fixtures, dtype="float64", device="cpu", backend="e3nn"
    )
    reference = harness.load_reference(REFERENCE_PATH)
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


def test_the_loader_really_produced_a_float64_cpu_evaluation(polar_calc):
    """The dtype discipline, asserted rather than assumed.

    The published weights are float32 and ``MACECalculator`` upcasts them when
    ``default_dtype="float64"`` (it logs a warning while doing so). The upcast
    is exact, so the arithmetic is genuinely fp64 -- but only if the model
    really was converted, and the loader's own default would not have done it.
    """
    import torch  # noqa: PLC0415

    for model in polar_calc.models:
        assert next(model.parameters()).dtype == torch.float64
        assert next(model.parameters()).device.type == "cpu"
    assert polar_calc.default_dtype == "float64"


def test_polar_mace_emits_no_polarizability(polar_calc, fixtures):
    """The decision this file rests on, checked against the running model.

    Pinned as a contract because the reference cannot express it: a reference
    records the channels that were produced, and "this key was never produced"
    looks exactly like "nobody asked for it". If a future PolarMACE gains a
    polarizability, this fails and says the golden has to be extended
    deliberately rather than acquiring a channel by accident.

    Deliberately *not* asserted against ``implemented_properties``, which
    would look like the obvious check and is worthless: ``MACECalculator``
    declares no such attribute of its own, so every instance extends ase's
    class-level list in place and this calculator advertises whatever an
    MDP calculator built earlier in the same session added. The first version
    of this test did assert it, passed on its own, and failed the moment the
    two golden files ran in one process. See
    ``test_mdp_foundation.py::test_implemented_properties_leak_onto_the_shared_ase_list``.
    """
    import inspect  # noqa: PLC0415

    from mace.modules.extensions import PolarMACE  # noqa: PLC0415

    probe = fixtures["water_cluster"].copy()
    probe.calc = polar_calc
    probe.get_potential_energy()
    assert "polarizability" not in polar_calc.results
    assert "polarizability_sh" not in polar_calc.results
    assert type(polar_calc.models[0]).__name__ == "PolarMACE"
    assert "polarizability" not in inspect.getsource(PolarMACE)


def test_the_reference_pins_the_electrostatics_and_not_only_the_energy():
    """A guard against the reference quietly shrinking.

    Every one of these keys is emitted only by this family, and each was, at
    some point in the harness's history, dropped without a word by a snapshot
    that recorded energy and forces and looked healthy. Naming them here means
    a regeneration that loses one fails on this list rather than on nothing.
    """
    reference = harness.load_reference(REFERENCE_PATH)
    required = {
        "energy",
        "forces",
        "dipole",
        "charges",
        "spins",
        "interaction_energy",
        "electrostatic_energy",
        "electron_energy",
        "density_coefficients",
        "spin_charge_density",
        "fukui_functions",
    }
    for name, entry in reference["fixtures"].items():
        missing = sorted(required - set(entry["outputs"]))
        assert not missing, f"{name}: the reference no longer pins {missing}"
        if entry["periodic"]:
            assert "stress" in entry["outputs"], name
        else:
            # A stress is meaningless without a cell and the harness drops it;
            # asserted so that a change of that rule shows up here.
            assert "stress" not in entry["outputs"], name


def test_reference_carries_dtype_units_and_provenance():
    reference = harness.load_reference(REFERENCE_PATH)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["metadata"]["model_class"] == "PolarMACE"
    assert reference["metadata"]["foundation_model"] == POLAR_MODEL
    provenance = reference["provenance"]
    assert POLAR_MODEL in provenance["source"]
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    for entry in reference["fixtures"].values():
        for channel in entry["outputs"].values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS
            assert np.isfinite(np.asarray(channel["value"], dtype=float)).all()
