"""Tests for the golden harness itself.

Everything here runs on numpy/ase/json only -- no model, no torch -- because
the harness is the one piece of machinery both the current stack and its
replacement have to share, and a test that needed the current stack to prove
that would be proving the opposite.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms

from tests.golden import harness

GOLDEN_ROOT = Path(harness.__file__).resolve().parent


# ---------------------------------------------------------------------------
# The structural constraint: framework-agnostic
# ---------------------------------------------------------------------------


def test_harness_source_never_names_the_framework():
    """`grep -c mace tests/golden/harness.py` must be 0.

    The parity suites that consume this module are forbidden from importing
    the legacy package; one convenience import here would make the shared
    comparison machinery unusable to exactly the tests it exists for.
    """
    source = Path(harness.__file__).read_text(encoding="utf-8")
    assert "mace" not in source.lower(), (
        "harness.py mentions the framework; it must stay importable by "
        "consumers that cannot import it"
    )


def test_harness_imports_with_the_framework_blocked():
    """Textual absence is not enough -- prove it imports and works anyway."""
    script = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name == 'mace' or name.startswith('mace.'):\n"
        "            raise ImportError('the framework is not importable here')\n"
        "        return None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        return self.find_module(name, path)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        f"sys.path.insert(0, {str(GOLDEN_ROOT.parent.parent)!r})\n"
        "from tests.golden import harness\n"
        "fixtures = harness.load_fixtures()\n"
        "assert len(fixtures) == 6, sorted(fixtures)\n"
        "assert 'mace' not in sys.modules\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


# ---------------------------------------------------------------------------
# The tolerance table
# ---------------------------------------------------------------------------


def test_tolerance_table_rows():
    assert set(harness.TOLERANCES) == {
        "fp64_cpu_reference",
        "fp64_accelerated_backend",
        "fp32",
    }
    assert harness.FP64_CPU_REFERENCE.atol == 1e-6
    assert harness.FP64_CPU_REFERENCE.rtol == 0.0
    assert harness.FP64_ACCELERATED_BACKEND.atol == 1e-5
    assert harness.FP32.rtol == 1e-3
    # The fp32 absolute floor is adopted from tests/extensions/polar, which
    # documents 5e-6 failing in CI. Keeping the two equal is the point.
    assert harness.FP32.atol == 5e-5
    for row in harness.TOLERANCES.values():
        assert row.rationale.strip(), f"row {row.name} has no rationale"


def test_unknown_tolerance_row_names_the_table():
    with pytest.raises(KeyError, match="fp64_cpu_reference"):
        harness.tolerance("whatever_is_convenient")


def test_no_other_golden_module_defines_its_own_tolerance():
    """The table is single-source; a local atol somewhere else defeats it."""
    literal = re.compile(r"\b(?:atol|rtol)\s*=\s*[0-9.]")
    offenders = []
    for path in sorted(GOLDEN_ROOT.rglob("*.py")):
        if path.name == "harness.py":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if literal.search(line):
                offenders.append(f"{path.relative_to(GOLDEN_ROOT)}:{number}: {line.strip()}")
    assert not offenders, (
        "tolerances must be imported from harness.py, not restated:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_fixture_manifest_matches_the_files():
    manifest = harness.load_manifest()
    assert len(manifest) == 6
    for name, entry in manifest.items():
        path = harness.FIXTURES_DIR / entry["file"]
        assert path.exists(), f"{name} points at a missing file {path}"


def test_fixture_set_reaches_every_neighbour_list_regime():
    fixtures = harness.load_fixtures()
    pbc = {name: tuple(bool(p) for p in at.pbc) for name, at in fixtures.items()}
    cells = {name: np.asarray(at.cell) for name, at in fixtures.items()}

    # fully aperiodic
    assert any(not any(p) for p in pbc.values())
    # mixed pbc with real vacuum
    assert pbc["slab_vacuum"] == (True, True, False)
    assert cells["slab_vacuum"][2].any()
    # mixed pbc whose vacuum row is all zeros -- the one that would divide a
    # stress by a zero volume if the neighbour-list layer did not patch it
    assert pbc["slab_zero_vacuum"] == (True, True, False)
    assert not cells["slab_zero_vacuum"][2].any()
    assert np.linalg.det(cells["slab_zero_vacuum"]) == 0.0
    # an isolated atom (zero edges) and a short dimer
    assert len(fixtures["isolated_atom"]) == 1
    dimer = fixtures["dimer_short"]
    assert len(dimer) == 2
    assert dimer.get_distance(0, 1) == pytest.approx(0.62, abs=1e-9)
    # three species across the set, all in {H, C, O}
    species = set()
    for atoms in fixtures.values():
        species |= set(atoms.get_atomic_numbers().tolist())
    assert species == {1, 6, 8}


def test_load_fixtures_by_tag_and_by_name():
    molecular = harness.load_fixtures(tags=["molecular"])
    assert set(molecular) == {"water_cluster", "isolated_atom", "dimer_short"}
    assert all(not any(at.pbc) for at in molecular.values())
    one = harness.load_fixtures(["triclinic_bulk"])
    assert list(one) == ["triclinic_bulk"]
    with pytest.raises(KeyError, match="nope"):
        harness.load_fixtures(["nope"])


def test_load_fixtures_returns_independent_copies():
    first = harness.load_fixtures(["water_cluster"])["water_cluster"]
    first.positions += 1.0
    second = harness.load_fixtures(["water_cluster"])["water_cluster"]
    assert not np.allclose(first.positions, second.positions)


# ---------------------------------------------------------------------------
# Snapshot schema
# ---------------------------------------------------------------------------


class FakeSource:
    """A calculator-shaped stub exercising every channel kind."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def golden_outputs(self, atoms):
        n = len(atoms)
        base = np.arange(n * 3, dtype=float).reshape(n, 3) / 10.0
        return {
            "energy": -1.25 + self.offset,
            "forces": base + self.offset,
            "stress": np.eye(3) * (0.5 + self.offset),
            "dipole": np.array([0.1, 0.2, 0.3]) + self.offset,
            "charges": np.linspace(-0.5, 0.5, n) + self.offset,
            "magforces": base * 2.0 + self.offset,
            "BEC": np.tile(np.eye(3), (n, 1, 1)) + self.offset,
            "latent_quads": np.ones((n, 5)) + self.offset,
            "scf_steps": 7,
        }


@pytest.fixture(name="two_fixtures")
def fixture_two_fixtures():
    return harness.load_fixtures(["triclinic_bulk", "water_cluster"])


def test_snapshot_records_kinds_units_and_shapes(two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    assert snap["schema_version"] == harness.SCHEMA_VERSION
    assert snap["dtype"] == "float64"
    bulk = snap["fixtures"]["triclinic_bulk"]["outputs"]
    assert bulk["energy"]["kind"] == harness.GRAPH_SCALAR
    assert bulk["energy"]["unit"] == "eV"
    assert bulk["forces"]["kind"] == harness.PER_ATOM_VECTOR
    assert bulk["forces"]["shape"] == [6, 3]
    assert bulk["dipole"]["kind"] == harness.GRAPH_VECTOR
    assert bulk["BEC"]["kind"] == harness.PER_ATOM_TENSOR
    assert bulk["BEC"]["shape"] == [6, 3, 3]
    assert bulk["magforces"]["unit"] == "eV/muB"
    assert bulk["latent_quads"]["shape"] == [6, 5]


def test_stress_is_recorded_only_where_it_means_something(two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    assert "stress" in snap["fixtures"]["triclinic_bulk"]["outputs"]
    assert "stress" not in snap["fixtures"]["water_cluster"]["outputs"]


def test_metadata_channels_are_recorded_but_not_asserted(two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    entry = snap["fixtures"]["triclinic_bulk"]
    assert entry["metadata"]["scf_steps"] == 7
    assert "scf_steps" not in entry["outputs"]
    # moving a metadata value must not fail a comparison
    other = harness.snapshot_outputs(FakeSource(), two_fixtures)
    other["fixtures"]["triclinic_bulk"]["metadata"]["scf_steps"] = 99
    harness.compare_to_reference(other, snap, row="fp64_cpu_reference")


def test_magmom_is_recorded_as_an_input():
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.set_initial_magnetic_moments([0.7, -0.2, 0.1] * 3)
    atoms.info["total_charge"] = -1.0
    snap = harness.snapshot_outputs(FakeSource(), {"spun": atoms})
    inputs = snap["fixtures"]["spun"]["inputs"]
    assert inputs["magmom"]["kind"] == harness.PER_ATOM_VECTOR
    assert inputs["magmom"]["shape"] == [9, 3]
    # a collinear moment is stored along z, so the schema is vector-valued
    # from the start rather than widened later
    assert inputs["magmom"]["value"][0] == [0.0, 0.0, 0.7]
    assert inputs["total_charge"]["value"] == -1.0


def test_unknown_channel_must_be_declared(two_fixtures):
    class Odd(FakeSource):
        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["not_a_channel"] = 1.0
            return out

    # unknown channels are dropped silently by default...
    snap = harness.snapshot_outputs(Odd(), two_fixtures)
    assert "not_a_channel" not in snap["fixtures"]["triclinic_bulk"]["outputs"]
    # ...and asking for one by name is an error naming the fix
    with pytest.raises(KeyError, match="were not produced"):
        harness.snapshot_outputs(
            Odd(), two_fixtures, channels=["energy", "not_a_channel"]
        )


def test_register_channel_rejects_a_conflicting_redefinition():
    harness.register_channel("golden_probe_channel", harness.GRAPH_SCALAR, "eV")
    harness.register_channel("golden_probe_channel", harness.GRAPH_SCALAR, "eV")
    with pytest.raises(ValueError, match="already registered"):
        harness.register_channel("golden_probe_channel", harness.GRAPH_SCALAR, "K")
    del harness.CHANNELS["golden_probe_channel"]


def test_wrong_shape_for_a_kind_is_rejected(two_fixtures):
    class Broken(FakeSource):
        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["forces"] = out["forces"][:-1]
            return out

    with pytest.raises(ValueError, match="per_atom_vector"):
        harness.snapshot_outputs(Broken(), two_fixtures)


def test_ase_calculator_shape_is_accepted():
    """The other accepted protocol: a plain ase calculator."""
    from ase.calculators.emt import EMT  # noqa: PLC0415

    atoms = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.5]], pbc=False)
    snap = harness.snapshot_outputs(EMT(), {"cu2": atoms})
    outputs = snap["fixtures"]["cu2"]["outputs"]
    assert set(outputs) >= {"energy", "forces"}
    assert outputs["forces"]["shape"] == [2, 3]


# ---------------------------------------------------------------------------
# Serialisation and comparison
# ---------------------------------------------------------------------------


PROVENANCE = {
    "source": "a stub",
    "recipe": "tests/golden/test_harness.py",
    "description": "round-trip probe",
}


def test_reference_round_trip(tmp_path, two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    path = harness.write_reference(
        tmp_path / "probe.json", snap, provenance=PROVENANCE
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["dtype"] == "float64"
    assert on_disk["units"] == {"length": "Ang", "energy": "eV"}
    assert on_disk["provenance"]["source"] == "a stub"
    reference = harness.load_reference(path)
    harness.compare_to_reference(
        harness.snapshot_outputs(FakeSource(), two_fixtures),
        reference,
        row="fp64_cpu_reference",
    )


def test_write_reference_refuses_to_overwrite(tmp_path, two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    path = harness.write_reference(tmp_path / "p.json", snap, provenance=PROVENANCE)
    with pytest.raises(FileExistsError, match="regenerate.py"):
        harness.write_reference(path, snap, provenance=PROVENANCE)
    harness.write_reference(path, snap, provenance=PROVENANCE, allow_overwrite=True)


def test_write_reference_demands_provenance(tmp_path, two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    with pytest.raises(ValueError, match="recipe"):
        harness.write_reference(
            tmp_path / "p.json", snap, provenance={"source": "x", "description": "y"}
        )


def test_load_reference_refuses_a_foreign_schema_version(tmp_path, two_fixtures):
    snap = harness.snapshot_outputs(FakeSource(), two_fixtures)
    path = harness.write_reference(tmp_path / "p.json", snap, provenance=PROVENANCE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = harness.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        harness.load_reference(path)


def test_comparison_failure_names_fixture_channel_and_element(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    drifted = harness.snapshot_outputs(FakeSource(offset=1e-3), two_fixtures)
    with pytest.raises(AssertionError) as excinfo:
        harness.compare_to_reference(drifted, reference, row="fp64_cpu_reference")
    message = str(excinfo.value)
    assert "triclinic_bulk/energy" in message
    assert "fp64_cpu_reference" in message
    assert "edit-locked" in message


def test_a_drift_below_the_row_passes(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    drifted = harness.snapshot_outputs(FakeSource(offset=1e-9), two_fixtures)
    harness.compare_to_reference(drifted, reference, row="fp64_cpu_reference")
    # ...and the same drift is caught by nothing looser being used by default
    assert harness.FP64_CPU_REFERENCE.atol > 1e-9


def test_a_vanished_channel_fails_even_when_the_rest_matches(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    snapshot = harness.snapshot_outputs(FakeSource(), two_fixtures)
    del snapshot["fixtures"]["triclinic_bulk"]["outputs"]["forces"]
    with pytest.raises(AssertionError, match="vanished"):
        harness.compare_to_reference(snapshot, reference, row="fp64_cpu_reference")


def test_a_unit_change_fails_even_at_identical_values(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    snapshot = harness.snapshot_outputs(FakeSource(), two_fixtures)
    snapshot["fixtures"]["triclinic_bulk"]["outputs"]["dipole"]["unit"] = "e*Ang"
    with pytest.raises(AssertionError, match="unit changed"):
        harness.compare_to_reference(snapshot, reference, row="fp64_cpu_reference")


def test_a_missing_fixture_fails(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    snapshot = harness.snapshot_outputs(
        FakeSource(), harness.load_fixtures(["triclinic_bulk"])
    )
    with pytest.raises(AssertionError, match="water_cluster"):
        harness.compare_to_reference(snapshot, reference, row="fp64_cpu_reference")


def test_new_channels_pass_unless_strict(two_fixtures):
    reference = harness.snapshot_outputs(
        FakeSource(), two_fixtures, channels=["energy", "forces", "stress"]
    )
    richer = harness.snapshot_outputs(FakeSource(), two_fixtures)
    harness.compare_to_reference(richer, reference, row="fp64_cpu_reference")
    with pytest.raises(AssertionError, match="unpinned new channel"):
        harness.compare_to_reference(
            richer, reference, row="fp64_cpu_reference", strict_channels=True
        )


def test_regenerate_refuses_without_the_acknowledgement():
    """The golden-discipline guard, exercised rather than described.

    A regeneration that can happen by accident destroys the only evidence
    that a change altered nothing it should not have.
    """
    script = GOLDEN_ROOT / "regenerate.py"
    done = subprocess.run(
        [sys.executable, str(script), "--target", "fixtures"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "--i-know-what-i-am-doing" in done.stderr


def test_deviations_reports_headroom(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    drifted = harness.snapshot_outputs(FakeSource(offset=2e-7), two_fixtures)
    worst = max(d.max_abs for d in harness.deviations(drifted, reference))
    assert worst == pytest.approx(2e-7, rel=1e-6)
