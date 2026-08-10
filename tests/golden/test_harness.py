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


def _spun(moments, key="REF_magmom"):
    """A fixture carrying per-atom moments in the array a model reads."""
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.arrays[key] = np.asarray(moments, dtype=float)
    return atoms


def test_magmom_is_read_from_the_array_the_model_reads():
    """The provenance defect: ase's attribute is not the model's input.

    MagneticMACECalculator reads `atoms.arrays[magmom_key]`, defaulting to
    `REF_magmom`; `set_initial_magnetic_moments` writes a different array
    that nothing in the forward pass ever looks at. Recording the latter
    would put moments in the reference that the evaluation never used.
    """
    atoms = _spun([0.7, -0.2, 0.1] * 3)
    atoms.info["total_charge"] = -1.0
    snap = harness.snapshot_outputs(FakeSource(), {"spun": atoms})
    inputs = snap["fixtures"]["spun"]["inputs"]
    assert inputs["magmom"]["kind"] == harness.PER_ATOM_VECTOR
    assert inputs["magmom"]["shape"] == [9, 3]
    # a collinear moment is stored along z, so the schema is vector-valued
    # from the start rather than widened later
    assert inputs["magmom"]["value"][0] == [0.0, 0.0, 0.7]
    assert inputs["total_charge"]["value"] == -1.0


def test_noncollinear_moments_are_recorded_as_given():
    atoms = _spun(np.arange(27, dtype=float).reshape(9, 3))
    snap = harness.snapshot_outputs(FakeSource(), {"spun": atoms})
    assert snap["fixtures"]["spun"]["inputs"]["magmom"]["shape"] == [9, 3]


def test_ase_initial_moments_without_the_array_are_refused():
    """Silence here is what made the wrong-array bug survive review."""
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.set_initial_magnetic_moments([0.7] * 9)
    with pytest.raises(ValueError, match="REF_magmom"):
        harness.snapshot_outputs(FakeSource(), {"spun": atoms})


def test_the_two_spellings_of_a_graph_input_must_agree():
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.info["total_charge"] = -1.0
    atoms.info["charge"] = -1.0
    snap = harness.snapshot_outputs(FakeSource(), {"c": atoms})
    assert snap["fixtures"]["c"]["inputs"]["total_charge"]["value"] == -1.0
    atoms.info["charge"] = 0.0
    with pytest.raises(ValueError, match="different values"):
        harness.snapshot_outputs(FakeSource(), {"c": atoms})


# ---------------------------------------------------------------------------
# Unknown outputs
# ---------------------------------------------------------------------------


class Odd(FakeSource):
    def golden_outputs(self, atoms):
        out = super().golden_outputs(atoms)
        out["not_a_channel"] = 1.0
        return out


def test_an_undeclared_output_is_a_hard_failure(two_fixtures):
    """The regression this whole module exists to prevent, applied to itself.

    An output the schema does not know used to be dropped without a word, so
    a reference could be committed that claimed to pin a family and pinned
    three channels of it. Nothing about that failure is visible in a passing
    test run, which is why it has to be an error.
    """
    with pytest.raises(KeyError) as excinfo:
        harness.snapshot_outputs(Odd(), two_fixtures)
    message = str(excinfo.value)
    assert "not_a_channel" in message
    assert "register_channel" in message
    assert "register_alias" in message
    assert "ignore_key" in message


def test_asking_for_an_undeclared_channel_by_name_also_fails(two_fixtures):
    with pytest.raises(KeyError, match="not_a_channel"):
        harness.snapshot_outputs(
            FakeSource(), two_fixtures, channels=["energy", "not_a_channel"]
        )


def test_a_declared_channel_that_is_not_produced_still_fails(two_fixtures):
    with pytest.raises(KeyError, match="were not produced"):
        harness.snapshot_outputs(
            FakeSource(), two_fixtures, channels=["energy", "polarizability"]
        )


def test_an_allowlisted_key_is_skipped_with_its_reason_on_record(two_fixtures):
    """Ignoring is allowed, but only one key at a time and only in writing."""
    class Committee(FakeSource):
        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["energy_var"] = 1e-3
            return out

    snap = harness.snapshot_outputs(Committee(), two_fixtures)
    assert "energy_var" not in snap["fixtures"]["triclinic_bulk"]["outputs"]
    assert harness.IGNORED_KEYS["energy_var"].strip()
    with pytest.raises(ValueError, match="reason"):
        harness.ignore_key("something_else", "  ")


def test_every_allowlist_entry_carries_a_reason():
    assert harness.IGNORED_KEYS, "an empty allowlist would make the rule vacuous"
    for key, reason in harness.IGNORED_KEYS.items():
        assert len(reason.strip()) > 20, f"{key} is ignored without a real reason"


# ---------------------------------------------------------------------------
# Calculator spellings
# ---------------------------------------------------------------------------


def test_the_calculator_spellings_resolve_to_one_channel():
    """The four names measured against mace/calculators/mace.py.

    Each of these is what the calculator writes; the right-hand side is what
    the model's forward and the registry call it. Before the alias map, a LES
    or magnetic golden taken through the calculator dropped all four.
    """
    assert harness.resolve_channel("LES_alphas") == "latent_alphas"
    assert harness.resolve_channel("LES_kappas") == "latent_kappas"
    assert harness.resolve_channel("bec") == "BEC"
    assert harness.resolve_channel("MACE_magmoms") == "equilibrated_magmom"


def test_an_aliased_output_lands_under_its_channel_name(two_fixtures):
    class Latent(FakeSource):
        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["LES_alphas"] = np.ones((len(atoms), 4))
            out["MACE_magmoms"] = np.zeros((len(atoms), 3))
            return out

    outputs = harness.snapshot_outputs(Latent(), two_fixtures)
    bulk = outputs["fixtures"]["triclinic_bulk"]["outputs"]
    assert "LES_alphas" not in bulk and "MACE_magmoms" not in bulk
    assert bulk["latent_alphas"]["shape"] == [6, 4]
    assert bulk["equilibrated_magmom"]["unit"] == "muB"


def test_an_alias_cannot_shadow_or_contradict(two_fixtures):
    with pytest.raises(KeyError, match="no such channel"):
        harness.register_alias("probe_alias", "not_a_channel")
    with pytest.raises(ValueError, match="already resolves"):
        harness.register_alias("bec", "charges")
    with pytest.raises(ValueError, match="declared channel"):
        harness.register_alias("forces", "charges")
    with pytest.raises(ValueError, match="declared channel"):
        harness.ignore_key("forces", "because it would be convenient")


def test_the_calculator_key_set_is_covered():
    """Every results key mace/calculators/mace.py can write is accounted for.

    Derived from the file rather than from a remembered list: `results_map`
    plus the special cases plus the committee suffixes. A new key added to
    the calculator without a channel, an alias or an allowlist entry fails
    here rather than in whichever golden happens to touch that model family.
    """
    source = (
        Path(harness.__file__).resolve().parents[2]
        / "mace"
        / "calculators"
        / "mace.py"
    ).read_text(encoding="utf-8")
    written = set(re.findall(r"""self\.results\[["'](\w+)["']\]""", source))
    written |= {
        f"{base}{suffix}"
        for base in ("energy", "forces", "stress", "dipole")
        for suffix in ("_comm", "_var")
    }
    # the results_map left-hand column, which is assigned through a variable
    written |= set(re.findall(r"""^\s*\(\s*["'](\w+)["'],\s*["']\w+["'],""",
                              source, flags=re.MULTILINE))
    unresolved = []
    for key in sorted(written):
        try:
            harness.resolve_channel(key)
        except KeyError:
            unresolved.append(key)
    assert not unresolved, (
        "these calculator result keys resolve to nothing, so a golden taken "
        f"through the calculator would fail on them: {unresolved}"
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


def test_a_changed_input_fails_even_when_every_output_matches():
    """The defect that let a verifier rewrite every moment and still pass.

    compare_to_reference used to look only at `outputs`, so the `inputs`
    block was carried in the JSON, printed in review, and never checked. The
    schema calls magmom a pinned input; without this the claim is decoration.
    """
    atoms = _spun([0.7, -0.2, 0.1] * 3)
    reference = harness.snapshot_outputs(FakeSource(), {"spun": atoms})
    tampered = harness.snapshot_outputs(FakeSource(), {"spun": _spun([9.0] * 9)})
    # the outputs are identical -- the stub does not read the moments
    assert (
        tampered["fixtures"]["spun"]["outputs"]
        == reference["fixtures"]["spun"]["outputs"]
    )
    with pytest.raises(AssertionError) as excinfo:
        harness.compare_to_reference(tampered, reference, row="fp64_cpu_reference")
    assert "inputs/magmom" in str(excinfo.value)


def test_an_input_that_appears_or_vanishes_fails_in_both_directions():
    plain = harness.load_fixtures(["water_cluster"])["water_cluster"]
    bare = harness.snapshot_outputs(FakeSource(), {"spun": plain})
    spun = harness.snapshot_outputs(FakeSource(), {"spun": _spun([0.3] * 9)})
    with pytest.raises(AssertionError, match="appeared"):
        harness.compare_to_reference(spun, bare, row="fp64_cpu_reference")
    with pytest.raises(AssertionError, match="vanished"):
        harness.compare_to_reference(bare, spun, row="fp64_cpu_reference")


def test_an_input_comparison_cannot_be_switched_off():
    """strict_channels relaxes outputs only; inputs are never optional."""
    reference = harness.snapshot_outputs(FakeSource(), {"spun": _spun([0.7] * 9)})
    tampered = harness.snapshot_outputs(FakeSource(), {"spun": _spun([0.8] * 9)})
    for strict in (False, True):
        with pytest.raises(AssertionError, match="inputs/magmom"):
            harness.compare_to_reference(
                tampered,
                reference,
                row="fp64_cpu_reference",
                channels=["energy"],
                strict_channels=strict,
            )


# ---------------------------------------------------------------------------
# Kinds no per-atom shape can express
# ---------------------------------------------------------------------------


def test_edge_and_hessian_outputs_are_expressible(two_fixtures):
    """Two real outputs whose leading axis is not the atom count.

    edge_forces is indexed by the neighbour list and hessian by the 3N
    Cartesian degrees of freedom. Every per-atom kind pins the leading axis
    to n_atoms, so before these kinds existed the harness could only have
    dropped them -- which is the failure mode this module is supposed to
    make impossible.
    """
    class Derivatives(FakeSource):
        def golden_outputs(self, atoms):
            n = len(atoms)
            out = super().golden_outputs(atoms)
            out["edge_forces"] = np.ones((7 * n, 3))
            out["hessian"] = np.zeros((3 * n, n, 3))
            return out

    snap = harness.snapshot_outputs(Derivatives(), two_fixtures)
    bulk = snap["fixtures"]["triclinic_bulk"]["outputs"]
    assert bulk["edge_forces"]["kind"] == harness.PER_EDGE_VECTOR
    assert bulk["edge_forces"]["shape"] == [42, 3]
    assert bulk["hessian"]["kind"] == harness.HESSIAN
    assert bulk["hessian"]["shape"] == [18, 6, 3]


def test_a_changed_edge_count_still_fails_a_comparison(two_fixtures):
    """The edge count is recorded, not predicted, but it is still pinned."""
    class Edges(FakeSource):
        def __init__(self, per_atom):
            super().__init__()
            self.per_atom = per_atom

        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["edge_forces"] = np.ones((self.per_atom * len(atoms), 3))
            return out

    reference = harness.snapshot_outputs(Edges(7), two_fixtures)
    fewer = harness.snapshot_outputs(Edges(6), two_fixtures)
    with pytest.raises(AssertionError, match="shape changed"):
        harness.compare_to_reference(fewer, reference, row="fp64_cpu_reference")


def test_a_hessian_of_the_wrong_rank_is_named_not_dropped(two_fixtures):
    class Wrong(FakeSource):
        def golden_outputs(self, atoms):
            out = super().golden_outputs(atoms)
            out["hessian"] = np.zeros((len(atoms), len(atoms), 3))
            return out

    with pytest.raises(ValueError, match=r"hessian.*\(18, 6, 3\)"):
        harness.snapshot_outputs(Wrong(), two_fixtures)


def test_every_kind_has_a_shape_rule():
    for kind in harness.KINDS:
        harness.expected_shape(kind, 4)


def test_deviations_reports_headroom(two_fixtures):
    reference = harness.snapshot_outputs(FakeSource(), two_fixtures)
    drifted = harness.snapshot_outputs(FakeSource(offset=2e-7), two_fixtures)
    worst = max(d.max_abs for d in harness.deviations(drifted, reference))
    assert worst == pytest.approx(2e-7, rel=1e-6)
