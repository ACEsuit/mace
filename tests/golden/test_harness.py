"""Tests for the golden harness itself.

Everything here runs on numpy/ase/json only -- no model, no torch -- because
the harness is the one piece of machinery both the current stack and its
replacement have to share, and a test that needed the current stack to prove
that would be proving the opposite.
"""

import ast
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
        # not selectable for outputs -- nothing reproduces bit for bit across
        # machines -- but it is what the recorded inputs are always compared
        # at, and it belongs in the one table rather than beside it
        "exact",
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
# Which array the input actually came from
#
# The defect the block above fixed was "the harness recorded the wrong array".
# It survives being fixed, in a quieter form: the *right* array is named by a
# constructor argument, and a table of literal defaults cannot know what a
# given instance was built with. A structure whose moments live under another
# key matches nothing, is recorded as no magmom at all, and then compares
# clean -- there is nothing on either side to disagree about.
# ---------------------------------------------------------------------------


def test_a_non_default_magmom_key_is_recorded_not_missed():
    """The parameterised form of the same silence."""
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.set_array("spins_from_dft", np.full(9, 1.3))

    plain = harness.snapshot_outputs(FakeSource(), {"spun": atoms})
    assert "magmom" not in plain["fixtures"]["spun"]["inputs"], (
        "nothing knows about this array yet, so this is the silent case"
    )

    class Configured(FakeSource):
        """As MagneticMACECalculator(magmom_key=...) presents itself."""

        magmom_key = "spins_from_dft"

    probed = harness.snapshot_outputs(Configured(), {"spun": atoms})
    assert probed["fixtures"]["spun"]["inputs"]["magmom"]["value"][0] == [
        0.0,
        0.0,
        1.3,
    ]

    explicit = harness.snapshot_outputs(
        FakeSource(), {"spun": atoms}, input_keys={"magmom": ["spins_from_dft"]}
    )
    assert explicit["fixtures"]["spun"]["inputs"] == (
        probed["fixtures"]["spun"]["inputs"]
    )


def test_a_non_default_charges_key_is_recorded_too():
    """`charges_key` defaults to "Qs" on both calculators, and the reference
    charges it names are a real input: the fixed-charge dipole baseline is
    computed from them."""
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.set_array("Qs", np.linspace(-0.4, 0.4, 9))

    class Configured(FakeSource):
        charges_key = "Qs"

    snap = harness.snapshot_outputs(Configured(), {"q": atoms})
    recorded = snap["fixtures"]["q"]["inputs"]["input_charges"]
    assert recorded["shape"] == [9]
    assert recorded["value"][0] == pytest.approx(-0.4)
    # and it is not confused with the `charges` a model predicts
    assert "charges" not in snap["fixtures"]["q"]["inputs"]
    assert "charges" in snap["fixtures"]["q"]["outputs"]


def test_the_probed_key_replaces_the_default_rather_than_joining_it():
    """Both arrays present, holding different moments.

    A configured key is not a synonym for the default -- it is the answer to
    which array the evaluation read. A stale `REF_magmom` on the same
    structure is an array nothing looked at, not a second opinion, so this
    must record 2.5 rather than raising the "present under both with
    different values" error that genuine synonyms get.
    """
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    atoms.set_array("REF_magmom", np.full(9, 0.5))
    atoms.set_array("other_magmom", np.full(9, 2.5))

    class Configured(FakeSource):
        magmom_key = "other_magmom"

    snap = harness.snapshot_outputs(Configured(), {"spun": atoms})
    assert snap["fixtures"]["spun"]["inputs"]["magmom"]["value"][0][2] == 2.5
    # ... while two spellings of the *same* input still have to agree
    atoms.info["total_charge"] = -1.0
    atoms.info["charge"] = 0.0
    with pytest.raises(ValueError, match="different values"):
        harness.snapshot_outputs(Configured(), {"spun": atoms})


def test_input_keys_must_name_an_input_channel():
    atoms = harness.load_fixtures(["water_cluster"])["water_cluster"]
    with pytest.raises(KeyError, match="not a declared input channel"):
        harness.snapshot_outputs(
            FakeSource(), {"w": atoms}, input_keys={"forces": ["REF_forces"]}
        )


def test_a_probe_must_name_an_input_channel():
    with pytest.raises(KeyError, match="not a declared input channel"):
        harness.register_input_probe("forces", attribute="forces_key", store="arrays")
    with pytest.raises(ValueError, match="arrays"):
        harness.register_input_probe("magmom", attribute="magmom_key", store="elsewhere")


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


REPO_ROOT = Path(harness.__file__).resolve().parents[2]


def _calculator_result_keys():
    """Every key mace/calculators/mace.py can write into `self.results`.

    Derived from the file rather than from a remembered list: `results_map`
    plus the special cases plus the committee suffixes.
    """
    source = (REPO_ROOT / "mace" / "calculators" / "mace.py").read_text(
        encoding="utf-8"
    )
    written = set(re.findall(r"""self\.results\[["'](\w+)["']\]""", source))
    written |= {
        f"{base}{suffix}"
        for base in ("energy", "forces", "stress", "dipole")
        for suffix in ("_comm", "_var")
    }
    # the results_map left-hand column, which is assigned through a variable
    written |= set(re.findall(r"""^\s*\(\s*["'](\w+)["'],\s*["']\w+["'],""",
                              source, flags=re.MULTILINE))
    return written


def _forward_output_keys():
    """Every key a model class's `forward` can put in its return dict.

    Parsed, not grepped, because the models build that dict three different
    ways -- a `return {...}` literal (MACE, ScaleShiftMACE, MACELES,
    PolarMACE), a local dict that is returned by name (AtomicDipolesMACE,
    AtomicDielectricMACE, EnergyDipolesMACE, MagneticScaleShiftMACE), and
    subscript assignment onto a dict returned from a nested call
    (MagneticSCFMACE, which adds the three SCF keys to what its inner model
    produced). A regex over any one of those three misses the other two.

    Nested function bodies are skipped: MagneticSCFMACE's LBFGS `closure`
    returns a tensor named `energy`, and walking into it would make every
    `energy = ...` assignment in the enclosing forward look like an output
    dict.

    Returns a mapping class name -> the keys its forward can emit, so a
    failure can say which family diverged.
    """
    per_class = {}
    for relative in ("mace/modules/models.py", "mace/modules/extensions.py"):
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for fn in node.body:
                if not isinstance(fn, ast.FunctionDef) or fn.name != "forward":
                    continue
                keys = _dict_keys_returned_by(fn)
                if keys:
                    per_class[f"{relative}::{node.name}"] = keys
    return per_class


def _walk_skipping_nested_functions(fn):
    """Every node in `fn`'s own body, not in any function defined inside it."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            stack.append(child)


def _dict_keys_returned_by(fn):
    nodes = list(_walk_skipping_nested_functions(fn))
    returned_names = {
        node.value.id
        for node in nodes
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
    }
    keys = set()
    for node in nodes:
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys |= {
                k.value for k in node.value.keys if isinstance(k, ast.Constant)
            }
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id in returned_names
            and isinstance(node.value, ast.Dict)
        ):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id in returned_names
            and isinstance(target.slice, ast.Constant)
        ):
            keys.add(target.slice.value)
    return keys


def test_the_key_extractors_still_find_the_families_they_are_meant_to():
    """A guard on the guard.

    Both derivations are static analysis of somebody else's file, and the
    failure mode that matters is not a false alarm -- it is finding nothing
    and reporting coverage. A regex that stops matching, or an AST walk that
    stops recognising how a forward builds its dict, would make the coverage
    test below pass vacuously.
    """
    calculator = _calculator_result_keys()
    assert len(calculator) >= 25, sorted(calculator)
    assert {"energy", "forces", "stresses", "virials"} <= calculator

    per_class = _forward_output_keys()
    # one class per way of building the return dict, so a change to any of
    # the three shows up here
    for name in (
        "mace/modules/models.py::MACE",  # return {...}
        "mace/modules/models.py::AtomicDielectricMACE",  # output = {...}; return output
        "mace/modules/extensions.py::PolarMACE",
        "mace/modules/extensions.py::MagneticSCFMACE",  # out["k"] = ...
    ):
        assert name in per_class, sorted(per_class)
    assert "scf_steps" in per_class["mace/modules/extensions.py::MagneticSCFMACE"]
    assert "edge_forces" in per_class["mace/modules/models.py::MACE"]
    everything = set().union(*per_class.values())
    assert len(everything) >= 40, sorted(everything)


def test_the_calculator_key_set_is_covered():
    """Every results key mace/calculators/mace.py can write is accounted for.

    A new key added to the calculator without a channel, an alias or an
    allowlist entry fails here rather than in whichever golden happens to
    touch that model family.
    """
    unresolved = []
    for key in sorted(_calculator_result_keys()):
        try:
            harness.resolve_channel(key, harness.SURFACE_CALCULATOR)
        except KeyError:
            unresolved.append(key)
    assert not unresolved, (
        "these calculator result keys resolve to nothing, so a golden taken "
        f"through the calculator would fail on them: {unresolved}"
    )


def test_the_model_forward_key_set_is_covered():
    """The other half of the surface, and the half that was missing.

    The alias map and its guard were both derived from
    mace/calculators/mace.py alone, so all 31 calculator keys resolved while
    13 of the 43 forward keys resolved to nothing: atomic_stresses,
    atomic_virials, displacement, node_feats, contributions, atomic_dipoles,
    dmu_dr, dalpha_dr, spin_density, charges_history, scf_energy_history,
    electrostatic_potentials and fermi_level.

    That gap is not academic for the goldens still to be written.
    `edge_forces` and `hessian` are returned by every energy model and by no
    calculator, so any golden that wants either has to go through the
    `golden_outputs` hook -- and a hook hands the harness the forward's whole
    dict, all 43 keys, not the calculator's 31. The first such golden would
    have died on a key the schema had never been shown.
    """
    unresolved = {}
    for owner, keys in sorted(_forward_output_keys().items()):
        for key in sorted(keys):
            try:
                harness.resolve_channel(key, harness.SURFACE_MODEL)
            except KeyError:
                unresolved.setdefault(owner, []).append(key)
    assert not unresolved, (
        "these model forward keys resolve to nothing, so a golden taken "
        f"through golden_outputs would fail on them: {unresolved}"
    )


def test_neither_surface_is_reachable_only_through_the_other():
    """The structural version of the same point.

    Coverage of one surface is not coverage of the other, and the way that
    stayed invisible was that nothing ever asked. These two sets genuinely
    differ in both directions, so a future test that checks one and calls it
    done is checking about two thirds of the schema.
    """
    calculator = _calculator_result_keys()
    model = set().union(*_forward_output_keys().values())
    assert {"edge_forces", "hessian"} <= model - calculator, (
        "if these became reachable through the calculator, the case for the "
        "golden_outputs hook would need restating"
    )
    assert {"energies", "free_energy"} <= calculator - model


def test_a_spelling_may_mean_different_things_on_the_two_surfaces():
    """`virials` is the collision, and both readings have to survive.

    The model's forward returns the graph-level virial under this name
    (mace/modules/models.py:433, shape (n_graphs, 3, 3)); the calculator
    returns the per-atom one (mace/calculators/mace.py:729-733, shape
    (n_atoms, 3, 3)) and has no key at all for the graph virial. A single
    spelling->channel map has to pick one and mis-shape the other.
    """
    assert harness.resolve_channel("virials", harness.SURFACE_MODEL) == "virials"
    assert (
        harness.resolve_channel("virials", harness.SURFACE_CALCULATOR)
        == "atomic_virials"
    )
    assert harness.CHANNELS["virials"].kind == harness.GRAPH_TENSOR
    assert harness.CHANNELS["atomic_virials"].kind == harness.PER_ATOM_TENSOR
    # a surface-scoped alias that shadows a declared channel has to say so
    with pytest.raises(ValueError, match="collision"):
        harness.register_alias("forces", "charges", surface=harness.SURFACE_MODEL)


def test_one_quantity_with_two_layouts_stays_one_channel():
    """Per-atom stress: (n_atoms, 6) from the calculator, (n_atoms, 3, 3) from
    the model, and exactly one channel holding it.

    Registering a channel each would have been the silent split -- both would
    hold the same physics and no comparison would ever put them side by side.
    """
    assert (
        harness.resolve_channel("stresses", harness.SURFACE_CALCULATOR)
        == "atomic_stresses"
    )
    assert harness.CHANNELS["atomic_stresses"].kind == harness.PER_ATOM_TENSOR
    assert "stresses" not in harness.CHANNELS

    n_atoms = 4
    rng = np.random.default_rng(0)
    full = rng.normal(size=(n_atoms, 3, 3))
    full = 0.5 * (full + full.transpose(0, 2, 1))
    voigt = np.stack(
        [full[:, 0, 0], full[:, 1, 1], full[:, 2, 2],
         full[:, 1, 2], full[:, 0, 2], full[:, 0, 1]],
        axis=-1,
    )
    assert np.array_equal(harness.voigt_6_to_full_3x3(voigt), full)

    class Calc:
        golden_surface = harness.SURFACE_CALCULATOR

        def golden_outputs(self, atoms):
            return {"energy": 1.0, "stresses": voigt[: len(atoms)]}

    class Model:
        golden_surface = harness.SURFACE_MODEL

        def golden_outputs(self, atoms):
            return {"energy": 1.0, "atomic_stresses": full[: len(atoms)]}

    atoms = {"probe": Atoms("H4", positions=np.eye(4, 3), pbc=False)}
    from_calc = harness.snapshot_outputs(Calc(), atoms)
    from_model = harness.snapshot_outputs(Model(), atoms)
    assert (
        from_calc["fixtures"]["probe"]["outputs"]["atomic_stresses"]
        == from_model["fixtures"]["probe"]["outputs"]["atomic_stresses"]
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


def test_inputs_are_compared_exactly_not_at_the_output_row():
    """A perturbation inside the output tolerance is still a different input.

    Inputs were compared at whichever row the outputs used, so at the fp32
    row a 2e-3 nudge to a moment sat inside the bound and passed. The moment
    below is 2.2 muB -- bcc iron, not a contrived magnitude -- and the fp32
    row is relative, so its bound there is 5e-5 + 1e-3 * 2.2 = 2.25e-3. A
    2e-3 change to a magnetic moment is a physically different structure, and
    it fitted underneath.

    The argument behind the output rows does not transfer to an input: an
    input is read verbatim off the committed fixture rather than computed, so
    two reads either agree exactly or the fixture changed.
    """
    moment = 2.2
    nudge = 2e-3
    reference = harness.snapshot_outputs(FakeSource(), {"spun": _spun([moment] * 9)})
    nudged = harness.snapshot_outputs(
        FakeSource(), {"spun": _spun([moment + nudge] * 9)}
    )

    fp32 = harness.TOLERANCES["fp32"]
    assert nudge < fp32.atol + fp32.rtol * moment, (
        "this test is only meaningful while the nudge is inside the fp32 row"
    )
    for row in sorted(harness.TOLERANCES):
        with pytest.raises(AssertionError, match="inputs/magmom"):
            harness.compare_to_reference(nudged, reference, row=row)


def test_the_exact_row_is_declared_and_is_actually_exact():
    assert harness.EXACT.atol == 0.0 and harness.EXACT.rtol == 0.0
    assert harness.TOLERANCES["exact"] is harness.EXACT
    # the smallest possible perturbation still fails
    tiny = 0.7 + np.spacing(0.7)
    reference = harness.snapshot_outputs(FakeSource(), {"spun": _spun([0.7] * 9)})
    nudged = harness.snapshot_outputs(FakeSource(), {"spun": _spun([tiny] * 9)})
    with pytest.raises(AssertionError, match="inputs/magmom"):
        harness.compare_to_reference(nudged, reference, row="fp64_cpu_reference")


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
