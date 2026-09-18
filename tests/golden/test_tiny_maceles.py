"""The LES electrostatics family reproduces its committed references.

`MACELES` is a `ScaleShiftMACE` with a long-range head: it predicts five
latent quantities per atom, hands them to the external `les` library for an
Ewald sum, and adds the result to the energy *outside* the scale-shift. None
of that was pinned by a number before this file. What existed was two
comparisons in `tests/extensions/les/test_maceles.py`, both `xfail`ed, because
their hardcoded reference energies were generated against an unrecorded `les`
and do not reproduce against the pinned one -- a golden with no provenance,
which is worse than no golden, because it cannot say whether the model or the
solver moved.

So these references record the `les` commit, and the first thing asserted is
that the installed solver is that one. Everything else in this file assumes
it.

Two further things this file exists to hold down, both of them defects fixed
on develop while it was being written:

* `make_alpha_positive` tested `les_alpha.dim() == 2` and the isotropic
  readout emits `dim() == 1` (two 1-D index arrays collapse the advanced
  index), so the flag was a silent no-op on the default path and negative
  polarizabilities went into the Ewald sum. The anchor is configured
  isotropic on purpose, and `test_the_isotropic_polarizability_is_squared`
  compares against the same weights with the flag off.
* the calculator's padded path did not slice the per-atom LES outputs, so
  `bec`, `LES_alphas` and `LES_kappas` came back with the padding rows still
  attached.

The whole file is `les`-marked, which under `tests/conftest.py` means it skips
where the extra is absent and *fails* in the two CI jobs that promise it
(ci-extensions `les`, nightly `coverage-full`).
"""

import argparse
import json

#: The manifest is shared with every other family, so a bare load_fixtures()
#: picks up whatever the next one adds. This anchor is built on H/C/O and
#: cannot evaluate an iron structure at all.
ANCHOR_ELEMENTS = (1, 6, 8)

import numpy as np
import pytest
import torch

from tests.golden import harness, les_pin, maceles_surfaces

pytestmark = pytest.mark.les

MODEL_PATH = harness.MODELS_DIR / "tiny_maceles.model"
SIDECAR_PATH = harness.MODELS_DIR / "tiny_maceles.build.json"
LES_ARGUMENTS_PATH = harness.MODELS_DIR / "tiny_maceles.les_arguments.yaml"
MODEL_REFERENCE = harness.REFERENCES_DIR / "tiny_maceles_e3nn_cpu_fp64.json"
FIELD_REFERENCE = harness.REFERENCES_DIR / "tiny_maceles_field_cpu_fp64.json"

#: Channels the eval CLI writes and the model reference also carries. It does
#: not write `les_energy` at all, which is why the model surface is the
#: reference and this is a cross-check.
EVAL_SHARED_CHANNELS = (
    "energy",
    "forces",
    "stress",
    "latent_charges",
    "latent_dipoles",
    "latent_alphas",
    "latent_kappas",
    "latent_quads",
    "BEC",
)


@pytest.fixture(name="model", scope="module")
def fixture_model():
    return maceles_surfaces.load_anchor(MODEL_PATH)


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(elements=ANCHOR_ELEMENTS)


@pytest.fixture(name="model_reference", scope="module")
def fixture_model_reference():
    return harness.load_reference(MODEL_REFERENCE)


@pytest.fixture(name="field_reference", scope="module")
def fixture_field_reference():
    return harness.load_reference(FIELD_REFERENCE)


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [MODEL_REFERENCE, FIELD_REFERENCE])
def test_the_reference_records_the_les_it_was_taken_with(path):
    """And this test is what makes the record load bearing.

    Without it the pin is a comment: the numbers would be compared against
    whatever solver happens to be installed, and a mismatch would surface as
    eight simultaneous tolerance failures with no indication that `les` is the
    variable that moved. That is exactly the state the two xfails in
    tests/extensions/les document.
    """
    reference = harness.load_reference(path)
    recorded = reference["provenance"]["les_commit"]
    assert len(recorded) == 40 and set(recorded) <= set("0123456789abcdef")
    assert reference["metadata"]["les_commit"] == recorded

    problem = les_pin.check_les_matches(recorded)
    assert problem is None, problem
    assert les_pin.installed_les_commit() == recorded


def test_the_reference_pin_is_the_one_requirements_asks_for():
    """A golden may lag the requirement, but only deliberately.

    Regenerating is a reviewed act, so the two are allowed to differ for the
    length of one change -- but not silently and not for a release, because
    then CI installs one solver and the reference describes another.
    """
    recorded = harness.load_reference(MODEL_REFERENCE)["provenance"]["les_commit"]
    assert recorded == les_pin.pinned_les_commit(), (
        "requirements/les.txt was bumped without regenerating the LES "
        "goldens: run tests/golden/regenerate.py --target les "
        "--i-know-what-i-am-doing and review the numeric diff."
    )


def test_a_foreign_les_is_named_instead_of_producing_a_tolerance_failure():
    """The failure mode, exercised rather than described."""
    message = les_pin.check_les_matches("0" * 40)
    assert message is not None
    assert "0000000000000000000000000000000000000000" in message
    assert les_pin.installed_les_commit() in message
    assert "regenerate" in message

    # ...and a reference with no pin at all is refused outright, because it
    # would agree with every version of the solver.
    assert "reproduced" in les_pin.check_les_matches(None)


# ---------------------------------------------------------------------------
# The two references
# ---------------------------------------------------------------------------


def test_the_model_surface_reproduces_its_reference(model, fixtures, model_reference):
    """Every LES output, on every fixture, through the forward.

    `strict_channels` because this reference is the whole surface: an output
    appearing that the reference does not carry means the forward's key set
    changed, and for this family that is a thing to notice rather than to
    absorb.
    """
    snapshot = harness.snapshot_outputs(
        maceles_surfaces.ModelSurface(model),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(
        snapshot,
        model_reference,
        row=harness.FP64_CPU_REFERENCE.name,
        strict_channels=True,
    )


def test_the_field_surface_reproduces_its_reference(model, field_reference):
    """The calculator with external_field, eps_infty and keep_neutral.

    Three of those four never reach the model: they scale a Born-charge force
    correction the calculator applies after the forward returns. The fourth,
    `external_field`, does reach it -- it is written into the batch and shows
    up inside the Ewald sum as the field that induces the dipoles -- so the
    harness records it as an input channel and compares it bit for bit.
    """
    calc = maceles_surfaces.field_calculator(model)
    snapshot = harness.snapshot_outputs(
        calc,
        harness.load_fixtures(list(maceles_surfaces.FIELD_FIXTURES)),
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    for entry in snapshot["fixtures"].values():
        assert entry["inputs"]["external_field"]["value"] == pytest.approx(
            maceles_surfaces.FIELD_SETTINGS["external_field"]
        )
    harness.compare_to_reference(
        snapshot, field_reference, row=harness.FP64_CPU_REFERENCE.name
    )


def test_every_latent_quantity_is_present_and_above_the_tolerance_floor(
    model_reference,
):
    """Asserted, not merely snapshotted.

    A reference is only a constraint where its numbers are bigger than the
    tolerance it is compared at. Two of these five sit at 1e-8 with the
    library's default scales -- below the 1e-6 absolute floor of the fp64 row
    -- so an implementation returning exact zeros would have 'reproduced' them.
    The anchor uses unit scales for that reason, and this asserts the result
    rather than trusting the yaml comment that explains it.

    `isolated_atom` is exempt and stays in the reference: a single atom has no
    edges, so every readout is zero and the long-range sum is empty. What it
    pins is that the degenerate case produces zeros rather than NaNs.
    """
    latents = (
        "latent_charges",
        "latent_dipoles",
        "latent_alphas",
        "latent_kappas",
        "latent_quads",
        "BEC",
        "les_energy",
    )
    floor = harness.FP64_CPU_REFERENCE.atol
    for name, entry in model_reference["fixtures"].items():
        missing = sorted(set(latents) - set(entry["outputs"]))
        assert not missing, f"{name}: {missing} not in the reference"
        if name == "isolated_atom":
            for channel in latents:
                values = np.asarray(entry["outputs"][channel]["value"], dtype=float)
                assert np.array_equal(values, np.zeros_like(values))
            continue
        for channel in latents:
            values = np.abs(
                np.asarray(entry["outputs"][channel]["value"], dtype=float)
            )
            assert values.max() > 10 * floor, (
                f"{name}/{channel} peaks at {values.max():.3e}, which is not "
                f"meaningfully above the {floor:g} tolerance it is asserted "
                f"at -- this channel is pinned in name only"
            )


# ---------------------------------------------------------------------------
# The two positivity flags
# ---------------------------------------------------------------------------


def _forward(model, atoms, **kwargs):
    from mace.tools import torch_tools  # noqa: PLC0415

    with torch_tools.default_dtype("float64"):
        return model(maceles_surfaces.graph_for(model, atoms), **kwargs)


def test_the_isotropic_polarizability_is_squared(model, fixtures):
    """The regression this anchor is configured to catch.

    `make_alpha_positive` used to test `les_alpha.dim() == 2`, and the scalar
    alpha readout emits `dim() == 1`: it is indexed `out[num_atoms_arange,
    node_heads]`, and advanced indexing with two 1-D arrays collapses to
    [n_atoms]. So on the default isotropic path the flag did nothing and
    negative polarizabilities reached the Ewald sum. The comparison here is
    against the same weights with the flag off -- the flag changes no module,
    only the forward -- which is what makes it a statement about the fix and
    not about the checkpoint.
    """
    from tests.golden.build_maceles_anchor import (  # noqa: PLC0415
        build_model,
        load_les_arguments,
    )
    from mace.tools import torch_tools  # noqa: PLC0415

    arguments = load_les_arguments()
    assert arguments["make_alpha_positive"] is True
    assert arguments["use_anisotropic_polarizability"] is False, (
        "the anchor has to stay isotropic: the anisotropic branch is the one "
        "that always worked, so pinning it would pin the wrong thing"
    )
    unsquared = dict(arguments, make_alpha_positive=False, make_kappa_positive=False)
    with torch_tools.default_dtype("float64"):
        raw_model = build_model(unsquared).to(torch.float64)

    atoms = fixtures["water_cluster"]
    raw = _forward(raw_model, atoms, compute_force=False)["latent_alphas"]
    squared = _forward(model, atoms, compute_force=False)["latent_alphas"]
    raw = raw.detach().numpy()
    squared = squared.detach().numpy()

    assert raw.ndim == 1, "the isotropic path is what the flag used to skip"
    assert (raw < 0).any(), (
        "this fixture no longer produces negative raw polarizabilities, so it "
        "can no longer tell the squaring apart from the identity; pick "
        "another fixture rather than deleting the assertion"
    )
    assert squared == pytest.approx(raw**2, abs=harness.FP64_CPU_REFERENCE.atol)
    assert (squared >= 0).all()


def test_the_induced_charge_is_squared(model, fixtures):
    """`make_kappa_positive`, the flag whose branch was always right.

    Pinned next to its sibling because the two are configured together and a
    future refactor that unifies them has to keep both true.
    """
    from tests.golden.build_maceles_anchor import (  # noqa: PLC0415
        build_model,
        load_les_arguments,
    )
    from mace.tools import torch_tools  # noqa: PLC0415

    unsquared = dict(
        load_les_arguments(), make_alpha_positive=False, make_kappa_positive=False
    )
    with torch_tools.default_dtype("float64"):
        raw_model = build_model(unsquared).to(torch.float64)
    atoms = fixtures["water_cluster"]
    raw = _forward(raw_model, atoms, compute_force=False)["latent_kappas"]
    squared = _forward(model, atoms, compute_force=False)["latent_kappas"]
    raw = raw.detach().numpy()
    assert (raw < 0).any()
    assert squared.detach().numpy() == pytest.approx(
        raw**2, abs=harness.FP64_CPU_REFERENCE.atol
    )


# ---------------------------------------------------------------------------
# keep_neutral
# ---------------------------------------------------------------------------


def test_keep_neutral_leaves_the_reported_bec_alone_and_repeats_identically(model):
    """Two consecutive evaluations, and the flag's effect on stored state.

    The regression: on the 3-D BEC layout the neutralisation ran in place on
    the array kept in `results`, so turning the flag on silently changed the
    Born charges a caller read back while the 4-D layout was untouched -- the
    same flag meaning two different things depending on whether the model
    predicted dipoles. This anchor is on the 4-D layout, so the pair of
    assertions below is what covers it: the stored BEC is the same with the
    flag on and off, and a second call returns the identical array rather than
    a twice-neutralised one.
    """
    atoms = harness.load_fixtures(["triclinic_bulk"])["triclinic_bulk"].copy()

    neutral = maceles_surfaces.field_calculator(model)
    neutral.calculate(atoms)
    first = neutral.results["bec"].copy()
    first_forces = neutral.results["forces"].copy()
    neutral.calculate(atoms)

    assert np.array_equal(neutral.results["bec"], first)
    assert np.array_equal(neutral.results["forces"], first_forces)

    raw = maceles_surfaces.field_calculator(
        model, dict(maceles_surfaces.FIELD_SETTINGS, keep_neutral=False)
    )
    raw.calculate(atoms)
    assert np.array_equal(raw.results["bec"], first), (
        "keep_neutral changed the BEC handed to callers; it is supposed to "
        "change only the field force built from it"
    )


def test_keep_neutral_removes_exactly_a_uniform_field_force(model):
    """What the flag is *for*, as a number.

    Subtracting the mean Born charge subtracts the same vector from every
    atom's field force -- the acoustic sum rule on the field contribution --
    so the difference between the two runs must be a constant, and a non-zero
    one or the flag is doing nothing on this fixture.
    """
    atoms = harness.load_fixtures(["triclinic_bulk"])["triclinic_bulk"].copy()
    neutral = maceles_surfaces.field_calculator(model)
    raw = maceles_surfaces.field_calculator(
        model, dict(maceles_surfaces.FIELD_SETTINGS, keep_neutral=False)
    )
    neutral.calculate(atoms)
    raw.calculate(atoms)
    difference = neutral.results["forces"] - raw.results["forces"]

    spread = np.abs(difference - difference[0]).max()
    assert spread < harness.FP64_CPU_REFERENCE.atol, (
        f"keep_neutral shifted the atoms by different amounts (spread "
        f"{spread:.3e}); it should remove one uniform vector"
    )
    assert np.abs(difference[0]).max() > 10 * harness.FP64_CPU_REFERENCE.atol


def test_the_field_force_matches_the_documented_formula(model, field_reference):
    """The calculator's field correction, recomputed from the reference.

    `eps_infty` is the only knob in either reference that couples the latent
    polarizability to the forces, and it does so through a susceptibility that
    divides by the cell volume. Recomputing it from the committed BEC and
    alphas checks the chain end to end -- and would catch a change to the
    formula that left the BEC channel itself intact.
    """
    for name in maceles_surfaces.FIELD_FIXTURES:
        atoms = harness.load_fixtures([name])[name].copy()
        with_field = maceles_surfaces.field_calculator(model)
        with_field.calculate(atoms)
        entry = field_reference["fixtures"][name]["outputs"]
        expected = maceles_surfaces.bec_force_correction(
            np.asarray(entry["BEC"]["value"]).reshape(entry["BEC"]["shape"]),
            np.asarray(entry["latent_alphas"]["value"]),
            atoms.get_volume(),
            maceles_surfaces.FIELD_SETTINGS,
        )
        assert np.abs(expected).max() > 10 * harness.FP64_CPU_REFERENCE.atol

        # The correction on its own, isolated by the one knob that scales it
        # and nothing else. Dropping `external_field` instead would change the
        # model side too -- the field is in the batch and reaches the Ewald
        # sum -- and the difference would be the correction plus that
        # response, forcing a loose comparison. `electric_field_unit=0` leaves
        # the forward bit-identical.
        unscaled = maceles_surfaces.field_calculator(
            model, dict(maceles_surfaces.FIELD_SETTINGS, electric_field_unit=0.0)
        )
        unscaled.calculate(atoms)
        difference = with_field.results["forces"] - unscaled.results["forces"]
        assert difference == pytest.approx(
            expected, abs=harness.FP64_CPU_REFERENCE.atol
        )


def test_the_field_path_refuses_a_cell_that_has_no_volume(model):
    """Why the field reference is two fixtures and not six.

    With `eps_infty` set, the susceptibility divides by `atoms.get_volume()`,
    and ase refuses a volume for a cell that is not full rank. Every aperiodic
    fixture and the zero-vacuum slab therefore raise before any MACE code
    runs. Pinned as a contract so the absence is a documented limit of the
    calculator's field path rather than a gap somebody trims from the fixture
    list without noticing.
    """
    for name in ("water_cluster", "slab_zero_vacuum"):
        atoms = harness.load_fixtures([name])[name].copy()
        calc = maceles_surfaces.field_calculator(model)
        with pytest.raises(ValueError, match="volume not defined"):
            calc.calculate(atoms)


# ---------------------------------------------------------------------------
# The padded calculator path
# ---------------------------------------------------------------------------


def _padded_results(model, atoms, pad):
    calc = maceles_surfaces.field_calculator(
        model,
        {
            "external_field": None,
            "eps_infty": None,
            "keep_neutral": True,
            "electric_field_unit": 1.0,
        },
    )
    calc.pad_num_atoms = pad
    probe = atoms.copy()
    calc.calculate(probe)
    return calc.results


def test_padding_keeps_the_per_atom_les_outputs_aligned_with_the_real_atoms(
    model, fixtures
):
    """The second defect this file holds down.

    `_slice_real_outputs` knew about `forces` and `node_energy` and not about
    the LES block, so a padded evaluation returned `bec`, `LES_alphas` and
    `LES_kappas` with the padding graph's rows still attached -- more rows
    than atoms, silently misaligned against the structure.
    """
    atoms = fixtures["water_cluster"]
    plain = _padded_results(model, atoms, pad=0)
    padded = _padded_results(model, atoms, pad=len(atoms) + 5)

    for key in ("bec", "LES_alphas", "LES_kappas", "node_energy"):
        assert padded[key].shape[0] == len(atoms), (
            f"{key} came back with {padded[key].shape[0]} rows for "
            f"{len(atoms)} atoms: the padding graph is still in there"
        )
        assert padded[key] == pytest.approx(
            plain[key], abs=harness.FP64_CPU_REFERENCE.atol
        )
    assert padded["energy"] == pytest.approx(
        plain["energy"], abs=harness.FP64_CPU_REFERENCE.atol
    )


@pytest.mark.xfail(
    reason=(
        "MACELES + padding gives NaN forces on develop: the padding graph's "
        "atoms all sit at the origin, so the long-range sum over that graph "
        "is non-finite, and `get_outputs` differentiates the summed energy of "
        "the whole batch -- one non-finite graph poisons every real atom's "
        "force. Found while writing this golden; the per-atom slicing (the "
        "test above) is fixed, this is not. Pinned as an xfail rather than as "
        "an expectation of NaN, so whoever fixes it is told to delete this."
    ),
    strict=True,
)
def test_padding_should_not_poison_the_forces(model, fixtures):
    atoms = fixtures["water_cluster"]
    padded = _padded_results(model, atoms, pad=len(atoms) + 5)
    assert np.isfinite(padded["forces"]).all()


# ---------------------------------------------------------------------------
# The CLI surface: --les_arguments, --model MACELES, and the eval command
# ---------------------------------------------------------------------------


def test_the_committed_les_arguments_are_what_the_cli_would_pass():
    """The anchor's configuration is reachable from the command line.

    `--les_arguments` is `type=read_yaml`, so argparse turns the path into the
    dict that is handed to `MACELES(les_arguments=...)` verbatim
    (mace/tools/arg_parser.py:530-536, mace/tools/model_script_utils.py:431-436).
    Asserting the round trip is what makes the committed yaml the anchor's
    recipe rather than a description of it -- and it covers the two inventory
    rows this ticket owns, the flag and the model choice.
    """
    from mace.tools.arg_parser import build_default_arg_parser  # noqa: PLC0415
    from tests.golden.build_maceles_anchor import load_les_arguments  # noqa: PLC0415

    parser = build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name=probe",
            "--train_file=unused.xyz",
            "--model=MACELES",
            f"--les_arguments={LES_ARGUMENTS_PATH}",
        ]
    )
    assert args.model == "MACELES"
    assert args.les_arguments == load_les_arguments()

    model_action = next(
        action for action in parser._actions if action.dest == "model"  # noqa: SLF001
    )
    assert "MACELES" in model_action.choices


def test_the_eval_cli_lands_on_the_same_numbers_as_the_forward(model, tmp_path):
    """The third surface, cross-checked against the reference.

    `mace_eval_configs` writes its results onto the structures rather than
    returning them, renames nothing in this family, and flattens three of them
    to fit an extxyz column block -- `BEC` to (n, 18), `latent_quads` to
    (n, 9), `latent_alphas` to (n, 1), which ase then squeezes back to (n,) on
    the round trip. Those conversions live in tests/golden/eval_keys.py, and
    this is the measurement behind them: undone, the CLI's numbers are the
    forward's numbers, on the same reference.

    `les_energy` is excluded because the CLI does not write it at all -- which
    is the reason the model surface, and not this one, is the reference.
    """
    import ase.io  # noqa: PLC0415

    from mace.cli.eval_configs import run as eval_run  # noqa: PLC0415
    from mace.tools import torch_tools  # noqa: PLC0415

    fixtures = harness.load_fixtures(elements=ANCHOR_ELEMENTS)
    ase.io.write(tmp_path / "in.xyz", list(fixtures.values()), format="extxyz")
    args = argparse.Namespace(
        model=str(MODEL_PATH),
        configs=str(tmp_path / "in.xyz"),
        output=str(tmp_path / "out.xyz"),
        device="cpu",
        default_dtype="float64",
        batch_size=1,
        compute_stress=True,
        compute_bec=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        # Left off deliberately: with one structure per batch and differing
        # atom counts, eval_configs concatenates a ragged list and dies. That
        # is a defect in the CLI, unrelated to LES, and not this ticket's.
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )
    # eval_configs calls set_default_dtype and never restores it, which would
    # leak float64 into every later test in this worker.
    with torch_tools.default_dtype(torch.get_default_dtype()):
        eval_run(args)

    written = ase.io.read(tmp_path / "out.xyz", index=":", format="extxyz")
    assert len(written) == len(fixtures)
    by_name = dict(zip(fixtures, written))

    class EvalSurface:
        golden_surface = harness.SURFACE_EVAL

        def golden_outputs(self, atoms):
            return harness.collect_prefixed_outputs(
                by_name[atoms.info["golden_name"]], "MACE_"
            )

    snapshot = harness.snapshot_outputs(
        EvalSurface(),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
        channels=list(EVAL_SHARED_CHANNELS),
    )
    harness.compare_to_reference(
        snapshot,
        harness.load_reference(MODEL_REFERENCE),
        row=harness.FP64_CPU_REFERENCE.name,
        channels=list(EVAL_SHARED_CHANNELS),
    )


# ---------------------------------------------------------------------------
# The checkpoint and its sidecars
# ---------------------------------------------------------------------------


def test_the_anchor_is_a_maceles_built_from_the_committed_yaml(model):
    assert type(model).__name__ == "MACELES"
    # Forced by the class, not by the recipe: without it the last layer drops
    # its vector features and the dipole, quadrupole and polarizability
    # readouts have nothing to read (mace/modules/extensions.py:144-146).
    assert "1o" in str(model.readouts[0].linear.irreps_in)
    assert hasattr(model, "pair_repulsion"), "the anchor carries ZBL, as the others do"
    assert model.compute_bec is True
    assert model.use_dipoles and model.use_quads
    assert model.use_induced_charges and model.use_induced_dipoles
    assert model.make_alpha_positive and model.make_kappa_positive


def test_the_sidecar_records_how_the_anchor_was_built():
    from tests.golden.build_maceles_anchor import load_les_arguments  # noqa: PLC0415

    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    assert sidecar["model"] == MODEL_PATH.name
    assert sidecar["class"] == "MACELES"
    assert sidecar["dtype"] == "float64"
    assert sidecar["seed"]
    assert sidecar["command"]
    assert "regenerate.py" in sidecar["regenerate_with"]
    assert sidecar["les_arguments_file"] == LES_ARGUMENTS_PATH.name
    assert sidecar["les_arguments"] == load_les_arguments()
    assert sidecar["les_commit"] == les_pin.installed_les_commit()


@pytest.mark.parametrize("path", [MODEL_REFERENCE, FIELD_REFERENCE])
def test_the_reference_carries_dtype_units_and_provenance(path):
    reference = harness.load_reference(path)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    assert reference["units"]["energy"] == "eV"
    provenance = reference["provenance"]
    assert provenance["source"].endswith(MODEL_PATH.name)
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    assert provenance["les_arguments_file"].endswith(LES_ARGUMENTS_PATH.name)
    for entry in reference["fixtures"].values():
        for channel in entry["outputs"].values():
            assert channel["unit"]
            assert channel["kind"] in harness.KINDS


def test_the_field_reference_records_the_settings_that_are_not_channels():
    """eps_infty, keep_neutral and electric_field_unit reach no graph.

    They are configuration of the evaluation, so they travel in the metadata
    block rather than as input channels -- but they change every force in the
    file, so a reference that did not carry them could not be reproduced.
    """
    metadata = harness.load_reference(FIELD_REFERENCE)["metadata"]
    assert metadata["field_settings"] == maceles_surfaces.FIELD_SETTINGS


def test_the_checkpoint_stays_small():
    """A committed checkpoint is a permanent cost in every clone.

    Larger than the two energy anchors (about 1.65 MB against 1.07 MB) and for
    a reason: `MACELES` copies the MACE readout once per latent quantity, so
    the same backbone carries five extra readout stacks.
    """
    size_mb = MODEL_PATH.stat().st_size / 1e6
    assert size_mb < 2.0, f"{MODEL_PATH.name} is {size_mb:.2f} MB"
