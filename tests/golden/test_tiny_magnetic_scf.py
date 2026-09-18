"""The SCF-wrapped magnetic anchor reproduces its committed fixed point.

``MagneticSCFMACE`` (``mace/modules/extensions.py:1968``) is a wrapper, not a
model: it drives ``torch.optim.LBFGS`` over the magnetic moments, sets
``magmom.grad = -output["magforces"]`` by hand, and then re-evaluates the
model it wraps once at the relaxed moments. So the quantity it adds is a
*state* -- where the moments settle -- and pinning it is what gives TRN-2's
model-transform hook something to hit.

Three things are asserted here and they are deliberately different in kind:

* the fixed point itself -- ``equilibrated_magmom``, the energy and the forces
  at it -- against the committed reference, at the fp64 row;
* that it *is* a fixed point rather than an iterate, by perturbing the initial
  moments and measuring what comes back. That measurement is also what decides
  which fixtures the reference covers;
* the two behaviours the wrapper has that nothing about the numbers would
  reveal: the attribute delegation it grew so that consumers stop reaching
  through ``magmom_mace`` by hand, and the ``cache_magmom`` state that makes
  the same call return different things depending on what ran before it.

``scf_steps`` and ``scf_energy_history`` are recorded as metadata and never
asserted. They are LBFGS internals: a build that reaches the same point in one
iteration fewer changes both without moving a single physical number.
"""

import copy

import numpy as np
import pytest
import torch

from mace.modules.extensions import MagneticSCFMACE
from mace.tools import torch_tools
from tests.golden import harness
from tests.golden import magnetic_surfaces as ms

pytestmark = pytest.mark.magnetic

REFERENCE_PATH = harness.REFERENCES_DIR / "tiny_magnetic_scf_e3nn_cpu_fp64.json"

TOL = harness.FP64_CPU_REFERENCE


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return ms.scf_fixtures()


@pytest.fixture(name="all_fixtures", scope="module")
def fixture_all_fixtures():
    return ms.magnetic_fixtures()


@pytest.fixture(name="reference", scope="module")
def fixture_reference():
    return harness.load_reference(REFERENCE_PATH)


# ---------------------------------------------------------------------------
# The fixed point
# ---------------------------------------------------------------------------


def test_the_fixed_point_reproduces_its_reference(fixtures, reference):
    snapshot = harness.snapshot_outputs(
        ms.MagneticSCFForward(),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(snapshot, reference, row=TOL.name)


def test_the_reference_pins_the_state_and_only_records_the_trajectory(reference):
    """The distinction the whole second reference rests on.

    ``equilibrated_magmom``, ``energy`` and ``forces`` are outputs and are
    compared. ``scf_steps`` and ``scf_energy_history`` are metadata: present,
    dated, regenerable, and asserted by nothing. If either ever appeared in
    the outputs block the reference would start failing on optimiser
    bookkeeping, which is the failure mode that teaches people to widen
    tolerances.
    """
    for name, entry in reference["fixtures"].items():
        assert set(entry["outputs"]) == {
            "energy",
            "forces",
            "equilibrated_magmom",
        }, name
        assert set(entry["metadata"]) == {"scf_steps", "scf_energy_history"}, name
        assert entry["metadata"]["scf_steps"] == len(
            entry["metadata"]["scf_energy_history"]
        )
        assert harness.CHANNELS["scf_steps"].role == harness.ROLE_METADATA
        assert harness.CHANNELS["scf_energy_history"].role == harness.ROLE_METADATA
        assert harness.CHANNELS["equilibrated_magmom"].role == harness.ROLE_OUTPUT
    assert reference["metadata"]["model_class"] == "MagneticSCFMACE"
    assert reference["metadata"]["scf_config"]["use_scf"] is True


def test_the_relaxation_lowers_the_energy_and_flattens_the_magnetic_forces(
    fixtures, reference
):
    """What "converged" means here, measured on the committed reference.

    The moments start where the fixture put them and end where the reference
    says. Two things have to be true of that move for the second file to be
    worth having: the energy has to come down, and the thing being zeroed --
    ``dE/dm`` -- has to be much smaller at the end than at the start. Measured
    on the committed anchor, the residual falls by four to eight orders of
    magnitude (5.2 -> 7.4e-8 on the isolated atom).
    """
    model = ms.load_anchor()

    def residual_and_energy(atoms, moments):
        probe = atoms.copy()
        probe.arrays[ms.MAGMOM_KEY] = np.asarray(moments, dtype=float)
        with torch_tools.default_dtype("float64"):
            out = model(
                ms.build_batch(model, probe),
                compute_force=True,
                compute_magforces=True,
            )
        return (
            np.abs(out["magforces"].detach().numpy()).max(),
            float(out["energy"][0].detach()),
        )

    for name, atoms in fixtures.items():
        entry = reference["fixtures"][name]
        relaxed = np.asarray(entry["outputs"]["equilibrated_magmom"]["value"])
        start_residual, start_energy = residual_and_energy(
            atoms, atoms.arrays[ms.MAGMOM_KEY]
        )
        end_residual, end_energy = residual_and_energy(atoms, relaxed)
        assert end_energy < start_energy, name
        assert end_energy == pytest.approx(
            entry["outputs"]["energy"]["value"], abs=TOL.atol
        ), name
        assert end_residual < start_residual / 1e4, (
            f"{name}: the magnetic forces at the pinned state are "
            f"{end_residual:.3e}, against {start_residual:.3e} at the fixture's "
            f"own moments. That is not a relaxed state, so the reference is "
            f"pinning an iterate."
        )


def test_the_pinned_state_is_a_fixed_point_and_the_excluded_ones_are_not(
    all_fixtures,
):
    """The measurement that chose the reference's fixture list.

    LBFGS returns wherever it stopped, and "wherever it stopped" is only a
    fixed point if it is a function of where it started. Perturb the initial
    moments by 1e-9 and look at what comes back: on the three fixtures the
    reference covers, the answer moves by about 1e-9 -- the relaxation tracks
    its input, so the terminal point is determined by the model rather than by
    the arithmetic. On ``mag_fe3_canted`` and ``mag_feo_cluster`` it moves by
    1e-5, four orders of magnitude of amplification, which means two runs of
    the same code on two machines land in different places by more than the
    tolerance row allows. Those two are excluded, and this is why.

    The exclusion is asserted rather than assumed so it cannot quietly become
    wrong. If the wrapper grows a real convergence criterion -- it currently
    has none beyond LBFGS's own stalling -- these two stop being chaotic and
    this test fails, which is the signal to widen the reference.
    """
    model = ms.load_anchor()
    step = 1e-9

    def relaxed(atoms, perturbation):
        wrapper = ms.load_scf_anchor()
        batch = ms.build_batch(model, atoms)
        batch["magmom"] = batch["magmom"] + perturbation
        with torch_tools.default_dtype("float64"):
            out = wrapper(batch, compute_force=True)
        return out["equilibrated_magmom"].detach().numpy()

    amplification = {}
    for name, atoms in all_fixtures.items():
        generator = torch.Generator().manual_seed(20260810)
        moments = ms.build_batch(model, atoms)["magmom"]
        nudge = step * torch.randn(
            moments.shape, generator=generator, dtype=torch.float64
        )
        base = relaxed(atoms, torch.zeros_like(moments))
        moved = relaxed(atoms, nudge)
        amplification[name] = float(np.abs(moved - base).max()) / step

    for name in ms.SCF_REFERENCE_FIXTURES:
        assert amplification[name] < 10.0, (
            f"{name} is in the SCF reference and its relaxed moments moved by "
            f"{amplification[name]:.1f} times a 1e-9 perturbation of the "
            f"input. The pinned state has stopped being a fixed point."
        )
    excluded = set(amplification) - set(ms.SCF_REFERENCE_FIXTURES)
    assert excluded == {"mag_fe3_canted", "mag_feo_cluster"}
    for name in sorted(excluded):
        assert amplification[name] > 1e3, (
            f"{name} is excluded from the SCF reference because its relaxed "
            f"moments were not a function of the starting ones, and it now "
            f"amplifies a 1e-9 perturbation by only {amplification[name]:.1f}. "
            f"If the relaxation has become well posed, add it back."
        )


def test_the_relaxation_is_not_bounded_by_the_models_own_m_max(reference):
    """A characterisation, not a defect report, and MAG-1 needs it.

    Nothing in ``MagneticSCFMACE`` constrains the magnitude of the moments.
    The model's magnetic descriptor uses *solid* harmonics of ``m``
    (``sphericart.torch.SolidHarmonics``, mace/modules/extensions.py:1351),
    which grow as |m|^l, so the energy is polynomial in |m| and unbounded
    below outside the clamped radial term -- measured on the committed anchor,
    scaling the ferromagnetic dimer's moments by ten takes the energy from
    -9 eV to -2.9e13 eV. A relaxation therefore has no reason to stay inside
    the range the model was parameterised for, and on the ferromagnetic dimer
    it does not: it settles at 4.3 and 6.5 muB against an ``m_max`` of 4.5.

    That is pinned because it is exactly the sort of thing a rewrite would
    "fix" by clamping the moments, which would be a different model producing
    different numbers under the same name. If confinement is wanted it has to
    come from training or from an explicit constraint, and either is a change
    with a reference to regenerate.
    """
    relaxed = np.asarray(
        reference["fixtures"]["mag_fe_dimer_fm"]["outputs"]["equilibrated_magmom"][
            "value"
        ]
    )
    m_max = float(ms.load_anchor().m_max[1])
    assert np.linalg.norm(relaxed, axis=-1).max() > m_max, (
        "the ferromagnetic dimer's relaxed moments now stay inside m_max. "
        "Either the anchor changed or something started constraining the "
        "relaxation; both are reference-regenerating events."
    )


def test_the_collinear_variant_moves_the_moments_only_along_z(fixtures):
    """The single recorded ``use_collinear`` case.

    With ``use_collinear=True`` the wrapper zeroes the transverse components
    of the gradient before LBFGS sees them (mace/modules/extensions.py:
    2065-2070), so a relaxation started from moments along z stays exactly
    along z -- exactly, not approximately, because the components are never
    given a nonzero step. Off, the same structure is free to cant, and on the
    dimer it does not because the arrangement is symmetric; the trimer, which
    starts non-collinear, is where the difference is visible. One case, as the
    ticket scopes it: the reference itself is taken with the free variant.
    """
    atoms = fixtures["mag_fe_dimer_fm"]
    model = ms.load_anchor()
    wrapper = ms.load_scf_anchor(use_collinear=True)
    with torch_tools.default_dtype("float64"):
        out = wrapper(ms.build_batch(model, atoms), compute_force=True)
    relaxed = out["equilibrated_magmom"].detach().numpy()
    assert np.array_equal(relaxed[:, :2], np.zeros_like(relaxed[:, :2]))
    assert np.abs(relaxed[:, 2]).min() > 0.0


# ---------------------------------------------------------------------------
# The wrapper's two stateful behaviours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attribute",
    ["heads", "atomic_numbers", "r_max", "num_interactions", "atomic_energies_fn"],
)
def test_the_wrapper_answers_for_the_model_it_wraps(attribute):
    """Attribute delegation, on the committed checkpoint.

    Without it every consumer has to know to reach through ``magmom_mace``,
    and most do not: ``mace_eval_configs`` failed for every SCF checkpoint,
    ``create_lammps_model`` and ``select_head`` read ``model.heads``, and
    fine-tuning reads ``interactions`` and ``atomic_energies_fn``. The
    behavioural suite pins this on a throwaway model; here it is pinned on the
    artifact the goldens are taken from, because a checkpoint that cannot be
    introspected is a checkpoint MAG-1 cannot convert.
    """
    wrapper = ms.load_scf_anchor()
    inner = wrapper.magmom_mace
    assert hasattr(wrapper, attribute)
    got = getattr(wrapper, attribute)
    want = getattr(inner, attribute)
    if torch.is_tensor(got):
        assert torch.equal(got, want)
    else:
        assert got is want
    # ...and `forward` stays the wrapper's own, which is the whole point of
    # delegating through __getattr__ rather than through inheritance.
    assert type(wrapper).forward is MagneticSCFMACE.forward


def test_the_wrapper_is_not_convertible_and_the_delegation_does_not_hide_it():
    """Delegation stops at the class name, which is what saves this.

    ``extract_config_mace_model`` whitelists by class, and the wrapper is not
    on the list, so it comes back as an error payload -- the same contract the
    plain-MACE anchor pins in ``test_tiny_anchors.py``. It is worth an
    assertion of its own here because ``__getattr__`` delegation is exactly
    the mechanism that could have made this *succeed*: every attribute the
    extractor reads resolves through to ``magmom_mace``, so a whitelist keyed
    on anything but the type would have produced a config describing the inner
    model and labelled it the wrapper. TRN-2's model-transform hook has to
    know that an SCF checkpoint is unwrappable before it is convertible.
    """
    from mace.tools.scripts_utils import extract_config_mace_model  # noqa: PLC0415

    refused = extract_config_mace_model(ms.load_scf_anchor())
    assert isinstance(refused, dict) and "error" in refused
    accepted = extract_config_mace_model(ms.load_anchor())
    assert "error" not in accepted
    assert accepted["use_magmom_one_body"] is True


def test_cache_magmom_supplies_the_moments_when_the_batch_has_none(fixtures):
    """The state that makes two identical calls return different things.

    ``forward`` takes the moments from ``data["magmom"]`` when the batch has
    them and from ``self.cache_magmom`` when it does not
    (mace/modules/extensions.py:2018-2026), and it writes the relaxed moments
    into that cache on the way out. So a second call with the moments removed
    does not fail -- it silently continues from where the first one stopped,
    and returns the fixed point of the first call rather than of the fixture.

    Pinned because it is a property of the wrapper that a reference cannot
    show: every fixture here carries its own moments, precisely so that the
    goldens do not depend on the order they were evaluated in.
    """
    atoms = fixtures["mag_fe_dimer_fm"]
    model = ms.load_anchor()
    wrapper = ms.load_scf_anchor()
    assert wrapper.cache_magmom is None

    with torch_tools.default_dtype("float64"):
        first = wrapper(ms.build_batch(model, atoms), compute_force=True)
    relaxed = first["equilibrated_magmom"].detach().numpy()
    assert wrapper.cache_magmom is not None
    assert np.array_equal(wrapper.cache_magmom.numpy(), relaxed)

    without = ms.build_batch(model, atoms)
    del without["magmom"]
    with torch_tools.default_dtype("float64"):
        second = wrapper(without, compute_force=True)
    # It started from the first call's answer, so it is already there.
    assert np.abs(
        second["equilibrated_magmom"].detach().numpy() - relaxed
    ).max() < TOL.atol
    assert second["scf_steps"] < first["scf_steps"]


def test_a_fresh_wrapper_with_no_moments_refuses(fixtures):
    """The other half of the same branch, and the one that has to be loud.

    No moments in the batch and no cache is not a state the wrapper can
    invent its way out of: there is nothing to relax. It raises, and it says
    so, rather than defaulting to zeros -- which would be a valid-looking
    calculation of a nonmagnetic system wearing a magnetic model's name.
    """
    model = ms.load_anchor()
    wrapper = ms.load_scf_anchor()
    batch = ms.build_batch(model, fixtures["mag_fe_dimer_fm"])
    del batch["magmom"]
    with pytest.raises(ValueError, match="No initial magnetic moment provided"):
        wrapper(batch, compute_force=True)


# ---------------------------------------------------------------------------
# The two refusals develop added
# ---------------------------------------------------------------------------


def test_the_calculator_refuses_a_hessian_for_an_scf_wrapped_model(fixtures):
    """A hessian of what, exactly.

    ``MagneticSCFMACE.forward`` takes no ``compute_hessian``, so before this
    refusal ``get_hessian`` died on a bare ``TypeError`` from inside the
    forward. Falling through to ``magmom_mace`` would have been worse than the
    TypeError: the inner model's hessian holds the moments fixed, so it drops
    the term coming from their relaxation, ``dm*/dr``, and is the second
    derivative of a different energy than the one this calculator reports.

    Pinned here on the committed checkpoint, and pinned as a *refusal* --
    the plain anchor still answers, which is what says the refusal is scoped
    to the wrapper rather than to the family.
    """
    from mace.calculators import MagneticMACECalculator  # noqa: PLC0415

    atoms = fixtures["mag_fe_dimer_fm"].copy()
    wrapped = MagneticMACECalculator(
        models=[ms.load_scf_anchor()], device="cpu", default_dtype="float64",
        magmom_key=ms.MAGMOM_KEY,
    )
    with pytest.raises(NotImplementedError, match="SCF-wrapped magnetic models"):
        wrapped.get_hessian(atoms=atoms)

    plain = MagneticMACECalculator(
        models=[ms.load_anchor()], device="cpu", default_dtype="float64",
        magmom_key=ms.MAGMOM_KEY,
    )
    hessian = plain.get_hessian(atoms=atoms)
    assert np.isfinite(hessian).all()
    assert hessian.shape == (3 * len(atoms), len(atoms), 3)


def test_the_eval_cli_refuses_magforces_for_an_scf_wrapped_model(fixtures, tmp_path):
    """``--return_magforces`` against a wrapper, which cannot honour it.

    The wrapper's forward genuinely takes no ``compute_magforces``, and
    ``eval_configs`` says so before it does any work instead of letting the
    call raise a ``TypeError`` from inside the model. The same file evaluates
    an SCF checkpoint perfectly well *without* the flag, which is the part
    that makes this a refusal rather than a limitation of the CLI, and the
    plain anchor's magforces are pinned in test_tiny_magnetic.py.
    """
    import argparse  # noqa: PLC0415

    import ase.io  # noqa: PLC0415

    from mace.cli.eval_configs import run  # noqa: PLC0415

    model_path = tmp_path / "scf.model"
    torch.save(ms.load_scf_anchor(), model_path)
    configs = tmp_path / "in.xyz"
    probe = fixtures["mag_fe_dimer_fm"].copy()
    probe.info.pop("golden_name", None)
    ase.io.write(configs, probe, format="extxyz")

    def make_args(return_magforces):
        return argparse.Namespace(
            model=str(model_path),
            configs=str(configs),
            output=str(tmp_path / f"out_{return_magforces}.xyz"),
            device="cpu",
            default_dtype="float64",
            batch_size=1,
            compute_stress=False,
            compute_bec=False,
            enable_cueq=False,
            return_contributions=False,
            return_descriptors=False,
            return_node_energies=False,
            return_magforces=return_magforces,
            info_prefix="MACE_",
            head=None,
            magmom_key=ms.MAGMOM_KEY,
        )

    with torch_tools.default_dtype("float64"):
        with pytest.raises(ValueError, match="--return_magforces is not supported"):
            run(make_args(True))
        run(make_args(False))

    written = ase.io.read(tmp_path / "out_False.xyz", index=0, format="extxyz")
    assert "MACE_energy" in written.info
    assert "MACE_magforces" not in written.arrays


def test_the_wrapper_is_deep_copyable_and_its_cache_travels(fixtures):
    """Delegation through ``__getattr__`` is easy to get subtly wrong.

    ``nn.Module.__getattr__`` runs first so parameters and submodules resolve
    normally, and ``_modules`` is read out of ``__dict__`` because during
    unpickling the delegation can fire before it exists and ``self.magmom_mace``
    would recurse forever. A deep copy takes the same path, so it is the
    cheapest way to keep that from regressing -- and a checkpoint that cannot
    be copied is a checkpoint that cannot be converted either.
    """
    wrapper = ms.load_scf_anchor()
    model = ms.load_anchor()
    with torch_tools.default_dtype("float64"):
        wrapper(ms.build_batch(model, fixtures["mag_fe_atom"]), compute_force=True)
    clone = copy.deepcopy(wrapper)
    assert torch.equal(clone.cache_magmom, wrapper.cache_magmom)
    assert torch.equal(clone.atomic_numbers, wrapper.atomic_numbers)
    assert clone.heads == wrapper.heads
