import argparse
from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace.calculators import MACECalculator
from mace.cli.eval_configs import run as mace_eval_configs_run
from mace.cli.run_train import run as mace_run
from mace.modules import interaction_classes
from mace.modules.models import ScaleShiftMACE
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype
from mace.tools.utils import AtomicNumberTable
from mace.tools.model_script_utils import load_foundations_elements


# -----------------------------------------------------------------------------
# Helpers & fixtures
# -----------------------------------------------------------------------------


def _attach_field_response(atoms, energy=0.0, field=(0.01, 0.0, 0.0), seed=0):
    """Attach minimal field-response data to an Atoms object using REF_* keys."""
    rng = np.random.default_rng(seed)
    n = len(atoms)

    forces = rng.normal(scale=0.01, size=(n, 3))
    stress6 = np.zeros(6, dtype=float)
    virials6 = np.zeros(6, dtype=float)

    polarization = rng.normal(scale=0.1, size=3)
    becs = rng.normal(scale=0.1, size=(n, 9))          # (natoms, 9) flattened 3×3
    polarizability = rng.normal(scale=0.05, size=9)    # flattened 3×3

    atoms.info["REF_energy"] = float(energy)
    atoms.info["REF_stress"] = stress6
    atoms.info["REF_virials"] = virials6
    atoms.info["REF_electric_field"] = np.array(field, dtype=float)
    atoms.info["REF_polarization"] = polarization
    atoms.info["REF_polarizability"] = polarizability
    atoms.info["head"] = "Default"

    atoms.arrays["REF_forces"] = forces
    atoms.arrays["REF_becs"] = becs

    return atoms


@pytest.fixture
def field_fitting_configs():
    """Small synthetic dataset carrying all field-related keys."""
    cfgs = []

    cell = (6.0, 6.0, 6.0)
    geom = [
        (0.0, 0.0, 0.0),
        (0.96, 0.0, 0.0),
        (-0.32, 0.94, 0.0),
    ]

    fields = [
        (0.01, 0.00, 0.00),
        (-0.02, 0.01, 0.00),
        (0.00, -0.01, 0.02),
        (0.03, -0.02, 0.01),
    ]

    for i, ef in enumerate(fields):
        at = Atoms("OH2", positions=geom, cell=cell, pbc=True)
        _attach_field_response(
            at,
            energy=-10.0 + 0.1 * i,
            field=ef,
            seed=i,
        )
        cfgs.append(at)

    return cfgs


def _common_train_args(tmp_path: Path, xyz: Path, name: str = "MACEField-mini"):
    """Build an argparse.Namespace mimicking CLI args for a small MACEField run."""
    parser = build_default_arg_parser()

    arg_list = [
        f"--name={name}",
        "--model=MACEField",
        "--loss=universal_field",
        f"--train_file={xyz}",
        "--valid_fraction=0.25",
        "--device=cpu",
        "--default_dtype=float32",
        "--compute_stress=True",
        "--compute_forces=True",
        "--compute_polarization=True",
        "--compute_becs=True",
        "--compute_polarizability=True",
        "--batch_size=2",
        "--valid_batch_size=2",
        "--max_num_epochs=1",
        "--seed=7",
        f"--model_dir={tmp_path}",
        f"--checkpoints_dir={tmp_path}",
        f"--log_dir={tmp_path}",
        f"--results_dir={tmp_path}",
        "--work_dir=.",
        "--E0s={1: 0.0, 8: 0.0}",
        "--multiheads_finetuning=False",
        "--foundation_filter_elements=False",
        "--plot=False",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--stress_key=REF_stress",
        "--virials_key=REF_virials",
        "--electric_field_key=REF_electric_field",
        "--polarization_key=REF_polarization",
        "--becs_key=REF_becs",
        "--polarizability_key=REF_polarizability",
        "--head_key=head",
    ]

    args = parser.parse_args(arg_list)
    return args


def _locate_trained_model(tmp_path: Path, name: str, seed: int | None = None) -> Path:
    """Locate a trained model file for a given run name.

    Handles both compiled and tagged naming schemes used by MACE.
    """
    compiled = tmp_path / f"{name}_compiled.model"
    if compiled.exists():
        return compiled

    if seed is not None:
        tagged = tmp_path / f"{name}_run-{seed}.model"
        if tagged.exists():
            return tagged

    tagged_candidates = sorted(tmp_path.glob(f"{name}_run-*.model"))
    if tagged_candidates:
        return tagged_candidates[-1]

    fallback = tmp_path / f"{name}.model"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find trained model for name={name} in {tmp_path}")


def _make_macefield_calculator(
    model_path: Path,
    *,
    device: str = "cpu",
    default_dtype_str: str = "float32",
    head: str | None = None,
) -> MACECalculator:
    """Create a MACEField calculator with ASE state checking disabled.

    Disabling check_state avoids NumPy dtype promotion issues with Atoms.info
    in these tests; the calculator recomputes for every call.
    """
    calc = MACECalculator(
        model_paths=[str(model_path)],
        device=device,
        default_dtype=default_dtype_str,
        model_type="MACEField",
        head=head,
    )

    def _no_check_state(atoms):
        return []

    calc.check_state = _no_check_state
    return calc


# -----------------------------------------------------------------------------
# Plain MACE model fixture (for error-path tests)
# -----------------------------------------------------------------------------


MODEL_CONFIG = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=5,
    num_elements=2,
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=np.zeros(2),
    avg_num_neighbors=8,
    atomic_numbers=[1, 8],
    correlation=3,
    radial_type="bessel",
    atomic_inter_shift=0.0,
    atomic_inter_scale=1.0,
)


@pytest.fixture(name="mace_model_path")
def mace_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a standard ScaleShiftMACE model (no field)."""
    with default_dtype(torch.float32):
        model = ScaleShiftMACE(**MODEL_CONFIG)
        path = tmp_path / "mace.model"
        torch.save(model, path)
    return path


# -----------------------------------------------------------------------------
# Smoke tests
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_macefield_train_cli_smoke(tmp_path, field_fitting_configs):
    """End-to-end training run through the CLI and compilation."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_macefield_calculator_smoke(tmp_path, field_fitting_configs):
    """Check MACECalculator can be built and returns field-aware properties."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()

    calc = _make_macefield_calculator(model_path, default_dtype_str="float32")

    at = field_fitting_configs[0].copy()
    at.calc = calc

    e = at.get_potential_energy()
    f = at.get_forces()
    s = at.get_stress()

    assert e is not None
    assert f.shape == (len(at), 3)
    assert s.shape == (6,)

    results = at.calc.results
    assert "polarization" in results
    assert "becs" in results
    assert "polarizability" in results

    pol = results["polarization"]
    becs = results["becs"]
    polz = results["polarizability"]

    assert np.asarray(pol).shape[-1] == 3
    assert becs.ndim == 2 and becs.shape[1] == 9
    assert np.asarray(polz).shape[-1] == 9


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_universal_field_loss_loading_and_eval_cli(tmp_path, field_fitting_configs):
    """End-to-end check of eval_configs with MACEField-related flags."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()

    out = tmp_path / "out.xyz"

    eval_args = argparse.Namespace(
        model=str(model_path),
        configs=str(xyz),
        output=str(out),
        device="cpu",
        default_dtype="float32",
        batch_size=2,
        compute_stress=True,
        compute_polarization=True,
        compute_becs=True,
        compute_polarizability=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        descriptor_invariants_only=False,
        descriptor_aggregation_method=None,
        descriptor_num_layers=-1,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
        electric_field=None,
        electric_field_key="REF_electric_field",
        polarization_key="REF_polarization",
        becs_key="REF_becs",
        polarizability_key="REF_polarizability",
    )

    mace_eval_configs_run(eval_args)

    imgs = ase.io.read(out, index=":")
    assert len(imgs) == len(field_fitting_configs)

    at = imgs[0]
    assert "MACE_polarization" in at.info
    assert "MACE_polarizability" in at.info
    assert "MACE_becs" in at.arrays

    pol = at.info["MACE_polarization"]
    polz = at.info["MACE_polarizability"]
    becs = at.arrays["MACE_becs"]

    assert np.asarray(pol).shape[-1] == 3
    assert np.asarray(polz).size == 9
    assert becs.shape[-1] == 9 or becs.shape[-2:] == (3, 3)


# -----------------------------------------------------------------------------
# Autograd / physics consistency
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_energy_field_gradient_matches_polarization(tmp_path, field_fitting_configs):
    """Check -∂E/∂E = Ω P numerically for one configuration."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)
    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)

    calc = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float64",
    )

    at = field_fitting_configs[0].copy()
    at.set_cell((6.0, 6.0, 6.0))
    volume = at.get_volume()

    base_field = np.array([0.01, 0.0, 0.0])
    at.info["electric_field"] = base_field
    at.calc = calc
    E0 = at.get_potential_energy()
    P = np.asarray(at.calc.results["polarization"])
    assert np.isfinite(E0)

    h = 1e-3
    for comp in range(3):
        ef_plus = base_field.copy()
        ef_minus = base_field.copy()
        ef_plus[comp] += h
        ef_minus[comp] -= h

        at_plus = at.copy()
        at_plus.info["electric_field"] = ef_plus
        at_plus.calc = calc
        E_plus = at_plus.get_potential_energy()

        at_minus = at.copy()
        at_minus.info["electric_field"] = ef_minus
        at_minus.calc = calc
        E_minus = at_minus.get_potential_energy()

        dEdE_num = (E_plus - E_minus) / (2.0 * h)
        P_est = -dEdE_num / volume

        assert np.isfinite(P_est)
        assert np.allclose(P[comp], P_est, rtol=5e-2, atol=5e-2)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_becs_match_polarization_displacement_response(tmp_path, field_fitting_configs):
    """Check Z ≈ Ω ∂P/∂u numerically for one atom / direction."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)
    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)

    calc = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float64",
    )

    at = field_fitting_configs[0].copy()
    at.calc = calc
    at.get_potential_energy()
    P0 = np.asarray(at.calc.results["polarization"])
    Z = np.asarray(at.calc.results["becs"])  # (natoms, 9)

    h = 1e-3
    volume = at.get_volume()
    i_atom, alpha = 0, 0

    at_plus = at.copy()
    at_plus.positions[i_atom, alpha] += h
    at_plus.calc = calc
    at_plus.get_potential_energy()
    P_plus = np.asarray(at_plus.calc.results["polarization"])

    dP_num = (P_plus - P0) / h
    Z_num = volume * dP_num

    Zi = Z[i_atom].reshape(3, 3)
    assert np.allclose(Zi[:, alpha], Z_num, rtol=2e-1, atol=2e-1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_forces_are_energy_gradients_with_field(tmp_path, field_fitting_configs):
    """Check -∂E/∂R = F with field heads enabled (finite-difference)."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)
    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)

    calc = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float64",
    )

    at = field_fitting_configs[0].copy()
    at.calc = calc
    E0 = at.get_potential_energy()
    F = at.get_forces()
    assert np.isfinite(E0)

    h = 1e-3
    i_atom, alpha = 0, 0

    at_plus = at.copy()
    at_plus.positions[i_atom, alpha] += h
    at_plus.calc = calc
    E_plus = at_plus.get_potential_energy()

    at_minus = at.copy()
    at_minus.positions[i_atom, alpha] -= h
    at_minus.calc = calc
    E_minus = at_minus.get_potential_energy()

    dEdR_num = (E_plus - E_minus) / (2.0 * h)
    assert np.allclose(dEdR_num, -F[i_atom, alpha], rtol=5e-2, atol=5e-2)


# -----------------------------------------------------------------------------
# CLI behaviour and error paths
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_eval_global_field_overrides_per_frame(tmp_path, field_fitting_configs):
    """Global electric_field argument must override per-frame field keys."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz)
    mace_run(args)
    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)

    out = tmp_path / "out_global.xyz"
    global_field = (0.5, -0.5, 0.25)

    eval_args = argparse.Namespace(
        model=str(model_path),
        configs=str(xyz),
        output=str(out),
        device="cpu",
        default_dtype="float32",
        batch_size=2,
        compute_stress=False,
        compute_polarization=True,
        compute_becs=True,
        compute_polarizability=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        descriptor_invariants_only=False,
        descriptor_aggregation_method=None,
        descriptor_num_layers=-1,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
        electric_field=global_field,
        electric_field_key="REF_electric_field",
        polarization_key="REF_polarization",
        becs_key="REF_becs",
        polarizability_key="REF_polarizability",
    )

    mace_eval_configs_run(eval_args)
    imgs = ase.io.read(out, index=":")

    out2 = tmp_path / "out_perframe.xyz"
    eval_args.electric_field = None
    eval_args.output = str(out2)
    mace_eval_configs_run(eval_args)
    imgs2 = ase.io.read(out2, index=":")

    for a, b in zip(imgs, imgs2):
        P_global = np.asarray(a.info["MACE_polarization"])
        P_frame = np.asarray(b.info["MACE_polarization"])
        assert not np.allclose(P_global, P_frame)


def test_eval_field_flags_fail_with_plain_mace(tmp_path, mace_model_path, field_fitting_configs):
    """Requesting field outputs with a plain MACE model should raise."""
    xyz = tmp_path / "fit.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = argparse.Namespace(
        model=str(mace_model_path),
        configs=str(xyz),
        output=str(tmp_path / "out.xyz"),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=True,
        compute_polarization=True,
        compute_becs=True,
        compute_polarizability=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        descriptor_invariants_only=False,
        descriptor_aggregation_method=None,
        descriptor_num_layers=-1,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
        electric_field=None,
        electric_field_key="REF_electric_field",
        polarization_key="REF_polarization",
        becs_key="REF_becs",
        polarizability_key="REF_polarizability",
    )

    with pytest.raises(ValueError, match="MACEField"):
        mace_eval_configs_run(args)


# -----------------------------------------------------------------------------
# Edge cases in data / keys
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_universal_field_loss_handles_missing_keys_gracefully(tmp_path, field_fitting_configs):
    """Training with universal_field should tolerate missing field keys."""
    xyz = tmp_path / "train_missing.xyz"

    stripped = []
    for at in field_fitting_configs:
        at = at.copy()
        for key in ["REF_electric_field", "REF_polarization", "REF_polarizability"]:
            at.info.pop(key, None)
        if "REF_becs" in at.arrays:
            del at.arrays["REF_becs"]
        stripped.append(at)

    ase.io.write(xyz, stripped)

    args = _common_train_args(tmp_path, xyz)
    mace_run(args)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_zero_field_responses_do_not_crash(tmp_path):
    """All-zero field responses should train without errors."""
    cfgs = []
    cell = (6.0, 6.0, 6.0)
    for i in range(3):
        at = Atoms("OH2", positions=[(0, 0, 0), (1, 0, 0), (-0.3, 0.9, 0)], cell=cell, pbc=True)
        _attach_field_response(at, energy=-1.0 + 0.1 * i, field=(0.0, 0.0, 0.0), seed=i)
        at.info["REF_polarization"] = np.zeros(3)
        at.arrays["REF_becs"] = np.zeros((len(at), 9))
        at.info["REF_polarizability"] = np.zeros(9)
        cfgs.append(at)

    xyz = tmp_path / "train_zero.xyz"
    ase.io.write(xyz, cfgs)
    args = _common_train_args(tmp_path, xyz)
    mace_run(args)


# -----------------------------------------------------------------------------
# Calculator API variants
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_calculator_respects_per_atoms_electric_field_override(tmp_path, field_fitting_configs):
    """Check per-Atoms electric_field in info works without REF_electric_field."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)
    args = _common_train_args(tmp_path, xyz)
    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)

    calc = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float32",
    )

    at = field_fitting_configs[0].copy()
    at.info.pop("REF_electric_field", None)

    fields = [(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)]
    pols = []
    for ef in fields:
        at_ = at.copy()
        at_.info["electric_field"] = ef
        at_.calc = calc
        at_.get_potential_energy()
        pol = np.asarray(at_.calc.results["polarization"])
        pols.append(pol)
        assert pol.shape[-1] == 3
        assert np.all(np.isfinite(pol))

    assert pols[0].shape == pols[1].shape


# -----------------------------------------------------------------------------
# Foundation / multi-head integration
# -----------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_macefield_train_with_foundation(tmp_path, field_fitting_configs):
    """MACEField run with foundation_model options enabled (shape-compatible)."""
    xyz = tmp_path / "train.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz)
    args.foundation_model = "small"
    args.multiheads_finetuning = False

    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_macefield_multihead_finetuning_smoke(
    tmp_path, field_fitting_configs, mace_model_path
):
    """Multi-head replay-finetuning run with MACEField."""
    xyz = tmp_path / "train_mhft.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz, name="MACEField-mhft")

    # Build a tiny foundation checkpoint whose interaction class avoids the
    # skip_tp reshape branch in load_foundations_elements (no utils changes).
    foundation_path = tmp_path / "mace_nonlin.model"
    with default_dtype(torch.float32):
        cfg = dict(MODEL_CONFIG)
        cfg["interaction_cls"] = interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"]
        cfg["interaction_cls_first"] = interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"]
        model = ScaleShiftMACE(**cfg)
        torch.save(model, foundation_path)

    args.foundation_model = str(foundation_path)
    args.pt_train_file = str(xyz)
    args.atomic_numbers = "[1, 8]"
    args.multiheads_finetuning = True
    args.force_mh_ft_lr = True

    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_macefield_multihead_calculator_heads(
    tmp_path, field_fitting_configs, mace_model_path
):
    """Evaluate a multi-head MACEField model with both heads on the same Atoms."""
    xyz = tmp_path / "train_mhft2.xyz"
    ase.io.write(xyz, field_fitting_configs)

    args = _common_train_args(tmp_path, xyz, name="MACEField-mhft-heads")

    # Same foundation trick as the smoke test (avoid skip_tp reshape branch).
    foundation_path = tmp_path / "mace_nonlin.model"
    with default_dtype(torch.float32):
        cfg = dict(MODEL_CONFIG)
        cfg["interaction_cls"] = interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"]
        cfg["interaction_cls_first"] = interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"]
        model = ScaleShiftMACE(**cfg)
        torch.save(model, foundation_path)

    args.foundation_model = str(foundation_path)
    args.pt_train_file = str(xyz)
    args.atomic_numbers = "[1, 8]"
    args.multiheads_finetuning = True
    args.force_mh_ft_lr = True

    mace_run(args)

    model_path = _locate_trained_model(tmp_path, args.name, seed=args.seed)
    assert model_path.exists()

    at = field_fitting_configs[0].copy()

    calc_default = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float32",
        head="Default",
    )
    calc_pt = _make_macefield_calculator(
        model_path,
        device="cpu",
        default_dtype_str="float32",
        head="pt_head",
    )

    at_default = at.copy()
    at_default.calc = calc_default
    e_default = at_default.get_potential_energy()
    f_default = at_default.get_forces()

    at_pt = at.copy()
    at_pt.calc = calc_pt
    e_pt = at_pt.get_potential_energy()
    f_pt = at_pt.get_forces()

    assert np.isfinite(e_default)
    assert np.isfinite(e_pt)
    assert f_default.shape == (len(at), 3)
    assert f_pt.shape == (len(at), 3)


# -----------------------------------------------------------------------------
# Foundation loader unit tests (skip_tp + MACEField field modules)
# -----------------------------------------------------------------------------


class _FakeLinear(torch.nn.Module):
    # NOTE: loader expects `.bias` attribute (may be None, like torch.nn.Linear)
    def __init__(self, out_features, in_features, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features)) if bias else None


class _FakeBessel(torch.nn.Module):
    def __init__(self, num_basis):
        super().__init__()
        self.bessel_weights = torch.nn.Parameter(torch.randn(num_basis, 4))


class _FakeRadialEmbedding(torch.nn.Module):
    def __init__(self, out_dim, num_basis):
        super().__init__()
        self.out_dim = out_dim
        self.bessel_fn = _FakeBessel(num_basis)


class _FakeConvTPWeights(torch.nn.Module):
    def __init__(self, num_radial):
        super().__init__()
        # 4 layers, like in the loader
        self.layer0 = _FakeLinear(num_radial, 8)
        self.layer1 = _FakeLinear(8, 8)
        self.layer2 = _FakeLinear(8, 8)
        self.layer3 = _FakeLinear(8, 8)


class _FakeSkipTP(torch.nn.Module):
    def __init__(self, n_paths):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_paths))


class _FakeDensityFn(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer0 = _FakeLinear(out_features, in_features)


class _FakeInteraction(torch.nn.Module):
    def __init__(self, num_radial, skip_tp_paths, with_density=True):
        super().__init__()
        self.linear_up = _FakeLinear(16, 16)
        self.avg_num_neighbors = 1.23
        self.conv_tp_weights = _FakeConvTPWeights(num_radial)
        self.linear = _FakeLinear(32, 32)
        self.skip_tp = _FakeSkipTP(skip_tp_paths)
        if with_density:
            # match name check in loader
            self.__class__.__name__ = "RealAgnosticDensityResidualInteractionBlock"
            self.density_fn = _FakeDensityFn(16, 16)


class _FakeContraction(torch.nn.Module):
    def __init__(self, num_species, dim_in, dim_out):
        super().__init__()
        self.weights_max = torch.nn.Parameter(
            torch.randn(num_species, dim_in, dim_out)
        )
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.randn(num_species, dim_in, dim_out)),
                torch.nn.Parameter(torch.randn(num_species, 1, dim_out)),
            ]
        )


class _FakeSymmetricContractions(torch.nn.Module):
    def __init__(self, num_contractions, num_species, dim_in, dim_out):
        super().__init__()
        self.contractions = torch.nn.ModuleList(
            [
                _FakeContraction(num_species, dim_in, dim_out)
                for _ in range(num_contractions)
            ]
        )


class _FakeProductBlock(torch.nn.Module):
    def __init__(self, num_contractions, num_species):
        super().__init__()
        self.symmetric_contractions = _FakeSymmetricContractions(
            num_contractions=num_contractions,
            num_species=num_species,
            dim_in=4,
            dim_out=8,
        )
        self.linear = _FakeLinear(16, 16)


class _FakeReadout0(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # match name check in loader
        self.__class__.__name__ = "LinearReadoutBlock"
        self.linear = _FakeLinear(1, num_channels)


class _FakeReadout1(torch.nn.Module):
    def __init__(self, num_channels, hidden_irreps):
        super().__init__()
        # match name check in loader
        self.__class__.__name__ = "NonLinearReadoutBlock"

        class _IrrepsOut:
            def __init__(self, n):
                self.num_irreps = n

        self.linear_1 = _FakeLinear(hidden_irreps, num_channels)
        self.linear_1.__dict__["irreps_out"] = _IrrepsOut(hidden_irreps)

        self.linear_2 = _FakeLinear(1, hidden_irreps)
        self.linear_2.__dict__["irreps_out"] = _IrrepsOut(1)


class _FakeScaleShift(torch.nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.scale = torch.randn(n_heads)
        self.shift = torch.randn(n_heads)


class _FakeFieldFeat(torch.nn.Module):
    def __init__(self, n_paths):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_paths))


class _FakeFieldLinear(torch.nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))


class _FakeModel(torch.nn.Module):
    def __init__(
        self,
        atomic_numbers,
        *,
        num_interactions=2,
        skip_tp_paths=32_768,
        field_paths=128,
    ):
        super().__init__()
        self.r_max = 6.0
        self.atomic_numbers = atomic_numbers
        self.heads = ["Default"]
        self.num_interactions = num_interactions

        num_species = len(atomic_numbers)

        # node embedding: (num_species * num_channels) x in_features
        self.node_embedding = torch.nn.Module()
        self.node_embedding.linear = _FakeLinear(
            out_features=num_species * 8,
            in_features=4,
        )

        # radial embedding
        self.radial_embedding = _FakeRadialEmbedding(out_dim=5, num_basis=5)

        # dummy SH (for max_ell access in loader)
        class _SH:
            def __init__(self, lmax):
                self._lmax = lmax

        self.spherical_harmonics = _SH(lmax=3)

        # interactions
        self.interactions = torch.nn.ModuleList(
            [
                _FakeInteraction(
                    num_radial=self.radial_embedding.out_dim,
                    skip_tp_paths=skip_tp_paths,
                    with_density=True,
                )
                for _ in range(num_interactions)
            ]
        )

        # 2 product groups, like in loader
        self.products = torch.nn.ModuleList(
            [
                _FakeProductBlock(num_contractions=3, num_species=num_species)
                for _ in range(2)
            ]
        )

        # readouts [0]=linear, [1]=nonlinear
        self.readouts = torch.nn.ModuleList(
            [
                _FakeReadout0(num_channels=8),
                _FakeReadout1(num_channels=8, hidden_irreps=4),
            ]
        )

        self.scale_shift = _FakeScaleShift(n_heads=len(self.heads))

        # MACEField-like field modules
        self.field_feats = torch.nn.ModuleList([_FakeFieldFeat(n_paths=field_paths)])
        self.field_linear = torch.nn.ModuleList(
            [_FakeFieldLinear(out_features=8, in_features=8)]
        )


def test_foundations_loader_skip_tp_not_copied_when_shape_mismatch():
    """For a reduced element table, skip_tp is transferred by slicing Z and scaling (shape-consistent)."""
    # In _FakeModel: num_channels_foundation = 8
    # skip_tp is reshaped as (C, Z_found, C) => size C*Z*C
    foundation = _FakeModel(atomic_numbers=[1, 6, 8], skip_tp_paths=8 * 3 * 8, field_paths=128)
    target = _FakeModel(atomic_numbers=[1, 8], skip_tp_paths=8 * 2 * 8, field_paths=128)

    table = AtomicNumberTable([1, 8])

    # snapshots
    original_target_skip = [blk.skip_tp.weight.clone() for blk in target.interactions]

    load_foundations_elements(
        model=target,
        model_foundations=foundation,
        table=table,
        load_readout=True,
    )

    # Expected transfer for this interaction branch:
    # reshape to (C, Z_found, C), slice Z, flatten, scale by sqrt(Z_found/Z_target)
    z_table = AtomicNumberTable([int(z) for z in foundation.atomic_numbers])
    indices_weights = [z_table.z_to_index(z) for z in table.zs]
    C = foundation.node_embedding.linear.weight.shape[0] // len(z_table.zs)
    scale = (len(z_table.zs) / len(table.zs)) ** 0.5

    for i in range(target.num_interactions):
        expected = (
            foundation.interactions[i]
            .skip_tp.weight.reshape(C, len(z_table.zs), C)[:, indices_weights, :]
            .flatten()
            .clone()
            / scale
        )

        # target skip_tp must change from its original value
        assert not torch.allclose(target.interactions[i].skip_tp.weight, original_target_skip[i])
        # and match the expected sliced+scaled transfer
        assert torch.allclose(target.interactions[i].skip_tp.weight, expected)


def test_foundations_loader_skip_tp_and_field_modules_inherited_when_shapes_match():
    """When shapes match, skip_tp and MACEField field modules are inherited (current loader behaviour)."""
    # For Z=2 and C=8 => skip_tp size is 8*2*8 = 128
    foundation = _FakeModel(
        atomic_numbers=[1, 8],
        skip_tp_paths=8 * 2 * 8,
        field_paths=512,
    )
    target = _FakeModel(
        atomic_numbers=[1, 8],
        skip_tp_paths=8 * 2 * 8,
        field_paths=512,
    )

    table = AtomicNumberTable([1, 8])

    # snapshots of target field modules before transfer
    target_field_feat_before = target.field_feats[0].weight.clone()
    target_field_linear_w_before = target.field_linear[0].weight.clone()
    target_field_linear_b_before = target.field_linear[0].bias.clone()

    load_foundations_elements(
        model=target,
        model_foundations=foundation,
        table=table,
        load_readout=True,
    )

    # skip_tp should now match foundation (Z same => scale=1)
    for i in range(target.num_interactions):
        assert torch.allclose(
            target.interactions[i].skip_tp.weight,
            foundation.interactions[i].skip_tp.weight,
        )

    # field_feats should be inherited (copied during readout transfer for NonLinearReadoutBlock)
    assert torch.allclose(
        target.field_feats[0].weight,
        foundation.field_feats[0].weight,
    )
    assert not torch.allclose(
        target.field_feats[0].weight,
        target_field_feat_before,
    )

    # field_linear should be inherited (both weight and bias)
    assert torch.allclose(
        target.field_linear[0].weight,
        foundation.field_linear[0].weight,
    )
    assert torch.allclose(
        target.field_linear[0].bias,
        foundation.field_linear[0].bias,
    )

    assert not torch.allclose(
        target.field_linear[0].weight,
        target_field_linear_w_before,
    )
    assert not torch.allclose(
        target.field_linear[0].bias,
        target_field_linear_b_before,
    )


def test_foundations_loader_field_modules_not_inherited_when_flag_false():
    """With load_readout=False, field modules remain unchanged (no inheritance performed)."""
    # For Z=2 and C=8 => skip_tp size is 8*2*8 = 128
    foundation = _FakeModel(
        atomic_numbers=[1, 8],
        skip_tp_paths=8 * 2 * 8,
        field_paths=512,
    )
    target = _FakeModel(
        atomic_numbers=[1, 8],
        skip_tp_paths=8 * 2 * 8,
        field_paths=512,
    )

    table = AtomicNumberTable([1, 8])

    # snapshots before transfer
    target_field_feat_before = target.field_feats[0].weight.clone()
    target_field_linear_w_before = target.field_linear[0].weight.clone()
    target_field_linear_b_before = target.field_linear[0].bias.clone()

    load_foundations_elements(
        model=target,
        model_foundations=foundation,
        table=table,
        load_readout=False,
    )

    # Field modules must remain unchanged
    assert torch.allclose(
        target.field_feats[0].weight,
        target_field_feat_before,
    )
    assert torch.allclose(
        target.field_linear[0].weight,
        target_field_linear_w_before,
    )
    assert torch.allclose(
        target.field_linear[0].bias,
        target_field_linear_b_before,
    )
