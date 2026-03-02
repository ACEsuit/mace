from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.calculators.foundations_models import mace_polar
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric, utils

try:
    import graph_longrange  # noqa: F401

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)

POLAR_MODELS = [
    ("polar-1-l", -2079.86474609375),
    ("polar-1-m", -2079.86376953125),
    ("polar-1-s", -2079.86474609375),
]


def _water_atoms() -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2390, 0.9270, 0.0],
            ],
            dtype=float,
        ),
    )


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _skip_if_model_unavailable(exc: Exception, model_name: str) -> None:
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if "Model download failed and no local model found" not in msg:
            raise exc
    pytest.skip(f"Polar model {model_name} not available")


def _build_model_batch(model: torch.nn.Module) -> dict:
    atoms = _water_atoms()
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    keyspec = data.KeySpecification(
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        },
        arrays_keys={"Qs": "charges"},
    )
    config = data.config_from_atoms(
        atoms, key_specification=keyspec, head_name="Default"
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=model.heads,
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to("cpu").to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            batch[key] = value.to(model_dtype)
    return batch


@pytest.mark.parametrize("model_name, _", POLAR_MODELS)
def test_polar_models_run_with_and_without_cueq(model_name, _):
    try:
        calc_e3 = mace_polar(model=model_name, device="cpu", enable_cueq=False)
        calc_cueq = mace_polar(model=model_name, device="cpu", enable_cueq=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    atoms = _water_atoms()
    atoms.calc = calc_e3
    energy_e3 = float(atoms.get_potential_energy())
    assert np.isfinite(energy_e3)

    atoms.calc = calc_cueq
    energy_cueq = float(atoms.get_potential_energy())
    assert np.isfinite(energy_cueq)

    assert abs(energy_cueq - energy_e3) <= 3.0e-4


@pytest.mark.parametrize("model_name, _", POLAR_MODELS)
def test_polar_true_cueq_matches_e3nn(model_name, _):
    try:
        model_e3 = mace_polar(
            model=model_name, device="cpu", return_raw_model=True
        ).eval()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(next(model_e3.parameters()).dtype)
        batch = _build_model_batch(model_e3)

        out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
        model_cueq = run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval()
        out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

        energy_max_abs = torch.max(
            torch.abs(out_cueq["energy"] - out_e3["energy"])
        ).item()
        forces_max_abs = torch.max(
            torch.abs(out_cueq["forces"] - out_e3["forces"])
        ).item()
        stress_max_abs = torch.max(
            torch.abs(out_cueq["stress"] - out_e3["stress"])
        ).item()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert energy_max_abs <= 3.0e-4
    assert forces_max_abs <= 2.5e-4
    assert stress_max_abs <= 1.0e-8


def test_polar_2l_true_cueq_matches_e3nn_float64():
    model_name = "polar-1-m"
    try:
        model_e3 = (
            mace_polar(model=model_name, device="cpu", return_raw_model=True)
            .eval()
            .double()
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        batch = _build_model_batch(model_e3)

        out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
        model_cueq = (
            run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval().double()
        )
        out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

        energy_max_abs = torch.max(
            torch.abs(out_cueq["energy"] - out_e3["energy"])
        ).item()
        forces_max_abs = torch.max(
            torch.abs(out_cueq["forces"] - out_e3["forces"])
        ).item()
        stress_max_abs = torch.max(
            torch.abs(out_cueq["stress"] - out_e3["stress"])
        ).item()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert energy_max_abs <= 1.0e-7
    assert forces_max_abs <= 1.0e-7
    assert stress_max_abs <= 1.0e-11


try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

try:
    import cuequivariance_torch as cuet  # noqa: F401

    CUET_OPS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_OPS_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
@pytest.mark.skipif(
    not CUET_OPS_AVAILABLE, reason="cuequivariance_torch (ops) not installed"
)
class TestPolarCueqParity:
    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        hidden_irreps = o3.Irreps("16x0e + 16x1o")
        MLP_irreps = o3.Irreps("8x0e")
        return {
            "r_max": 5.0,
            "num_bessel": 6,
            "num_polynomial_cutoff": 6,
            "max_ell": 2,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "num_interactions": 2,
            "num_elements": 1,
            "hidden_irreps": hidden_irreps,
            "MLP_irreps": MLP_irreps,
            "gate": torch.nn.functional.silu,
            "atomic_energies": torch.tensor([0.0]),
            "avg_num_neighbors": 8.0,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
            "kspace_cutoff_factor": 1.0,
            "atomic_multipoles_max_l": 1,
            "atomic_multipoles_smearing_width": 1.5,
            "field_feature_max_l": 1,
            "field_feature_widths": [1.5],
            "field_feature_norms": [1.0, 1.0],
            "num_recursion_steps": 1,
            "include_electrostatic_self_interaction": False,
            "add_local_electron_energy": False,
            "quadrupole_feature_corrections": False,
            "return_electrostatic_potentials": False,
            "field_norm_factor": 1.0,
            "fixedpoint_update_config": {
                "type": "AgnosticEmbeddedOneBodyVariableUpdate",
                "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
                "nonlinearity_cls": "NoNonLinearity",
            },
            "field_readout_config": {"type": "OneBodyMLPFieldReadout"},
        }

    @pytest.fixture(params=(["cuda"] if CUDA_AVAILABLE else ["cpu"]))
    def device(self, request):
        return request.param

    @pytest.fixture
    def batch(self, device: str) -> Dict[str, torch.Tensor]:
        from ase import build

        table = tools.AtomicNumberTable([6])
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms_list = [atoms.repeat((2, 2, 2))]
        configs = [data.config_from_atoms(at) for at in atoms_list]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(cfg, z_table=table, cutoff=5.0)
                for cfg in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(loader)).to(device)
        n_graphs = int(batch["num_graphs"]) if "num_graphs" in batch else 1
        batch["fermi_level"] = torch.zeros(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        batch["external_field"] = torch.zeros(
            n_graphs, 3, device=device, dtype=batch["positions"].dtype
        )
        batch["total_charge"] = torch.zeros(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        batch["total_spin"] = torch.ones(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        return batch

    def test_e3nn_cueq_parity(self, model_config: Dict[str, Any], batch, device):
        previous_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model_config = dict(model_config)
            model_config["atomic_energies"] = torch.tensor(
                [0.0], dtype=torch.get_default_dtype()
            )
            model_e3 = modules.PolarMACE(**model_config).to(device)

            model_cu = run_e3nn_to_cueq(model_e3, device=device, layout="mul_ir").to(
                device
            )
            model_e3_back = run_cueq_to_e3nn(model_cu).to(device)

            def cast_batch(b):
                d = b.to_dict()
                out = {}
                for k, v in d.items():
                    if torch.is_tensor(v) and v.dtype.is_floating_point:
                        out[k] = v.to(dtype=torch.get_default_dtype())
                    else:
                        out[k] = v
                return out

            bd = cast_batch(batch)
            out_e3 = model_e3(bd, training=True, compute_stress=True)
            out_cu = model_cu(bd, training=True, compute_stress=True)
            out_back = model_e3_back(bd, training=True, compute_stress=True)
        finally:
            torch.set_default_dtype(previous_default_dtype)

        torch.testing.assert_close(out_e3["energy"], out_cu["energy"])
        torch.testing.assert_close(out_cu["energy"], out_back["energy"])
        torch.testing.assert_close(out_e3["forces"], out_cu["forces"])
        torch.testing.assert_close(out_cu["forces"], out_back["forces"])
        torch.testing.assert_close(out_e3["stress"], out_cu["stress"])
        torch.testing.assert_close(out_cu["stress"], out_back["stress"])
