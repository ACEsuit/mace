import os
from typing import Any, Dict

import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric


try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

import importlib.util as _ilu
CUET_OPS_AVAILABLE = _ilu.find_spec("cuequivariance_torch") is not None

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
@pytest.mark.skipif(
    not CUET_OPS_AVAILABLE, reason="cuequivariance_torch (ops) not installed"
)
class TestFieldFukuiCueqParity:
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
            # FieldFukui specifics
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
    def batch(self, device: str, model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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
        # Supply extra FieldFukui keys
        n_graphs = int(batch["num_graphs"]) if "num_graphs" in batch else 1
        batch["fermi_level"] = torch.zeros(n_graphs, device=device, dtype=batch["positions"].dtype)
        batch["external_field"] = torch.zeros(n_graphs, 3, device=device, dtype=batch["positions"].dtype)
        batch["total_charge"] = torch.zeros(n_graphs, device=device, dtype=batch["positions"].dtype)
        batch["total_spin"] = torch.ones(n_graphs, device=device, dtype=batch["positions"].dtype)
        return batch

    def test_e3nn_cueq_parity(self, model_config: Dict[str, Any], batch, device):
        torch.set_default_dtype(torch.float64)
        # Ensure dtypes in config match default dtype
        model_config = dict(model_config)
        model_config["atomic_energies"] = torch.tensor([0.0], dtype=torch.get_default_dtype())
        model_e3 = modules.FieldFukuiMACE(**model_config).to(device)

        # Convert to CuEq and back
        # Request mul_ir layout to match e3nn internal ops exactly
        model_cu = run_e3nn_to_cueq(model_e3, device=device, layout="mul_ir").to(device)
        model_e3_back = run_cueq_to_e3nn(model_cu).to(device)

        # Forward
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

        torch.testing.assert_close(out_e3["energy"], out_cu["energy"])
        torch.testing.assert_close(out_cu["energy"], out_back["energy"])
        torch.testing.assert_close(out_e3["forces"], out_cu["forces"])
        torch.testing.assert_close(out_cu["forces"], out_back["forces"])
        torch.testing.assert_close(out_e3["stress"], out_cu["stress"])
        torch.testing.assert_close(out_cu["stress"], out_back["stress"])
