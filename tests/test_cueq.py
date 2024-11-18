from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import torch_geometric
from mace.cli.convert_e3nn_cueq import run as run_convert

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

torch.set_default_dtype(torch.float64)

@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
class TestCueq:
    @pytest.fixture
    def model_config(self, interaction_cls_first, hidden_irreps) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        print("interaction_cls_first", interaction_cls_first)
        print("hidden_irreps", hidden_irreps)
        return {
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            "interaction_cls_first": interaction_cls_first,
            "num_interactions": 2,
            "num_elements": 1,
            "hidden_irreps": hidden_irreps,
            "MLP_irreps": o3.Irreps("16x0e"),
            "gate": F.silu,
            "atomic_energies": torch.tensor([1.0]),
            "avg_num_neighbors": 8,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
        }

    @pytest.fixture
    def batch(self, device: str):
        from ase import build
        table = tools.AtomicNumberTable([6])

        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        import numpy as np
        displacement = np.random.uniform(
            -0.1, 0.1, size=atoms.positions.shape
        )
        atoms.positions += displacement
        atoms_list = [atoms.repeat((2, 2, 2))]

        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
                for config in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))
        return batch.to(device).to_dict()

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize("interaction_cls_first", [
        modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        modules.interaction_classes["RealAgnosticInteractionBlock"],
        modules.interaction_classes["RealAgnosticDensityInteractionBlock"],
    ])
    @pytest.mark.parametrize("hidden_irreps", [
        #o3.Irreps("32x0e + 32x1o"),
        #o3.Irreps("32x0e + 32x1o + 32x2e"),
        o3.Irreps("32x0e"),
    ])
    def test_cueq_equivalence(
        self, 
        model_config: Dict[str, Any], 
        batch: Dict[str, torch.Tensor], 
        device: str,
        interaction_cls_first,
        hidden_irreps
    ):
        torch.manual_seed(42)

        # Create model without cuequivariance
        model_std = modules.ScaleShiftMACE(**model_config)
        model_std = model_std.to(device)

        # Create model with cuequivariance
        cueq_config = CuEquivarianceConfig(
            enabled=True, layout="mul_ir", group="O3_e3nn", optimize_all=True
        )
        model_config["cueq_config"] = cueq_config
        model_cueq = modules.ScaleShiftMACE(**model_config)
        model_cueq = model_cueq.to(device)

        # Copy weights 
        model_cueq_convert = run_convert(model_std, None)
        model_cueq_convert = model_cueq_convert.to(device)
        
        # Compare outputs
        out_std = model_std(batch, training=True)
        out_cueq_convert = model_cueq_convert(batch, training=True)

        torch.testing.assert_close(out_std["energy"], out_cueq_convert["energy"])
        torch.testing.assert_close(out_std["forces"], out_cueq_convert["forces"])
        
        loss_std = out_std["energy"].sum()
        loss_cueq = out_cueq_convert["energy"].sum()

        loss_std.backward()
        loss_cueq.backward()

        for (name_1, p1), (name_2, p2) in zip(model_std.named_parameters(), model_cueq_convert.named_parameters()):
            if p1.grad is not None:
                if p1.grad.shape == p2.grad.shape:
                    if name_1.split(".", 2)[:2] == name_2.split(".", 2)[:2]:
                        error = torch.abs(p1.grad - p2.grad)
                        print(f"Parameter {name_1}, Parameter {name_2}, Max error: {error.max()}")
                        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-10)