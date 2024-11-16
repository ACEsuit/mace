from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import torch_geometric

try:
    import cuequivariance as cue # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
class TestCueq:
    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        return {
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "num_interactions": 2,
            "num_elements": 1,
            "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
            "MLP_irreps": o3.Irreps("16x0e"),
            "gate": F.silu,
            "atomic_energies": torch.tensor([1.0]),
            "avg_num_neighbors": 8,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
        }

    @pytest.fixture
    def batch(self, device: str):
        from ase import build

        table = tools.AtomicNumberTable([6])

        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
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
    def test_cueq_equivalence(
        self, model_config: Dict[str, Any], batch: Dict[str, torch.Tensor], device: str
    ):
        torch.manual_seed(42)

        # Create model without cuequivariance
        model_std = modules.MACE(**model_config)
        model_std = model_std.to(device)

        # Create model with cuequivariance
        cueq_config = CuEquivarianceConfig(
            enabled=True, layout="mul_ir", group="O3_e3nn", optimize_all=True
        )
        model_config["cueq_config"] = cueq_config
        model_cueq = modules.MACE(**model_config)
        model_cueq = model_cueq.to(device)

        # Copy weights
        # model_cueq.load_state_dict(model_std.state_dict())

        # Compare outputs
        out_std = model_std(batch, training=True)
        out_cueq = model_cueq(batch, training=True)

        torch.testing.assert_close(out_std["energy"], out_cueq["energy"])
        torch.testing.assert_close(out_std["forces"], out_cueq["forces"])

        loss_std = out_std["energy"].sum()
        loss_cueq = out_cueq["energy"].sum()

        loss_std.backward()
        loss_cueq.backward()

        for p1, p2 in zip(model_std.parameters(), model_cueq.parameters()):
            if p1.grad is not None:
                torch.testing.assert_close(p1.grad, p2.grad)
