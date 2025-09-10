# pylint: disable=wrong-import-position
import os
from typing import Any, Dict

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
from mace.cli.convert_oeq_e3nn import run as run_oeq_to_e3nn
from mace.tools import torch_geometric

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

try:
    import openequivariance as oeq  # pylint: disable=unused-import

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


class BackendTestBase:
    @pytest.fixture
    def model_config(
        self,
        interaction_cls_first,
        hidden_irreps,
        use_agnostic_product,
        use_last_readout_only,
        use_reduced_cg,
    ) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        return {
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
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
            "use_agnostic_product": use_agnostic_product,
            "use_last_readout_only": use_last_readout_only,
            "use_reduced_cg": use_reduced_cg,
        }

    @pytest.fixture
    def batch(self, device: str, default_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        from ase import build

        torch.set_default_dtype(default_dtype)

        table = tools.AtomicNumberTable([6])

        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        import numpy as np

        displacement = np.random.uniform(-0.1, 0.1, size=atoms.positions.shape)
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
        return batch.to(device)

    @pytest.mark.parametrize(
        "interaction_cls_first",
        [
            modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            modules.interaction_classes["RealAgnosticInteractionBlock"],
            modules.interaction_classes["RealAgnosticDensityInteractionBlock"],
            modules.interaction_classes[
                "RealAgnosticResidualNonLinearInteractionBlock"
            ],
        ],
    )
    @pytest.mark.parametrize(
        "hidden_irreps",
        [
            o3.Irreps("32x0e + 32x1o"),
            o3.Irreps("32x0e + 32x1o + 32x2e"),
            o3.Irreps("32x0e"),
        ],
    )
    @pytest.mark.parametrize("default_dtype", [torch.float64])
    @pytest.mark.parametrize("use_agnostic_product", [False])
    @pytest.mark.parametrize("use_last_readout_only", [False])
    @pytest.mark.parametrize("use_reduced_cg", [True, False])
    def test_bidirectional_conversion(
        self,
        model_config: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        device: str,
        default_dtype: torch.dtype,
        conversion_functions: tuple,
    ):
        run_e3nn_to_backend, run_backend_to_e3nn = conversion_functions

        if device == "cuda" and not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        torch.manual_seed(42)

        # Create original E3nn model
        model_e3nn = modules.ScaleShiftMACE(**model_config).to(device)

        # Convert E3nn to CuEq
        model_backend = run_e3nn_to_backend(model_e3nn).to(device)

        # Convert CuEq back to E3nn
        model_e3nn_back = run_backend_to_e3nn(model_backend).to(device)

        # Test forward pass equivalence
        out_e3nn = model_e3nn(batch.clone().to_dict(), training=True, compute_stress=True)
        out_backend = model_backend(batch.clone().to_dict(), training=True, compute_stress=True)

        out_e3nn_back = model_e3nn_back(
            batch.clone().to_dict(), training=True, compute_stress=True
        )

        # Check outputs match for both conversions

        torch.testing.assert_close(out_e3nn["energy"], out_backend["energy"])
        torch.testing.assert_close(out_backend["energy"], out_e3nn_back["energy"])
        torch.testing.assert_close(out_e3nn["forces"], out_backend["forces"])
        torch.testing.assert_close(out_backend["forces"], out_e3nn_back["forces"])
        torch.testing.assert_close(out_e3nn["stress"], out_backend["stress"])
        torch.testing.assert_close(out_backend["stress"], out_e3nn_back["stress"])

        # Test backward pass equivalence
        loss_e3nn = out_e3nn["energy"].sum()
        loss_backend = out_backend["energy"].sum()
        loss_e3nn_back = out_e3nn_back["energy"].sum()

        loss_e3nn.backward()
        loss_backend.backward()
        loss_e3nn_back.backward()

        tol = 1e-4 if default_dtype == torch.float32 else 1e-8

        def print_gradient_diff(name1, p1, name2, p2, conv_type):
            if p1.grad is not None and p1.grad.shape == p2.grad.shape:
                if name1.split(".", 2)[:2] == name2.split(".", 2)[:2]:
                    error = torch.abs(p1.grad - p2.grad)
                    print(
                        f"{conv_type} - Parameter {name1}/{name2}, Max error: {error.max()}"
                    )
                    torch.testing.assert_close(p1.grad, p2.grad, atol=tol, rtol=tol)

        # E3nn to CuEq gradients
        for (name_e3nn, p_e3nn), (name_backend, p_backend) in zip(
            model_e3nn.named_parameters(), model_backend.named_parameters()
        ):
            print_gradient_diff(
                name_e3nn, p_e3nn, name_backend, p_backend, "E3nn->CuEq"
            )

        # CuEq to E3nn gradients
        for (name_backend, p_backend), (name_e3nn_back, p_e3nn_back) in zip(
            model_backend.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_backend, p_backend, name_e3nn_back, p_e3nn_back, "CuEq->E3nn"
            )

        # Full circle comparison (E3nn -> E3nn)
        for (name_e3nn, p_e3nn), (name_e3nn_back, p_e3nn_back) in zip(
            model_e3nn.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_e3nn, p_e3nn, name_e3nn_back, p_e3nn_back, "Full circle"
            )


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
class TestCueq(BackendTestBase):
    @pytest.fixture
    def conversion_functions(self):
        return run_e3nn_to_cueq, run_cueq_to_e3nn

    @pytest.fixture(params=(["cuda"] if CUDA_AVAILABLE else ["cpu"]))
    def device(self, request):
        return request.param


@pytest.mark.skipif(not OEQ_AVAILABLE, reason="openequivariance not installed")
class TestOeq(BackendTestBase):
    @pytest.fixture
    def conversion_functions(self):
        return run_e3nn_to_oeq, run_oeq_to_e3nn

    @pytest.fixture(params=(["cuda"] if CUDA_AVAILABLE else []))
    def device(self, request):
        return request.param
