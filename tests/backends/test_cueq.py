"""cuEquivariance backend: e3nn parity (CPU or GPU) and the CUDA multi-head
regression (issue #1298). Capability: cueq; GPU cases carry the gpu marker.
"""
# pylint: disable=wrong-import-position
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from tests.backends.backend_parity import BackendTestBase
from tests.helpers import CUDA_AVAILABLE, CUET_AVAILABLE



@pytest.mark.cueq
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
class TestCueq(BackendTestBase):
    @pytest.fixture
    def conversion_functions(self):
        return run_e3nn_to_cueq, run_cueq_to_e3nn

    @pytest.fixture(params=(["cuda"] if CUDA_AVAILABLE else ["cpu"]))
    def device(self, request):
        return request.param


@pytest.mark.cueq
@pytest.mark.gpu
@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="cuda is not available")
@pytest.mark.parametrize("head_name", ["DFT", "MP2"])
def test_cueq_cuda_multihead_matches_e3nn(head_name):
    """Regression for CUDA CuEq multi-head inference.

    Issue #1298 reports large energy disagreements for MACE-MH-1 with
    enable_cueq=True on CUDA.  This smaller model exercises the same
    multi-head readout/indexing path under CuEq conv fusion.
    """
    import numpy as np
    from ase import build

    device = "cuda"
    default_dtype = torch.float64
    heads = ["DFT", "MP2"]
    torch.manual_seed(17)

    table = tools.AtomicNumberTable([1, 8])
    atoms = build.molecule("H2O")
    atoms.cell = [8.0, 8.0, 8.0]
    atoms.pbc = [False, False, False]
    atoms.positions += np.array([3.0, 3.0, 3.0])
    config = data.config_from_atoms(atoms, head_name=head_name)
    # Build the batch under float64 so node_attrs/positions match the model dtype.
    with tools.torch_tools.default_dtype(default_dtype):
        batch = next(
            iter(
                torch_geometric.dataloader.DataLoader(
                    dataset=[
                        data.AtomicData.from_config(
                            config,
                            z_table=table,
                            cutoff=4.5,
                            heads=heads,
                        )
                    ],
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
            )
        ).to(device)

    with tools.torch_tools.default_dtype(default_dtype):
        model_e3nn = modules.ScaleShiftMACE(
            r_max=4.5,
            num_bessel=4,
            num_polynomial_cutoff=3,
            max_ell=2,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=len(table),
            hidden_irreps=o3.Irreps("16x0e + 16x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=F.silu,
            atomic_energies=torch.tensor(
                [[-1.0, -2.0], [-8.0, -9.0]], dtype=default_dtype
            ),
            avg_num_neighbors=4,
            atomic_numbers=table.zs,
            correlation=2,
            radial_type="bessel",
            atomic_inter_scale=[1.0, 1.5],
            atomic_inter_shift=[0.1, -0.2],
            heads=heads,
            use_reduced_cg=False,
        ).to(device=device, dtype=default_dtype)

    model_cueq = run_e3nn_to_cueq(model_e3nn, device=device).to(
        device=device, dtype=default_dtype
    )

    batch_e3nn = batch.clone().to_dict()
    batch_cueq = batch.clone().to_dict()
    out_e3nn = model_e3nn(batch_e3nn, training=False, compute_stress=True)
    out_cueq = model_cueq(batch_cueq, training=False, compute_stress=True)

    torch.testing.assert_close(
        out_cueq["energy"], out_e3nn["energy"], atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        out_cueq["forces"], out_e3nn["forces"], atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        out_cueq["stress"], out_e3nn["stress"], atol=1e-7, rtol=1e-7
    )

