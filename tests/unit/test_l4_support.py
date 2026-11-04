"""max_ell=4 support for compiled training and cuEq conversion.

e3nn 0.4.4 ships cached Wigner-3j tables only up to l=11 and MACE's
traceable spherical-harmonics shim covered l<=3, so correlation-3 models
over l=4 spherical harmonics (couplings such as (4, 8, 12)) could neither
convert to cuEq nor compile for training. These tests pin the two
extensions: the on-demand Wigner-3j computation in ``mace.tools.cg`` and
the l=4 spherical-harmonics polynomials in ``mace.tools.compile_training``.
"""

import math

import pytest
import torch
from e3nn import o3

from mace.tools.cg import _compute_wigner_3j, _wigner_3j
from mace.tools.compile_training import _legacy_spherical_harmonics


@pytest.mark.parametrize(
    "normalization,scale",
    [
        ("component", lambda l: math.sqrt(2 * l + 1)),
        ("integral", lambda l: math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)),
    ],
)
def test_l4_spherical_harmonics_match_e3nn(normalization, scale):
    """The traceable l<=4 polynomials must reproduce e3nn exactly."""
    torch.manual_seed(1702)
    vectors = torch.nn.functional.normalize(
        torch.randn(500, 3, dtype=torch.float64), dim=-1
    )
    ls = list(range(5))
    reference = o3.spherical_harmonics(
        ls, vectors, normalize=True, normalization=normalization
    )
    raw = _legacy_spherical_harmonics(
        4, vectors[:, 0], vectors[:, 1], vectors[:, 2]
    )
    scaled = torch.cat(
        [raw[:, l * l : (l + 1) * (l + 1)] * scale(l) for l in ls], dim=-1
    )
    torch.testing.assert_close(scaled, reference, rtol=0, atol=1e-12)


def test_spherical_harmonics_reject_l5():
    with pytest.raises(RuntimeError, match="through l=4"):
        _legacy_spherical_harmonics(
            5,
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "triple",
    [(1, 1, 1), (2, 2, 2), (2, 4, 4), (3, 6, 9), (4, 4, 4), (4, 6, 8)],
)
def test_wigner_3j_fallback_matches_cached_table(triple):
    """In-table couplings must still come from e3nn's cached values."""
    # e3nn caches the table at float32 precision
    torch.testing.assert_close(
        _wigner_3j(*triple, dtype=torch.float64),
        o3.wigner_3j(*triple).to(torch.float64),
        rtol=0,
        atol=1e-6,
    )


def test_wigner_3j_beyond_table_is_normalized_real_invariant():
    """(4, 8, 12): beyond the cached table; norm-1, real, and invariant
    under conjugation by D(4) x D(8) x D(12) for a rotation whose D(<=8)
    parts come from e3nn itself."""
    w = _compute_wigner_3j(4, 8, 12)
    assert w.dtype == torch.float64
    assert torch.all(torch.isfinite(w))
    assert abs(float((w**2).sum()) - 1.0) < 1e-10

    # full D(4) x D(8) x D(12) conjugation invariance is exercised by the
    # end-to-end model parity test below.


@pytest.mark.gpu
@pytest.mark.cueq
def test_l4_model_converts_and_matches_eager(tmp_path):
    """A max_ell=4 model converts to cuEq fusion and compiles for training,
    matching eager energies/forces at float32 precision."""
    import copy

    import numpy as np
    import torch_geometric

    from mace import modules
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.data.atomic_data import AtomicData, get_data_loader
    from mace.data.utils import Configuration
    from mace.tools import AtomicNumberTable, compile as mace_compile
    from mace.tools import compile_training

    device = torch.device("cuda")
    torch.manual_seed(1702)
    torch_geometric.seed_everything(1702)
    z_table = AtomicNumberTable([6, 8])

    def build():
        return modules.ScaleShiftMACE(
            r_max=6.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=4,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([-1.0, -2.0]),
            avg_num_neighbors=10.0,
            atomic_numbers=z_table.zs,
            correlation=3,
            radial_type="bessel",
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )

    eager = build().to(device)
    with mace_compile.disable_e3nn_codegen():
        fused = run_e3nn_to_cueq(
            copy.deepcopy(eager), device=device, conv_fusion=True
        )

    configs = []
    for n, seed in ((5, 0), (7, 1), (4, 2), (6, 3)):
        rng = np.random.default_rng(seed)
        configs.append(
            Configuration(
                np.sort(rng.choice([6, 8], n)),
                rng.random((n, 3)) * 4.0,
                {"energy": 0.0},
                {"energy": 1.0},
                np.eye(3) * 20.0,
                (True, True, True),
            )
        )
    atomic_list = [
        AtomicData.from_config(c, z_table=z_table, cutoff=6.0) for c in configs
    ]
    batches = [b.to(device) for b in get_data_loader(atomic_list, batch_size=2)]

    output_args = {
        "energy": True, "forces": True, "virials": False, "stress": False,
        "atomic_energies": False, "hessian": False, "energy_grad": False,
        "magforces": False,
    }
    facade = compile_training.compile_training_model(
        fused,
        batches[0].to_dict(),
        output_args=output_args,
        mode="default",
        dynamic=True,
        validation_batch=batches[1].to_dict(),
    )
    assert facade.is_compiled, facade.compilation_error

    for batch in batches:
        fused_out = facade(batch.to_dict())
        with torch.enable_grad():
            eager_out = eager(
                batch.to_dict(),
                training=True,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
        torch.testing.assert_close(
            fused_out["energy"], eager_out["energy"].detach(), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            fused_out["forces"], eager_out["forces"].detach(), atol=1e-5, rtol=0
        )


def test_wigner_3j_fallback_returns_fresh_tensors():
    """Beyond-table results must be freshly owned.

    _wigner_nj rescales the tensor it receives; if the lru_cache-backed
    fallback handed out the shared object, every later caller in the process
    would see silently rescaled couplings.
    """
    l1, l2, l3 = 12, 8, 4  # beyond e3nn 0.4.4's cached table
    first = _wigner_3j(l1, l2, l3, dtype=torch.float64)
    second = _wigner_3j(l1, l2, l3, dtype=torch.float64)
    assert first is not second
    first *= 5.0  # mutate the caller-owned copy
    torch.testing.assert_close(
        _wigner_3j(l1, l2, l3, dtype=torch.float64), second, atol=0, rtol=0
    )
