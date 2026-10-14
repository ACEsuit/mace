"""Regression tests for the ``num_graphs == 1`` fast path in
``get_symmetric_displacement`` (commit "fix get_symmetric_displacement slowness").

When a batch holds a single graph the per-edge gather ``cell[batch[sender]]`` is
replaced by a direct ``cell[0]`` (einsum ``"be,ec->bc"`` instead of ``"be,bec->bc"``).
That is only an optimisation, so the single-graph (new) path must give *exactly* the
same forces / stress / virials / energy as:

* the multi-graph (``else``) path, which is the original pre-commit code, and
* a finite-difference estimate of the stress.

The first comparison is the decisive regression check for the commit: the ``else``
branch is the unchanged original code, so equality proves the fast path introduces no
regression.
"""

import numpy as np
import pytest
import torch
from ase import build
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)


def _build_model_and_atoms():
    """Small ScaleShiftMACE on a strained bulk-Si cell (non-zero stress)."""
    atoms = build.bulk("Si", "diamond", a=5.43)
    # 2% strain so the stress is clearly non-zero.
    atoms.set_cell(np.dot(atoms.get_cell(), np.eye(3) * 1.02), scale_atoms=True)

    z_table = tools.AtomicNumberTable([14])
    model_config = dict(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([0.0]),
        avg_num_neighbors=4.0,
        atomic_numbers=z_table.zs,
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )
    torch.manual_seed(0)
    model = modules.ScaleShiftMACE(**model_config)
    return model, atoms, z_table


def _atomic_data(atoms, z_table, r_max=5.0):
    return data.AtomicData.from_config(
        data.config_from_atoms(
            atoms, key_specification=data.KeySpecification.from_defaults()
        ),
        z_table=z_table,
        cutoff=r_max,
    )


def _run(model, dataset, batch_size, device):
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to(device)
    return model.to(device)(
        batch.to_dict(),
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no CUDA device"
            ),
        ),
    ],
)
def test_single_graph_matches_multi_graph(device):
    """batch_size=1 (new fast path) == batch_size=2 duplicate (original else path)."""
    model, atoms, z_table = _build_model_and_atoms()
    atomic_data = _atomic_data(atoms, z_table)

    # batch of 1 -> num_graphs == 1 -> new "be,ec->bc" path
    single = _run(model, [atomic_data], batch_size=1, device=device)
    # batch of 2 identical graphs -> num_graphs == 2 -> original "be,bec->bc" path
    double = _run(model, [atomic_data, atomic_data], batch_size=2, device=device)

    # energy: graph 0 of the batch-of-2 must equal the lone graph
    assert torch.allclose(single["energy"][0], double["energy"][0], atol=1e-10)
    assert torch.allclose(double["energy"][0], double["energy"][1], atol=1e-10)

    # forces: the single graph's atoms are the first N atoms of the batch-of-2
    n = atoms.get_global_number_of_atoms()
    assert torch.allclose(single["forces"], double["forces"][:n], atol=1e-10)

    # stress / virials: per-graph, graph 0 must match the single-graph result
    assert torch.allclose(single["stress"][0], double["stress"][0], atol=1e-10)
    assert torch.allclose(single["stress"][0], double["stress"][1], atol=1e-10)
    assert torch.allclose(single["virials"][0], double["virials"][0], atol=1e-10)


def test_single_graph_stress_matches_finite_difference():
    """Absolute check: single-graph stress matches a finite-difference of the energy.

    MACE deforms positions/cell by the deformation gradient ``F = I + s`` (right
    multiplication) where ``s`` is the symmetric displacement, and reports
    ``stress = (1/V) dE/ds``.  For a fixed symmetric direction ``D`` and scalar ``t``
    (``s = t * D``) this gives ``dE/dt = V0 * sum(stress * D)``, which we compare to a
    central finite difference.
    """
    model, atoms, z_table = _build_model_and_atoms()

    ref_cell = np.array(atoms.get_cell())
    ref_pos = atoms.get_positions()
    volume = atoms.get_volume()

    def energy_at_strain(s):
        deformed = atoms.copy()
        F = np.eye(3) + s
        deformed.set_cell(ref_cell @ F.T, scale_atoms=False)
        deformed.set_positions(ref_pos @ F.T)
        out = _run(model, [_atomic_data(deformed, z_table)], batch_size=1, device="cpu")
        return out["energy"][0].item()

    stress = _run(model, [_atomic_data(atoms, z_table)], batch_size=1, device="cpu")[
        "stress"
    ][0]

    rng = np.random.default_rng(1)
    delta = 1e-5
    for _ in range(3):
        a = rng.normal(size=(3, 3))
        D = 0.5 * (a + a.T)  # symmetric direction
        e_plus = energy_at_strain(delta * D)
        e_minus = energy_at_strain(-delta * D)
        fd_slope = (e_plus - e_minus) / (2 * delta)
        predicted_slope = volume * float(
            torch.sum(stress * torch.as_tensor(D, dtype=stress.dtype))
        )
        assert np.isclose(fd_slope, predicted_slope, rtol=1e-5, atol=1e-6), (
            f"FD slope {fd_slope} != predicted {predicted_slope}"
        )
