"""Real-vs-ghost parity: the core physics contract of the LAMMPS integration.

LAMMPS partitions atoms into local + ghost periodic images and expects the
per-atom energies of the local atoms (and their folded forces) to reproduce
the periodic calculation. This exercises the LAMMPS_MACE wrapper and, through
it, the lammps-aware branches of the model — code that previously had zero
test coverage and only failed in production.

No LAMMPS binary involved (contract tier); receptive field is
num_interactions * r_max = 2 * 3.5 = 7 A, so a 5x5x5 replica (+-8 A shell)
gives exact locality parity in float64.
"""

import numpy as np
import pytest
import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE
from tests.integrations.lammps._harness import (
    fold_ghost_forces,
    lammps_style_cluster,
    model_batch,
    water_unit_cell,
)


@pytest.fixture(scope="module", name="tiny_model")
def fixture_tiny_model(trained_tiny_model_path):
    torch.set_default_dtype(torch.float64)
    return torch.load(trained_tiny_model_path, map_location="cpu").double()


@pytest.fixture(scope="module", name="periodic_reference")
def fixture_periodic_reference(tiny_model):
    batch = model_batch(tiny_model, water_unit_cell())
    out = tiny_model(batch, training=False, compute_force=True)
    return out["energy"].detach(), out["forces"].detach()


def test_local_energy_matches_periodic(tiny_model, periodic_reference):
    e_ref, _ = periodic_reference
    lammps_model = LAMMPS_MACE(tiny_model)
    batch, local_or_ghost, _, _ = lammps_style_cluster(tiny_model, n_repeat=5)
    out = lammps_model(batch, local_or_ghost)
    e_local = out["total_energy_local"].detach()
    assert torch.allclose(e_local, e_ref, atol=1e-8), (
        f"local-atom energy {e_local.item():.10f} != periodic "
        f"{e_ref.item():.10f}"
    )


def test_folded_forces_match_periodic(tiny_model, periodic_reference):
    _, f_ref = periodic_reference
    lammps_model = LAMMPS_MACE(tiny_model)
    batch, local_or_ghost, image_of, _ = lammps_style_cluster(
        tiny_model, n_repeat=5
    )
    out = lammps_model(batch, local_or_ghost)
    folded = fold_ghost_forces(out["forces"].detach(), image_of, f_ref.shape[0])
    assert torch.allclose(folded, f_ref, atol=1e-8), (
        f"max |dF| = {(folded - f_ref).abs().max().item():.3e}"
    )


def test_ghost_energies_excluded(tiny_model):
    """Only the masked (local) atoms may contribute to total_energy_local."""
    lammps_model = LAMMPS_MACE(tiny_model)
    batch, local_or_ghost, _, local_idx = lammps_style_cluster(
        tiny_model, n_repeat=3
    )
    out = lammps_model(batch, local_or_ghost)
    node_sum_local = out["node_energy"][local_idx].sum()
    assert torch.allclose(out["total_energy_local"].sum(), node_sum_local, atol=1e-10)


def test_torchscript_matches_eager(tiny_model):
    """The compiled artifact LAMMPS actually loads must agree with eager."""
    lammps_model = LAMMPS_MACE(tiny_model)
    compiled = jit.compile(lammps_model)
    batch, local_or_ghost, _, _ = lammps_style_cluster(tiny_model, n_repeat=3)

    def clone(b):
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}

    out_eager = lammps_model(clone(batch), local_or_ghost)
    out_jit = compiled(clone(batch), local_or_ghost)
    assert torch.allclose(
        out_jit["total_energy_local"], out_eager["total_energy_local"], atol=1e-10
    )
    assert torch.allclose(out_jit["forces"], out_eager["forces"], atol=1e-10)


def test_virials_path_runs(tiny_model):
    """compute_virials=True exercises the strain-displacement branch."""
    lammps_model = LAMMPS_MACE(tiny_model)
    batch, local_or_ghost, _, _ = lammps_style_cluster(tiny_model, n_repeat=3)
    out = lammps_model(batch, local_or_ghost, compute_virials=True)
    assert out["virials"] is not None
    assert torch.isfinite(out["virials"]).all()
    assert torch.isfinite(out["forces"]).all()
    np.testing.assert_array_equal(out["virials"].shape[-2:], (3, 3))
