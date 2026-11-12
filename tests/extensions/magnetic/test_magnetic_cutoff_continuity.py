"""Energy continuity at r_max for the magnetic interaction blocks."""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import MagneticScaleShiftMACE
from mace.tools.torch_tools import default_dtype

_RMAX = 4.0
_BLOCKS = [
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
]
# only the 0-2 edge crosses r_max
_POS01 = np.array([[0.0, 0.0, 0.0], [-2.0, 0.3, 0.1]])
_MAG = np.array([[0.1, 0.2, 2.2], [0.2, -0.3, 2.1], [-0.1, 0.1, 2.0]])  # ferromagnetic


def _batch(pos, mag):
    n = len(pos)
    s, d = [], []
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(pos[i] - pos[j]) < _RMAX:
                s.append(j)
                d.append(i)
    ei = torch.tensor([s, d], dtype=torch.long)
    return {
        "positions": torch.tensor(pos, requires_grad=True),
        "cell": torch.zeros(1, 3, 3),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n]),
        "node_attrs": torch.ones(n, 1),
        "edge_index": ei,
        "magmom": torch.tensor(mag),
        "unit_shifts": torch.zeros(ei.shape[1], 3),
        "shifts": torch.zeros(ei.shape[1], 3),
    }


def _build(name, seed=1):
    torch.manual_seed(seed)
    return MagneticScaleShiftMACE(
        r_max=_RMAX,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=interaction_classes[name],
        interaction_cls_first=interaction_classes[name],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=2.0,
        atomic_numbers=[26],
        correlation=[2, 2],
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=4,
        num_mag_radial_basis_one_body=4,
        max_m_ell=1,
        use_magmom_one_body=False,
    )


def _energy(model, d, mag=_MAG):
    pos = np.vstack([_POS01, [[d, 0.0, 0.0]]])
    out = model(
        _batch(pos, mag), training=False, compute_force=False, compute_stress=False
    )
    return float(out["energy"].detach())


def _step(model, mag=_MAG, eps=1e-9):
    return _energy(model, _RMAX - eps, mag) - _energy(model, _RMAX + eps, mag)


def _as_saved_before_the_fix(model):
    model.radial_embedding.apply_cutoff = True
    return model


@pytest.mark.parametrize("name", _BLOCKS)
def test_envelope_inside_the_radial_basis_steps_at_cutoff(name):
    with default_dtype(torch.float64):
        step = _step(_as_saved_before_the_fix(_build(name)))
    assert abs(step) > 1e-6, step


@pytest.mark.parametrize("name", _BLOCKS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_energy_is_continuous_at_cutoff(name, seed):
    with default_dtype(torch.float64):
        model = _build(name, seed)
        assert not model.radial_embedding.apply_cutoff
        assert abs(_step(model)) < 1e-10
        assert abs(_step(model, _MAG * 0.0)) < 1e-10
        mag = _MAG.copy()
        mag[2] *= -1
        assert abs(_step(model, mag)) < 1e-10


@pytest.mark.parametrize("name", _BLOCKS)
def test_forces_are_continuous_at_cutoff(name):
    with default_dtype(torch.float64):
        model = _build(name)
        forces = []
        for d in (_RMAX - 1e-6, _RMAX + 1e-6):
            pos = np.vstack([_POS01, [[d, 0.0, 0.0]]])
            forces.append(
                model(_batch(pos, _MAG), training=False, compute_force=True)["forces"]
            )
    assert torch.allclose(forces[0], forces[1], atol=1e-8)


@pytest.mark.parametrize("name", _BLOCKS)
def test_the_envelope_is_applied_once_outside_the_radial_basis(name):
    with default_dtype(torch.float64):
        model = _build(name)
        r = torch.tensor([[1.0], [2.5], [_RMAX - 1e-3]])
        args = (torch.ones(3, 1), torch.zeros(2, 3, dtype=torch.long))
        raw, cutoff = model.radial_embedding(r, *args, model.atomic_numbers)
        model.radial_embedding.apply_cutoff = True
        enveloped, none = model.radial_embedding(r, *args, model.atomic_numbers)
    assert none is None
    assert torch.allclose(enveloped, raw * cutoff)


@pytest.mark.parametrize("name", _BLOCKS)
def test_models_saved_before_the_fix_are_unchanged(name):
    with default_dtype(torch.float64):
        model = _as_saved_before_the_fix(_build(name))
        e = _energy(model, 3.0)
        calls = []
        for inter in model.interactions:
            inter.register_forward_pre_hook(
                lambda mod, args, kwargs: calls.append(kwargs.get("cutoff")),
                with_kwargs=True,
            )
        assert _energy(model, 3.0) == e
    assert calls and all(c is None for c in calls)
