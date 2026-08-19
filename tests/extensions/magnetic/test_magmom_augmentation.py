import types

import pytest
import torch

from mace.data.augmentation import Random3DRotation
from mace.tools.torch_tools import default_dtype

# ----------------------------------------------------------
# Which symmetries each mode augments
# ----------------------------------------------------------
# non-soc: the full O(3)_spin -- a random rotation AND a random global sign, because a
# non-SOC energy is invariant under rotating the moments independently of the lattice and
# is even under time reversal.
# soc: ONLY the sign. With spin-orbit coupling a free spin rotation is not a symmetry, so
# augmenting with it would teach an invariance the model must not have; time reversal
# still holds at zero field.


def _sample(seed=0, n=5):
    g = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        magmom=torch.randn(n, 3, dtype=torch.float64, generator=g),
        magforces=torch.randn(n, 3, dtype=torch.float64, generator=g),
    )


def _applied_transform(before, after):
    """Recover the 3x3 map T with after = before @ T.T (least squares)."""
    return torch.linalg.lstsq(before, after).solution.T


@pytest.mark.parametrize("mode", ["soc", "non-soc"])
def test_moment_magnitudes_are_preserved(mode):
    """Both modes apply an orthogonal map, so |m| per atom must not change."""
    with default_dtype(torch.float64):
        transform = Random3DRotation(mode=mode)
        for seed in range(8):
            data = _sample(seed)
            before = data.magmom.norm(dim=1).clone()
            out = transform.forward(_sample(seed))
            assert torch.allclose(out.magmom.norm(dim=1), before, atol=1e-12)


def test_non_soc_mode_samples_the_full_o3():
    """Both signs of det must appear: a proper rotation and its improper partner."""
    with default_dtype(torch.float64):
        transform = Random3DRotation(mode="non-soc")
        dets = set()
        for seed in range(40):
            torch.manual_seed(seed)
            data = _sample(seed)
            out = transform.forward(_sample(seed))
            T = _applied_transform(data.magmom, out.magmom)
            dets.add(int(round(float(torch.det(T)))))
        assert dets == {-1, 1}, f"expected both det = +-1 over O(3), saw {sorted(dets)}"


def test_soc_mode_applies_only_a_sign_flip():
    """With SOC the rotation is not a symmetry: the moments may only be negated."""
    with default_dtype(torch.float64):
        transform = Random3DRotation(mode="soc")
        seen = set()
        for seed in range(40):
            torch.manual_seed(seed)
            data = _sample(seed)
            out = transform.forward(_sample(seed))
            if torch.allclose(out.magmom, data.magmom, atol=1e-12):
                seen.add(+1)
            elif torch.allclose(out.magmom, -data.magmom, atol=1e-12):
                seen.add(-1)
            else:
                pytest.fail("soc mode rotated the moments; only +-m is permitted")
        assert seen == {-1, 1}, f"expected both +m and -m to occur, saw {sorted(seen)}"


@pytest.mark.parametrize("mode", ["soc", "non-soc"])
def test_magforces_follow_the_moments(mode):
    """F_M = -dE/dM lives in spin space, so it takes the same map as the moments.

    In particular a sign flip must send F_M -> -F_M, or the augmented sample would be
    internally inconsistent.
    """
    with default_dtype(torch.float64):
        transform = Random3DRotation(mode=mode)
        for seed in range(8):
            torch.manual_seed(seed)
            data = _sample(seed)
            out = transform.forward(_sample(seed))
            T = _applied_transform(data.magmom, out.magmom)
            assert torch.allclose(out.magforces, data.magforces @ T.T, atol=1e-10)


def test_samples_without_moments_pass_through():
    data = types.SimpleNamespace(magmom=None, magforces=None)
    assert Random3DRotation().forward(data).magmom is None


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        Random3DRotation(mode="both")
