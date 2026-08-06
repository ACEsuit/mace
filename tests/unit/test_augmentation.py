"""Tests for the magnetic data augmentation used to train non-SOC models.

--data_aug_magmom is not a regularizer here: it is how a spin-orbit-coupled
architecture is taught the non-SOC symmetry. It rotates the magnetic moments
while leaving the positions alone, so over training the model sees every
orientation of the spins relative to the same lattice and learns that the
energy does not depend on it.

These cover the rotation itself. That it does not disturb the stored dataset is
a property of the loader path, not of `forward`, and is tested there: see
test_random_rotation_loader_over_real_atomic_data in the magnetic suite.
"""

import numpy as np
import pytest
import torch

from mace.data.augmentation import Random3DRotation


class _Sample:
    """Minimal carrier for the two fields the rotation touches."""

    magforces = None

    def __init__(self, magmom, magforces=None):
        self.magmom = magmom
        self.magforces = magforces


def test_rotation_keeps_magmom_and_magforces_consistent():
    """magmom is an input and magforces (dE/dm) is its conjugate label.

    Both must turn by the same R or the pair stops describing the same physical
    state. This is the property that makes repeated rotation harmless.
    """
    torch.manual_seed(0)
    m = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    f = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

    out = Random3DRotation().forward(_Sample(m.clone(), f.clone()))

    # One rotation applied to every row preserves all inner products, including
    # the cross terms between moments and forces. Had the two been turned by
    # different rotations, those cross terms would move while the lengths
    # within each block still looked right.
    before = torch.cat([m, f]) @ torch.cat([m, f]).T
    after = (
        torch.cat([out.magmom, out.magforces])
        @ torch.cat([out.magmom, out.magforces]).T
    )
    assert torch.allclose(after, before, atol=1e-5)


@pytest.mark.parametrize("n_draws", [1, 5])
def test_rotation_samples_orientations_uniformly(n_draws):
    """Every epoch must present a uniformly random orientation.

    Acting on a fixed unit vector, a uniform SO(3) rotation gives a z-component
    uniform on [-1, 1], so std = 1/sqrt(3). Checked after repeated draws too:
    a product of uniform rotations is still uniform, which is why it would not
    matter even if rotations did accumulate across epochs.
    """
    torch.manual_seed(0)
    transform = Random3DRotation()
    z = []
    for _ in range(4000):
        sample = _Sample(torch.tensor([[0.0, 0.0, 1.0]]))
        for _ in range(n_draws):
            sample = transform.forward(sample)
        z.append(sample.magmom[0, 2].item())

    z = np.array(z)
    assert abs(z.mean()) < 0.03
    assert abs(z.std() - 1.0 / np.sqrt(3.0)) < 0.03
