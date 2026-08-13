from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from e3nn import o3

from mace.data.atomic_data import AtomicData
from mace.tools import AtomicNumberTable
from mace.data.utils import config_from_atoms
from mace.data.rigid_features import rigid_feature_spec
from mace.modules.models import _rigid_feature_data_keys


def _atoms_with_rank2_arrays(quaternion):
    atoms = Atoms("XX", positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

    atoms.arrays["quaternions"] = np.asarray(
        [quaternion, quaternion],
        dtype=float,
    )
    atoms.arrays["c_diameter[1]"] = np.asarray([2.0, 2.0])
    atoms.arrays["c_diameter[2]"] = np.asarray([3.0, 3.0])
    atoms.arrays["c_diameter[3]"] = np.asarray([4.0, 4.0])

    atoms.arrays["c_gyration[1]"] = np.asarray([1.0, 1.0])
    atoms.arrays["c_gyration[2]"] = np.asarray([2.0, 2.0])
    atoms.arrays["c_gyration[3]"] = np.asarray([5.0, 5.0])

    atoms.arrays["c_steric_extent[1]"] = np.asarray([4.0, 4.0])
    atoms.arrays["c_steric_extent[2]"] = np.asarray([9.0, 9.0])
    atoms.arrays["c_steric_extent[3]"] = np.asarray([16.0, 16.0])

    atoms.arrays["c_quadrupole[1]"] = np.asarray([-2.0, -2.0])
    atoms.arrays["c_quadrupole[2]"] = np.asarray([0.5, 0.5])
    atoms.arrays["c_quadrupole[3]"] = np.asarray([1.5, 1.5])

    return atoms


def _atomic_data(atoms):
    config = config_from_atoms(
        atoms,
        config_type_weights={"Default": 1.0},
    )
    return AtomicData.from_config(
        config,
        z_table=AtomicNumberTable([0]),
        cutoff=5.0,
        heads=["Default"],
    )


@pytest.mark.parametrize(
    "mode,irreps_key,tensor_key",
    [
        ("moi", "inertia_irreps", "inertia_tensor"),
        ("gyration", "gyration_irreps", "gyration_tensor"),
        ("steric_extent", "steric_extent_irreps", "steric_extent_tensor"),
        (
            "electrostatic_quadrupole",
            "electrostatic_quadrupole_irreps",
            "electrostatic_quadrupole_tensor",
        ),
    ],
)
def test_rank2_modes_have_expected_keys_and_dimensions(mode, irreps_key, tensor_key):
    spec = rigid_feature_spec(mode)
    assert spec.irreps == "1x0e + 1x2e"
    assert spec.dimension == 6

    selected_irreps_key, selected_tensor_key = _rigid_feature_data_keys(mode)
    assert selected_irreps_key == irreps_key
    assert selected_tensor_key == tensor_key

    data = _atomic_data(_atoms_with_rank2_arrays([1.0, 0.0, 0.0, 0.0]))

    assert irreps_key in data
    assert tensor_key in data
    assert data[irreps_key].shape == (2, 6)
    assert data[tensor_key].shape == (2, 3, 3)


@pytest.mark.parametrize(
    "irreps_key,tensor_key",
    [
        ("gyration_irreps", "gyration_tensor"),
        ("steric_extent_irreps", "steric_extent_tensor"),
        ("electrostatic_quadrupole_irreps", "electrostatic_quadrupole_tensor"),
    ],
)
def test_rank2_features_rotate_with_quaternion(irreps_key, tensor_key):
    identity = _atomic_data(_atoms_with_rank2_arrays([1.0, 0.0, 0.0, 0.0]))

    # 90 degree rotation around z, scalar-first quaternion.
    qz90 = [
        np.cos(np.pi / 4.0),
        0.0,
        0.0,
        np.sin(np.pi / 4.0),
    ]
    rotated = _atomic_data(_atoms_with_rank2_arrays(qz90))

    assert not torch.allclose(identity[tensor_key], rotated[tensor_key])
    assert not torch.allclose(identity[irreps_key], rotated[irreps_key])

    # Scalar channel should be rotation invariant.
    torch.testing.assert_close(identity[irreps_key][:, :1], rotated[irreps_key][:, :1])


def test_quadrupole_fallback_is_zero_without_quadrupole_arrays():
    atoms = _atoms_with_rank2_arrays([1.0, 0.0, 0.0, 0.0])
    for key in [
        "c_quadrupole[1]",
        "c_quadrupole[2]",
        "c_quadrupole[3]",
    ]:
        del atoms.arrays[key]

    data = _atomic_data(atoms)

    torch.testing.assert_close(
        data["electrostatic_quadrupole_tensor"],
        torch.zeros_like(data["electrostatic_quadrupole_tensor"]),
    )
    torch.testing.assert_close(
        data["electrostatic_quadrupole_irreps"],
        torch.zeros_like(data["electrostatic_quadrupole_irreps"]),
    )
