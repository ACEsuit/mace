from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from e3nn import o3
from scipy.spatial.transform import Rotation

from mace import data, modules, tools

DTYPE = torch.float64
ATOL = 1.0e-7
RTOL = 1.0e-7

TABLE = tools.AtomicNumberTable([0])
CUTOFF = 5.0


def _wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([w, x, y, z], dtype=float)


def _atoms(
    body_i: Rotation,
    body_j: Rotation,
    positions=None,
):
    if positions is None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.1, 0.7, -0.4],
            ],
            dtype=float,
        )

    atoms = Atoms(
        "XX",
        positions=np.asarray(positions, dtype=float),
    )

    atoms.arrays["quaternions"] = np.stack(
        [
            _wxyz(body_i),
            _wxyz(body_j),
        ]
    )

    # Supply the same rigid-body fields already used by the existing
    # rank-2 AtomicData tests. Even with rigid_feature_mode="none",
    # the current MACE data path still constructs these tensors.
    atoms.arrays["c_diameter[1]"] = np.asarray([2.0, 2.0])
    atoms.arrays["c_diameter[2]"] = np.asarray([3.0, 3.0])
    atoms.arrays["c_diameter[3]"] = np.asarray([4.0, 4.0])

    return atoms


def _batch(atoms):
    config = data.config_from_atoms(
        atoms,
        config_type_weights={"Default": 1.0},
    )

    graph = data.AtomicData.from_config(
        config,
        z_table=TABLE,
        cutoff=CUTOFF,
        heads=["Default"],
    )

    loader = tools.torch_geometric.dataloader.DataLoader(
        dataset=[graph],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(loader)).to_dict()

    # The model below is explicitly converted to DTYPE. AtomicData/PyG
    # may otherwise leave fields such as node_attrs in float32, which
    # causes an unrelated dtype failure in the ordinary MACE embedding
    # before the rigid-pair path is reached.
    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype=DTYPE)

    return batch


def _model(rigid_pair_mode: str):
    torch.manual_seed(1234)

    model = modules.ScaleShiftMACE(
        r_max=CUTOFF,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(
            "8x0e + 8x1o + 8x2e"
        ),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=F.silu,
        atomic_energies=np.asarray([0.0]),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        rigid_feature_mode="none",
        rigid_pair_mode=rigid_pair_mode,
    )

    return model.to(dtype=DTYPE)


def test_none_mode_has_no_rigid_pair_embedding():
    model = _model("none")

    assert model.rigid_pair_mode == "none"
    assert not hasattr(
        model,
        "rigid_pair_edge_embedding",
    )


def test_full_frame_mode_constructs_rigid_pair_embedding():
    model = _model("full_frame")

    assert model.rigid_pair_mode == "full_frame"
    assert hasattr(
        model,
        "rigid_pair_edge_embedding",
    )

    assert (
        model.rigid_pair_edge_embedding.edge_irreps
        == model.spherical_harmonics.irreps_out
    )


def test_none_mode_is_orientation_blind():
    model = _model("none")
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_full_frame_mode_detects_neighbor_rotation():
    model = _model("full_frame")
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    delta = torch.max(
        torch.abs(
            out_a["energy"] - out_b["energy"]
        )
    )

    assert delta > 1.0e-10


def test_full_frame_energy_is_globally_rotation_invariant():
    model = _model("full_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_full_frame_forces_rotate_covariantly():
    model = _model("full_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=True,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=True,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    expected = out_a["forces"] @ S.T

    torch.testing.assert_close(
        out_b["forces"],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_rigid_pair_projection_gets_gradient():
    model = _model("full_frame")
    model.train()

    body_i = Rotation.identity()
    body_j = Rotation.from_rotvec(
        np.asarray([0.2, 0.4, -0.1])
    )

    output = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )

    loss = output["energy"].sum()
    loss.backward()

    parameters = list(
        model.rigid_pair_edge_embedding
        .projection.parameters()
    )

    assert parameters
    assert all(
        parameter.grad is not None
        for parameter in parameters
    )

    total_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in parameters
    )

    assert total_gradient > 0.0
