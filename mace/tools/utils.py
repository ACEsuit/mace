###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .torch_tools import to_numpy


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def _exact_closest_lattice_shift(
    fractional_coords: torch.Tensor,
    polarization_lattice: torch.Tensor,
) -> torch.Tensor:
    """
    Solve the 3D closest vector problem exactly for one or more polarization
    lattice coordinates.

    We find the integer lattice shift n* minimizing

        || (c - n) Q ||_2

    where c are fractional polarization coordinates and Q contains the
    polarization lattice basis vectors as rows. The returned shift is piecewise
    constant in c, so the final folded polarization remains differentiable
    almost everywhere when used as dP_folded = (c - n*) Q.
    """

    def _nearest_integer(value: float) -> int:
        return int(np.rint(value))

    def _babai_point(r_mat: torch.Tensor, y_tilde: torch.Tensor) -> list[int]:
        n_dim = r_mat.shape[0]
        z = [0] * n_dim
        for level in range(n_dim - 1, -1, -1):
            offset = sum(float(r_mat[level, j]) * z[j] for j in range(level + 1, n_dim))
            center = (float(y_tilde[level]) - offset) / float(r_mat[level, level])
            z[level] = _nearest_integer(center)
        return z

    def _residual_sq(r_mat: torch.Tensor, y_tilde: torch.Tensor, z: list[int]) -> float:
        residual = r_mat @ torch.tensor(z, dtype=r_mat.dtype)
        diff = residual - y_tilde
        return float(torch.dot(diff, diff))

    fractional_cpu = fractional_coords.detach().to(device="cpu", dtype=torch.float64)
    lattice_cpu = polarization_lattice.detach().to(device="cpu", dtype=torch.float64)
    shifts = []
    tolerance = 1e-12

    for c_vec, q_rows in zip(fractional_cpu, lattice_cpu):
        target = (c_vec @ q_rows).to(torch.float64)
        basis = q_rows.transpose(0, 1).contiguous()
        q_orth, r_mat = torch.linalg.qr(basis)
        y_tilde = q_orth.transpose(0, 1) @ target

        diag = torch.diagonal(r_mat)
        signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
        r_mat = signs.unsqueeze(-1) * r_mat
        y_tilde = signs * y_tilde

        initial = _babai_point(r_mat, y_tilde)
        best = initial[:]
        best_dist = _residual_sq(r_mat, y_tilde, best)
        current = initial[:]
        n_dim = r_mat.shape[0]

        def recurse(level: int, partial_dist: float) -> None:
            nonlocal best, best_dist, current

            if partial_dist > best_dist + tolerance:
                return
            if level < 0:
                best = current[:]
                best_dist = partial_dist
                return

            r_diag = float(r_mat[level, level])
            offset = sum(
                float(r_mat[level, j]) * current[j] for j in range(level + 1, n_dim)
            )
            center = (float(y_tilde[level]) - offset) / r_diag
            remaining = max(best_dist - partial_dist, 0.0)
            radius = math.sqrt(remaining) / abs(r_diag)
            z_min = math.ceil(center - radius)
            z_max = math.floor(center + radius)

            candidates = sorted(
                range(z_min, z_max + 1),
                key=lambda val: abs(val - center),
            )
            for z_val in candidates:
                diff = r_diag * z_val + offset - float(y_tilde[level])
                next_dist = partial_dist + diff * diff
                if next_dist <= best_dist + tolerance:
                    current[level] = z_val
                    recurse(level - 1, next_dist)

        recurse(n_dim - 1, 0.0)
        shifts.append(best)

    return torch.tensor(
        shifts,
        device=fractional_coords.device,
        dtype=fractional_coords.dtype,
    )


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: Optional[int] = 0,
):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add filter for rank
    logger.addFilter(lambda _: rank == 0)

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(lambda record: record.levelno >= logging.DEBUG)
        logger.addHandler(fh_debug)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")


# pylint: disable=abstract-method, arguments-differ
class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


def get_cache_dir() -> Path:
    # get cache dir from XDG_CACHE_HOME if set, otherwise appropriate default
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mace"


def filter_nonzero_weight(
    batch,
    quantity_l,
    weight,
    quantity_weight,
    spread_atoms=False,
    spread_quantity_vector=True,
) -> float:
    quantity = quantity_l[-1]
    # repeat with interleaving for per-atom quantities
    if spread_atoms:
        weight = torch.repeat_interleave(
            weight, batch.ptr[1:] - batch.ptr[:-1], dim=0
        ).unsqueeze(-1)
        quantity_weight = torch.repeat_interleave(
            quantity_weight, batch.ptr[1:] - batch.ptr[:-1], dim=0
        )

    # repeat for additional dimensions
    if len(quantity.shape) > 1:
        repeats = [1] + list(quantity.shape[1:])
        view = [-1] + [1] * (len(quantity.shape) - 1)
        weight = weight.view(*view).repeat(*repeats)
        if spread_quantity_vector:
            quantity_weight = quantity_weight.view(*view).repeat(*repeats)

    filtered_q = quantity[weight * quantity_weight > 0]

    if len(filtered_q) == 0:
        quantity_l.pop()
        return 0.0

    quantity_l[-1] = filtered_q
    return 1.0


def fold_polarization(
    pred_polarization: torch.Tensor,  # [n_graphs, 3] intensive P_pred
    ref_polarization: torch.Tensor,  # [n_graphs, 3] intensive P_ref  (branch anchor)
    cell: torch.Tensor,  # [n_graphs, 3, 3] ASE-style cell (rows = lattice vectors)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the nearest-image *difference* ΔP_folded to add to P_ref:
        P_pred_fold = P_ref + ΔP_folded
    with ΔP identified modulo the polarization lattice Qpol = cell / |Ω|.

    This is the exact closest-vector problem on the polarization lattice:
    we choose the integer lattice shift that gives the smallest Cartesian norm.
    The mapping is therefore exact for any full-rank crystal cell and remains
    differentiable almost everywhere.
    """
    B = cell.view(-1, 3, 3)
    vol = torch.linalg.det(B).abs().clamp_min(1e-30).view(-1, 1, 1)
    Q = B / vol

    dP = pred_polarization.view(-1, 3) - ref_polarization.view(-1, 3)
    c = torch.linalg.solve(Q.transpose(-2, -1), dP.unsqueeze(-1)).squeeze(-1)

    integer_shift = _exact_closest_lattice_shift(c, Q)
    c_folded = c - integer_shift
    dP_folded = dP - torch.einsum("bi,bij->bj", integer_shift, Q)
    return dP_folded, c_folded
