"""One statistics implementation, over any backend, with no framework.

The frozen tree has two that must agree and are written differently: an
in-memory pass over a torch dataloader (``mace/modules/utils.py:476-654``) and
``pool_compute_stats`` inside preprocessing
(``mace/cli/preprocess_data.py:60-85``). Reconciling two implementations is a
standing cost; there is one here, and both the preparation command and the
in-memory path call it.

The isolated-atom energies are **passed in already resolved**. That is the
point of the signature: the interaction-energy mean and spread are then always
taken against the same values that become the model's buffer, whether they came
from isolated-atom structures, a file, a foundation model or the least-squares
fit below.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np

from mace_core.data.backend import DataBackend, DatasetStatistics
from mace_core.data.configuration import Configuration

__all__ = [
    "compute_statistics",
    "count_neighbours",
    "least_squares_atomic_energies",
]

Scaling = Literal["std", "rms_forces", "rms_dipoles"]


def _shift_counts(cell: np.ndarray, pbc: Sequence[bool], cutoff: float) -> np.ndarray:
    """How many periodic images to search along each lattice vector.

    The spacing between lattice planes normal to reciprocal vector ``b`` is
    ``1 / |b|``, so covering ``cutoff`` needs ``ceil(cutoff * |b|)`` of them.
    Deriving it rather than guessing a constant is what keeps the count right
    for a thin or a skewed cell, where a fixed number of images silently misses
    neighbours.
    """
    counts = np.zeros(3, dtype=int)
    if cell is None or not np.any(pbc):
        return counts
    volume = abs(float(np.linalg.det(cell)))
    if volume < 1e-12:
        return counts
    reciprocal = np.linalg.inv(cell).T
    for axis in range(3):
        if pbc[axis]:
            spacing = float(np.linalg.norm(reciprocal[axis]))
            counts[axis] = math.ceil(cutoff * spacing)
    return counts


def count_neighbours(
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: Sequence[bool] | None,
    cutoff: float,
) -> int:
    """How many directed edges a structure has at ``cutoff``.

    Brute force over the periodic images, which is the oracle definition of the
    edge set rather than the fast one. A self pair at zero shift is excluded;
    a self pair at a non-zero shift is a real neighbour and is counted, which is
    what makes a small cell with a large cutoff come out right.

    Args:
        positions: ``[n_atoms, 3]`` in Angstrom.
        cell: ``[3, 3]`` lattice vectors as rows, or ``None``.
        pbc: Which directions are periodic, or ``None`` for none.
        cutoff: In Angstrom.

    Returns:
        The directed edge count, so each neighbour pair contributes twice.
    """
    positions = np.asarray(positions, dtype=float)
    periodic = (
        (False, False, False) if pbc is None else tuple(bool(flag) for flag in pbc)
    )
    lattice = None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3)
    counts = _shift_counts(lattice, periodic, cutoff)

    total = 0
    ranges = [range(-counts[axis], counts[axis] + 1) for axis in range(3)]
    for a in ranges[0]:
        for b in ranges[1]:
            for c in ranges[2]:
                shift = np.array([a, b, c], dtype=float)
                offset = shift @ lattice if lattice is not None else np.zeros(3)
                deltas = positions[None, :, :] + offset - positions[:, None, :]
                distances = np.linalg.norm(deltas, axis=-1)
                within = (distances <= cutoff) & (distances > 1e-10)
                total += int(np.count_nonzero(within))
    return total


def least_squares_atomic_energies(
    configurations: Iterable[Configuration],
    atomic_numbers: Sequence[int],
    energy_key: str = "energy",
) -> dict[int, float]:
    """Fit isolated-atom energies by least squares over the dataset.

    Solves ``counts @ e0 = energies`` for the per-element reference, where
    ``counts[i, z]`` is how many atoms of element ``z`` structure ``i`` has.
    One of the ways a caller may produce the energies this module's statistics
    are then taken against; it is deliberately not the only one.

    Raises:
        ValueError: If no structure carries an energy, since the fit would
            otherwise return zeros that look like a result.
    """
    order = list(atomic_numbers)
    index = {number: position for position, number in enumerate(order)}
    rows, targets = [], []
    for configuration in configurations:
        energy = configuration.properties.get(energy_key)
        if energy is None:
            continue
        row = np.zeros(len(order))
        for number in configuration.atomic_numbers:
            row[index[int(number)]] += 1.0
        rows.append(row)
        targets.append(float(energy))
    if not rows:
        raise ValueError(
            f"no configuration carries a {energy_key!r}, so the isolated-atom "
            f"energies cannot be fitted. Supply them, or read them from "
            f"isolated-atom structures."
        )
    solution, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(targets), rcond=None)
    return {number: float(value) for number, value in zip(order, solution, strict=True)}


def compute_statistics(
    backend: DataBackend,
    atomic_numbers: Sequence[int],
    r_max: float,
    atomic_energies: Mapping[int, float],
    *,
    scaling: Scaling = "rms_forces",
    energy_key: str = "energy",
    forces_key: str = "forces",
    dipole_key: str = "dipole",
) -> DatasetStatistics:
    """Everything a model needs from its data, in one pass.

    Args:
        backend: Any data backend. Only the Protocol is used, so a third-party
            format gets this for free.
        atomic_numbers: The element table.
        r_max: The cutoff, in Angstrom.
        atomic_energies: Isolated-atom references, already resolved.
        scaling: Which spread to report as ``std``. ``"rms_forces"`` is the
            frozen tree's default model behaviour, ``"std"`` the spread of the
            per-atom interaction energy itself, ``"rms_dipoles"`` for a dipole
            model. Named rather than inferred, because on the frozen tree the
            choice is spread across a command-line flag and a model default
            that disagree.
        energy_key, forces_key, dipole_key: Which property carries what.

    Returns:
        The statistics, with ``mean`` always the mean per-atom interaction
        energy and ``std`` whatever ``scaling`` asked for.

    Raises:
        ValueError: On an unknown scaling, or when the chosen one has no data
            to compute from. A spread of 1.0 returned quietly would divide the
            model's outputs by a number that means nothing.
    """
    if scaling not in ("std", "rms_forces", "rms_dipoles"):
        raise ValueError(
            f"{scaling!r} is not a scaling this computes. The choices are "
            f"'std', 'rms_forces' and 'rms_dipoles'."
        )

    per_atom_energies: list[float] = []
    force_squares: list[np.ndarray] = []
    dipole_squares: list[np.ndarray] = []
    edges = 0
    atoms = 0

    for configuration in backend.iter_range():
        numbers = np.asarray(configuration.atomic_numbers, dtype=int)
        atoms += int(numbers.size)
        edges += count_neighbours(
            configuration.positions, configuration.cell, configuration.pbc, r_max
        )
        energy = configuration.properties.get(energy_key)
        if energy is not None:
            reference = sum(float(atomic_energies.get(int(z), 0.0)) for z in numbers)
            per_atom_energies.append((float(energy) - reference) / max(numbers.size, 1))
        forces = configuration.properties.get(forces_key)
        if forces is not None:
            force_squares.append(np.asarray(forces, dtype=float).reshape(-1) ** 2)
        dipole = configuration.properties.get(dipole_key)
        if dipole is not None:
            dipole_squares.append(np.asarray(dipole, dtype=float).reshape(-1) ** 2)

    mean = float(np.mean(per_atom_energies)) if per_atom_energies else 0.0
    if scaling == "std":
        if not per_atom_energies:
            raise ValueError(
                f"no configuration carries a {energy_key!r}, so the energy "
                f"spread cannot be computed."
            )
        spread = float(np.std(per_atom_energies))
    elif scaling == "rms_forces":
        if not force_squares:
            raise ValueError(
                f"no configuration carries a {forces_key!r}, so the force RMS "
                f"cannot be computed. Choose scaling='std' if this dataset has "
                f"energies only."
            )
        spread = float(np.sqrt(np.mean(np.concatenate(force_squares))))
    else:
        if not dipole_squares:
            raise ValueError(
                f"no configuration carries a {dipole_key!r}, so the dipole RMS "
                f"cannot be computed."
            )
        spread = float(np.sqrt(np.mean(np.concatenate(dipole_squares))))

    return DatasetStatistics(
        atomic_energies={int(z): float(e) for z, e in atomic_energies.items()},
        avg_num_neighbors=(edges / atoms) if atoms else 0.0,
        mean=mean,
        std=spread,
        atomic_numbers=sorted(int(z) for z in atomic_numbers),
        r_max=float(r_max),
    )
