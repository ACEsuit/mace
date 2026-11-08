"""Framework-independent statistics shared by preprocessing and in-memory loading.

Isolated-atom energies arrive already resolved. Interaction-energy statistics
use the same E0 values stored in the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mace_core.config.model import ScalingMethod
from mace_core.data.backend import DataBackend, DatasetStatistics
from mace_core.neighbors import AllPairsBackend, get_neighborhood

__all__ = [
    "compute_statistics",
    "count_neighbours",
]

Scaling = ScalingMethod


def root_mean_square(values) -> float:
    """Scalar RMS over samples and components; also the spread of centred data."""
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def count_neighbours(
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: Sequence[bool] | None,
    cutoff: float,
) -> int:
    """How many directed edges a structure has at ``cutoff``.

    Uses the graph's reference neighbour search and self-edge filtering.
    Periodic self images count; a self pair at zero shift does not.

    Args:
        positions: ``[n_atoms, 3]`` in Angstrom.
        cell: ``[3, 3]`` lattice vectors as rows, or ``None``.
        pbc: Which directions are periodic, or ``None`` for none.
        cutoff: In Angstrom.

    Returns:
        The directed edge count, so each neighbour pair contributes twice.
    """
    neighbors = get_neighborhood(
        positions, cutoff, pbc=pbc, cell=cell, backend=AllPairsBackend()
    )
    return neighbors.edge_index.shape[1]


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
        scaling: Which spread to report as ``std``. ``"none"`` measures only
            neighbour counts and uses a unit scale for models without energy.
            ``"rms_forces"`` is the frozen tree's default model behaviour,
            ``"std"`` the population spread of per-atom interaction energies,
            ``"rms_dipoles"`` for a dipole
            model. Named rather than inferred, because on the frozen tree the
            choice is spread across a command-line flag and a model default
            that disagree.
        energy_key, forces_key, dipole_key: Which property carries what.

    Returns:
        The statistics, with ``mean`` always the mean per-atom interaction
        energy and ``std`` whatever ``scaling`` asked for. Neighbour counts
        include atoms with no edges. Both conventions are the native V1
        contract; legacy uses sample spread and omits zero-neighbour atoms.
        A zero spread is retained here; the output layer uses a unit scale
        for it so a constant target does not freeze the readout.

    Raises:
        ValueError: On an unknown scaling, or when the chosen one has no data
            to compute from. A spread of 1.0 returned quietly would divide the
            model's outputs by a number that means nothing.
    """
    if scaling not in ("none", "std", "rms_forces", "rms_dipoles"):
        raise ValueError(
            f"{scaling!r} is not a scaling this computes. The choices are "
            f"'none', 'std', 'rms_forces' and 'rms_dipoles'."
        )

    per_atom_energies: list[float] = []
    force_samples: list[np.ndarray] = []
    dipole_samples: list[np.ndarray] = []
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
            force_samples.append(np.asarray(forces, dtype=float).reshape(-1))
        dipole = configuration.properties.get(dipole_key)
        if dipole is not None:
            dipole_samples.append(np.asarray(dipole, dtype=float).reshape(-1))

    mean = float(np.mean(per_atom_energies)) if per_atom_energies else 0.0
    if scaling == "none":
        spread = 1.0
    elif scaling == "std":
        if not per_atom_energies:
            raise ValueError(
                f"no configuration carries a {energy_key!r}, so the energy "
                f"spread cannot be computed."
            )
        spread = root_mean_square(np.asarray(per_atom_energies) - mean)
    elif scaling == "rms_forces":
        if not force_samples:
            raise ValueError(
                f"no configuration carries a {forces_key!r}, so the force RMS "
                f"cannot be computed. Choose scaling='std' if this dataset has "
                f"energies only."
            )
        spread = root_mean_square(np.concatenate(force_samples))
    else:
        if not dipole_samples:
            raise ValueError(
                f"no configuration carries a {dipole_key!r}, so the dipole RMS "
                f"cannot be computed."
            )
        spread = root_mean_square(np.concatenate(dipole_samples))

    return DatasetStatistics(
        atomic_energies={int(z): float(e) for z, e in atomic_energies.items()},
        avg_num_neighbors=(edges / atoms) if atoms else 0.0,
        mean=mean,
        std=spread,
        atomic_numbers=sorted(int(z) for z in atomic_numbers),
        r_max=float(r_max),
    )
