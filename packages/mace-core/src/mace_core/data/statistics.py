"""What a model needs to know about its training set, measured once.

The isolated-atom energies arrive already resolved, whatever their origin, so
the interaction energies are measured against the values that become the
model's E0 buffer.

**One head per call, by index.** The statistics read one head's structures at
the indices the caller passes, usually its training split, in whichever
backend the run reads from. Nothing is copied out of the backend. The
model-wide neighbour count over several heads is combined from their counts by
:func:`joint_avg_num_neighbors`.

**A quantity is measured over the structures labelled with it**, as
:meth:`~mace_core.data.configuration.Configuration.is_labelled` reports them,
never as a weight suggests. The frozen tree averages its zero-filled missing
labels in instead, so the two agree on a fully labelled set only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated

import numpy as np
from pydantic import Field, FiniteFloat, NonNegativeInt, PositiveInt, model_validator

from mace_core.config.base import ConfigSection
from mace_core.data.backend import DataBackend, DataBackendError
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.neighbors import NeighborListBackend, get_neighborhood

logger = logging.getLogger(__name__)

#: A measured spread or RMS: finite and never negative.
_Spread = Annotated[float, Field(ge=0, allow_inf_nan=False)]

__all__ = [
    "DatasetStatistics",
    "compute_statistics",
    "fit_atomic_energies",
    "joint_avg_num_neighbors",
]


class DatasetStatistics(ConfigSection):
    """What one head's training set looks like, as measured.

    Every value is measured, never adjusted: which one a model normalises by,
    and any explicit override, is the model configuration's choice. A
    quantity no structure in the set is labelled with is ``None``.

    Args:
        cutoff: The neighbour cutoff the counts were taken at, in Angstrom.
        atomic_energies: ``{atomic number: eV}``, the isolated-atom energies
            the interaction energies were measured against.
        n_edges: Directed edges over the set; each neighbour pair counts
            twice.
        n_receivers: Atoms that receive at least one edge. An atom without a
            neighbour is not counted, as the frozen tree does not count it.
            Never more than ``n_edges``.
        energy_mean: Mean per-atom interaction energy, ``(E - sum E0) /
            n_atoms``, in eV/atom.
        energy_std: Its sample standard deviation (``ddof=1``), in eV/atom;
            ``None`` below two labelled structures.
        rms_forces: Root mean square over every force component, in eV/A.
        rms_dipoles: Root mean square over every dipole component, in e A.
    """

    cutoff: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    atomic_energies: dict[PositiveInt, FiniteFloat]
    n_edges: NonNegativeInt
    n_receivers: NonNegativeInt
    energy_mean: FiniteFloat | None = None
    energy_std: _Spread | None = None
    rms_forces: _Spread | None = None
    rms_dipoles: _Spread | None = None

    @model_validator(mode="after")
    def _every_receiver_has_an_edge(self) -> DatasetStatistics:
        if self.n_receivers > self.n_edges:
            raise ValueError(
                f"n_receivers={self.n_receivers} exceeds n_edges={self.n_edges}; "
                f"an atom counts as a receiver only through an edge."
            )
        return self

    @property
    def avg_num_neighbors(self) -> float:
        """Mean neighbour count over the atoms that have one.

        Raises:
            ValueError: if no atom has a neighbour within the cutoff.
        """
        return joint_avg_num_neighbors([self])


def joint_avg_num_neighbors(statistics: Sequence[DatasetStatistics]) -> float:
    """The mean neighbour count over the union of several heads' sets.

    One model has one neighbour normalisation, measured over every head's
    training structures together. Summing the counts reproduces that value
    exactly, where averaging the heads' averages would weight them wrongly.

    Raises:
        ValueError: if ``statistics`` is empty, the statistics were taken at
            different cutoffs, or no atom has a neighbour within the cutoff.
    """
    if not statistics:
        raise ValueError("there are no statistics to combine.")
    cutoffs = sorted({s.cutoff for s in statistics})
    if len(cutoffs) != 1:
        raise ValueError(
            f"neighbour counts taken at cutoffs {cutoffs} A cannot be combined; "
            f"measure every head at the model's cutoff."
        )
    receivers = sum(s.n_receivers for s in statistics)
    if receivers == 0:
        raise ValueError(
            f"no atom has a neighbour within the cutoff of {cutoffs[0]} A, so "
            f"there is no average neighbour count."
        )
    return sum(s.n_edges for s in statistics) / receivers


def _one_head(
    backend: DataBackend, indices: Sequence[int] | None
) -> Iterator[tuple[int, Configuration]]:
    """``(index, structure)`` at ``indices``, every one when ``None``.

    Raises:
        DataBackendError: at the first structure of a second head.
    """
    if indices is None:
        pairs = enumerate(backend.iter_range())
    else:
        pairs = ((index, backend[index]) for index in indices)
    head = None
    for index, structure in pairs:
        if head is None:
            head = structure.head
        elif structure.head != head:
            raise DataBackendError(
                backend.source,
                index,
                f"its head is {structure.head!r} where the earlier structures' "
                f"is {head!r}. Statistics are measured per head: pass the "
                f"indices of one head's structures.",
            )
        yield index, structure


def _check_energy_structure(
    backend: DataBackend, index: int, structure: Configuration
) -> None:
    if not len(structure):
        raise DataBackendError(
            backend.source, index, "it has an energy and no atoms to divide it among."
        )


def _label(
    backend: DataBackend, index: int, structure: Configuration, name: str
) -> np.ndarray | None:
    """The label as float64, ``None`` when the structure is not labelled with it.

    Raises:
        DataBackendError: if the label is not finite. A NaN or infinite label
            would turn every statistic computed from it into NaN.
    """
    # The quantity names are fixed here, not declared by the caller, so a key
    # specification that does not declare one simply has none of it.
    if name not in structure.properties or not structure.is_labelled(name):
        return None
    value = np.asarray(structure.properties[name], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise DataBackendError(backend.source, index, f"its {name} is not finite.")
    return value


def compute_statistics(
    backend: DataBackend,
    indices: Sequence[int] | None = None,
    *,
    cutoff: float,
    atomic_energies: Mapping[int, float],
    neighbor_backend: NeighborListBackend,
) -> DatasetStatistics:
    """Measure one head's structures in one pass.

    Args:
        backend: Where the structures are read from.
        indices: Which structures, usually the head's training split; every
            structure when ``None``. Only these are read. An index given
            twice counts twice, as a repeated structure in a split does.
        cutoff: The model's neighbour cutoff, in Angstrom.
        atomic_energies: ``{atomic number: eV}``, the head's resolved
            isolated-atom energies. Required for every element of a structure
            labelled with an energy.
        neighbor_backend: The neighbour-list backend the run resolved. Every
            backend yields the same edge set, so it changes the time and not
            the counts.

    Raises:
        DataBackendError: naming the structure, if the structures carry more
            than one head, a label is not finite, or an energy-labelled
            structure is empty or holds an element ``atomic_energies`` does
            not cover.
        ValueError: if an isolated-atom energy is not finite.
    """
    bad = sorted(z for z, e in atomic_energies.items() if not np.isfinite(e))
    if bad:
        raise ValueError(
            f"the isolated-atom energies of elements {bad} are not finite."
        )
    interaction_energies: list[float] = []
    force_squares = dipole_squares = 0.0
    force_components = dipole_components = 0
    n_edges = n_receivers = 0
    for index, structure in _one_head(backend, indices):
        receivers = get_neighborhood(
            structure.positions,
            cutoff,
            pbc=structure.pbc,
            cell=structure.cell,
            backend=neighbor_backend,
        ).edge_index[1]
        n_edges += receivers.size
        n_receivers += np.unique(receivers).size
        energy = _label(backend, index, structure, "energy")
        if energy is not None:
            _check_energy_structure(backend, index, structure)
            missing = sorted(
                set(map(int, structure.atomic_numbers)) - set(atomic_energies)
            )
            if missing:
                raise DataBackendError(
                    backend.source,
                    index,
                    f"no isolated-atom energy is given for its elements {missing}, "
                    f"so its interaction energy is undefined.",
                )
            reference = sum(atomic_energies[int(z)] for z in structure.atomic_numbers)
            interaction_energies.append((float(energy) - reference) / len(structure))
        forces = _label(backend, index, structure, "forces")
        if forces is not None:
            force_squares += float(np.sum(forces**2))
            force_components += forces.size
        dipole = _label(backend, index, structure, "dipole")
        if dipole is not None:
            dipole_squares += float(np.sum(dipole**2))
            dipole_components += dipole.size
    return DatasetStatistics(
        cutoff=cutoff,
        atomic_energies={int(z): float(e) for z, e in atomic_energies.items()},
        n_edges=n_edges,
        n_receivers=n_receivers,
        energy_mean=(
            float(np.mean(interaction_energies)) if interaction_energies else None
        ),
        energy_std=(
            float(np.std(interaction_energies, ddof=1))
            if len(interaction_energies) > 1
            else None
        ),
        rms_forces=(
            float(np.sqrt(force_squares / force_components))
            if force_components
            else None
        ),
        rms_dipoles=(
            float(np.sqrt(dipole_squares / dipole_components))
            if dipole_components
            else None
        ),
    )


def fit_atomic_energies(
    backend: DataBackend,
    indices: Sequence[int] | None = None,
    *,
    atomic_numbers: AtomicNumberTable,
) -> dict[int, float]:
    """Isolated-atom energies fitted to one head's total energies.

    The least-squares solution of ``E = sum_Z count_Z * E0_Z`` over the
    structures labelled with an energy, one unknown per element of
    ``atomic_numbers``. When the compositions do not determine every energy,
    as in a set of one stoichiometry, the solution is the minimum-norm one, as
    the frozen tree's: the interaction energies of those compositions are
    right, and each element's value on its own is not. That is logged.

    Args:
        backend: Where the structures are read from.
        indices: Which structures; every structure when ``None``.
        atomic_numbers: The elements to fit, in the order of the columns.

    Returns:
        ``{atomic number: eV}`` for every element of ``atomic_numbers``.

    Raises:
        DataBackendError: naming the structure, if the structures carry more
            than one head, a structure holds an element outside
            ``atomic_numbers``, or its energy is not finite or has no atoms.
        ValueError: if an element of ``atomic_numbers`` is in no
            energy-labelled structure. Give its energy explicitly.
    """
    columns = {z: column for column, z in enumerate(atomic_numbers.zs)}
    counts: list[np.ndarray] = []
    energies: list[float] = []
    for index, structure in _one_head(backend, indices):
        energy = _label(backend, index, structure, "energy")
        if energy is None:
            continue
        _check_energy_structure(backend, index, structure)
        outside = sorted(set(map(int, structure.atomic_numbers)) - set(columns))
        if outside:
            raise DataBackendError(
                backend.source,
                index,
                f"it holds elements {outside} outside the table "
                f"{list(atomic_numbers.zs)} the energies are fitted over.",
            )
        row = np.zeros(len(columns))
        for z in structure.atomic_numbers:
            row[columns[int(z)]] += 1.0
        counts.append(row)
        energies.append(float(energy))
    matrix = np.array(counts).reshape(len(counts), len(columns))
    unseen = [z for z, column in columns.items() if not matrix[:, column].any()]
    if unseen:
        raise ValueError(
            f"no energy-labelled structure in {backend.source!r} holds elements "
            f"{unseen}, so their isolated-atom energies cannot be fitted. Give "
            f"them explicitly."
        )
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.array(energies), rcond=None)
    if rank < len(columns):
        logger.warning(
            "The compositions in %r determine only %d of the %d isolated-atom "
            "energies of %s; the fit is the minimum-norm one. Give the energies "
            "explicitly if the model will see other compositions.",
            backend.source,
            rank,
            len(columns),
            list(columns),
        )
    return {int(z): float(solution[columns[z]]) for z in columns}
