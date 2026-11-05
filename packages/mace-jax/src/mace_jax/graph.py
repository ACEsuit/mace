"""Tensorize mace_core's natural graph batching contract as jax.Array."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.graph import collate, graph_sample

from mace_jax.neighbors import resolve_neighbor_backend


def graph_from_atoms(
    structures: Atoms | list[Atoms],
    *,
    elements: tuple[int, ...],
    cutoff: float,
    dtype: str = "float64",
    head: str = "default",
) -> dict[str, jnp.ndarray]:
    """One or more ASE structures -> unpadded flat graph, positions/cell in Å."""
    if isinstance(structures, Atoms):
        structures = [structures]
    samples = [
        graph_sample(
            Configuration(
                atomic_numbers=np.asarray(atoms.numbers),
                positions=np.asarray(atoms.positions),
                cell=np.asarray(atoms.cell),
                pbc=(bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2])),
                head=head,
            ),
            atomic_number_table=AtomicNumberTable(elements),
            cutoff=cutoff,
            heads=(head,),
            neighbor_backend=resolve_neighbor_backend(),
        )
        for atoms in structures
    ]
    arrays = collate(samples)
    return {
        name: jnp.asarray(
            value, dtype=dtype if np.issubdtype(value.dtype, np.floating) else None
        )
        for name, value in arrays.items()
        if isinstance(value, np.ndarray)
    }
