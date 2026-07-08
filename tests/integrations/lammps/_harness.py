"""Test harness for the LAMMPS integration contract.

Reproduces, in pure Python, the data layout LAMMPS's pair style feeds to the
exported model: an open (non-periodic) cluster of LOCAL atoms plus GHOST
periodic images within the receptive field, and a local_or_ghost mask. This is
what lets us exercise the lammps_class/lammps_natoms branches of the model and
the LAMMPS_MACE wrapper without a LAMMPS binary.
"""

from pathlib import Path

import numpy as np
import torch
from ase.atoms import Atoms

from mace import data
from mace.tools import AtomicNumberTable, torch_geometric

REPO_ROOT = Path(__file__).resolve().parents[3]


def water_unit_cell() -> Atoms:
    """The periodic water box the tiny session model was trained on."""
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )


def model_batch(model: torch.nn.Module, atoms: Atoms) -> dict:
    """Standard AtomicData batch for `atoms` matching the model's metadata."""
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = data.config_from_atoms(atoms)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return next(iter(loader)).to_dict()


def lammps_style_cluster(model: torch.nn.Module, n_repeat: int):
    """LAMMPS-style inputs for the periodic water cell.

    Returns (batch, local_or_ghost, image_of): an open cluster of
    n_repeat^3 replicas with the central replica as the LOCAL atoms, a
    float mask (1.0 local / 0.0 ghost), and for every cluster atom the index
    of the unit-cell atom it is an image of (for force folding).
    """
    unit = water_unit_cell()
    n_atoms = len(unit)
    supercell = unit.repeat((n_repeat, n_repeat, n_repeat))
    # ase.Atoms.repeat orders cell blocks outer, atoms inner:
    image_of = np.arange(len(supercell)) % n_atoms
    center_block = (
        (n_repeat // 2) * n_repeat * n_repeat
        + (n_repeat // 2) * n_repeat
        + (n_repeat // 2)
    )
    local_idx = np.arange(center_block * n_atoms, (center_block + 1) * n_atoms)

    cluster = Atoms(
        numbers=supercell.numbers, positions=supercell.positions, pbc=False
    )
    batch = model_batch(model, cluster)
    local_or_ghost = torch.zeros(len(cluster), dtype=batch["positions"].dtype)
    local_or_ghost[local_idx] = 1.0
    return batch, local_or_ghost, image_of, local_idx


def fold_ghost_forces(forces: torch.Tensor, image_of: np.ndarray, n_atoms: int):
    """LAMMPS 'reverse communication': accumulate ghost-image forces onto the
    owning local atom."""
    folded = torch.zeros((n_atoms, 3), dtype=forces.dtype)
    folded.index_add_(0, torch.tensor(image_of, dtype=torch.long), forces)
    return folded
