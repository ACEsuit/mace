"""Generate the committed golden fixtures.

Run through ``tests/golden/regenerate.py``; this module is importable so the
generation is one seeded function per structure rather than a script whose
output nobody can reproduce.

The six evaluation structures are not a convenience sample. Each one exists
to reach a distinct regime of the neighbour-list layer
(``mace/data/neighborhood.py``), because that layer decides which cell a
downstream stress is divided by and its three returned-cell branches are the
easiest thing in the stack to "simplify" back into a bug:

============================  ================================================
fixture                       regime it is the only one to reach
============================  ================================================
``triclinic_bulk``            fully periodic, non-orthogonal cell -> physical
                              cell returned, stress meaningful
``water_cluster``             fully aperiodic -> the *extended* cell is
                              returned (the search box), not the input cell
``isolated_atom``             zero edges: the empty-graph path
``dimer_short``               a 0.62 Ang separation, deep inside the
                              short-range repulsion envelope
``slab_vacuum``               mixed pbc with real vacuum -> physical cell
``slab_zero_vacuum``          mixed pbc whose vacuum row is all zeros -> the
                              patched-row branch, the one that would divide
                              the stress by a zero volume if the row were not
                              replaced
============================  ================================================

Species are drawn from {H, C, O} only, so one three-element anchor model
covers the whole set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase.atoms import Atoms
from ase.io import write as ase_write

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: One seed for every generated structure. Changing it changes every golden.
SEED = 20260810

#: Anchor cutoff. Small enough to keep the graphs tiny, large enough that the
#: 4 Ang cells below produce periodic images rather than isolated fragments.
R_MAX = 3.5


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def triclinic_bulk() -> Atoms:
    """Fully periodic, non-orthogonal, three species."""
    cell = np.array([[4.30, 0.00, 0.00], [1.05, 4.10, 0.00], [0.65, 0.85, 4.45]])
    scaled = np.array(
        [
            [0.05, 0.10, 0.08],
            [0.55, 0.12, 0.51],
            [0.10, 0.58, 0.53],
            [0.60, 0.62, 0.05],
            [0.32, 0.35, 0.30],
            [0.82, 0.86, 0.78],
        ]
    )
    atoms = Atoms(
        numbers=[8, 6, 1, 8, 6, 1],
        scaled_positions=scaled,
        cell=cell,
        pbc=[True, True, True],
    )
    rng = _rng()
    atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
    return atoms


def water_cluster() -> Atoms:
    """Three waters, fully aperiodic: the extended-cell regime."""
    monomer = np.array([[0.000, 0.000, 0.119], [0.000, 0.763, -0.477], [0.000, -0.763, -0.477]])
    offsets = np.array([[0.0, 0.0, 0.0], [2.85, 0.35, 0.20], [1.30, 2.60, -0.45]])
    positions: List[np.ndarray] = []
    numbers: List[int] = []
    rng = _rng()
    for offset in offsets:
        positions.append(monomer + offset + rng.normal(scale=0.02, size=monomer.shape))
        numbers += [8, 1, 1]
    return Atoms(
        numbers=numbers,
        positions=np.concatenate(positions, axis=0),
        pbc=[False, False, False],
    )


def isolated_atom() -> Atoms:
    """One atom, no neighbours: the zero-edge path."""
    return Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]], pbc=[False, False, False])


def dimer_short() -> Atoms:
    """A 0.62 Ang C-O dimer.

    Well inside the sum of covalent radii (0.76 + 0.66 = 1.42 Ang), so the
    short-range repulsion term is large and its polynomial envelope is far
    from both of its ends -- a separation of ~0.6 Ang between two *hydrogens*
    would instead sit at 0.97 of that pair's envelope cutoff and contribute
    almost nothing, which is the opposite of what this fixture is for.
    """
    return Atoms(
        numbers=[6, 8],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.62]],
        pbc=[False, False, False],
    )


def _slab_atoms() -> tuple:
    rng = _rng()
    positions = np.array(
        [
            [0.30, 0.40, 0.20],
            [2.30, 0.55, 0.35],
            [0.55, 2.35, 1.55],
            [2.45, 2.20, 1.70],
        ]
    )
    positions = positions + rng.normal(scale=0.02, size=positions.shape)
    return [6, 8, 6, 1], positions


def slab_vacuum() -> Atoms:
    """Mixed pbc with real vacuum: the physical-cell regime."""
    numbers, positions = _slab_atoms()
    return Atoms(
        numbers=numbers,
        positions=positions,
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
        pbc=[True, True, False],
    )


def slab_zero_vacuum() -> Atoms:
    """Mixed pbc whose non-periodic row is all zeros: the patched-row regime.

    Same atoms as ``slab_vacuum``, so the two differ only in the cell row.
    Without the patch this cell has determinant zero and the stress divides
    by a zero volume.
    """
    numbers, positions = _slab_atoms()
    return Atoms(
        numbers=numbers,
        positions=positions,
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
        pbc=[True, True, False],
    )


BUILDERS = {
    "triclinic_bulk": triclinic_bulk,
    "water_cluster": water_cluster,
    "isolated_atom": isolated_atom,
    "dimer_short": dimer_short,
    "slab_vacuum": slab_vacuum,
    "slab_zero_vacuum": slab_zero_vacuum,
}

MANIFEST_ENTRIES = {
    "triclinic_bulk": {
        "description": "6-atom H/C/O cell with a non-orthogonal lattice, fully periodic.",
        "regime": "fully periodic, physical cell returned",
        "tags": ["periodic", "stress", "multispecies"],
    },
    "water_cluster": {
        "description": "Three rattled water molecules, no periodic boundary.",
        "regime": "fully aperiodic, extended search cell returned",
        "tags": ["aperiodic", "molecular"],
    },
    "isolated_atom": {
        "description": "A single oxygen atom in vacuum.",
        "regime": "zero edges",
        "tags": ["aperiodic", "molecular", "single_atom"],
    },
    "dimer_short": {
        "description": "C-O dimer at 0.62 Ang, deep in the repulsive wall.",
        "regime": "short-range repulsion envelope",
        "tags": ["aperiodic", "molecular", "repulsion"],
    },
    "slab_vacuum": {
        "description": "Two-layer C/O/H slab, pbc (T, T, F), 12 Ang of real vacuum.",
        "regime": "mixed pbc, physical cell returned",
        "tags": ["periodic", "stress", "slab", "multispecies"],
    },
    "slab_zero_vacuum": {
        "description": "The same slab with an all-zero third cell row.",
        "regime": "mixed pbc, vacuum row patched from the search cell",
        "tags": ["periodic", "stress", "slab", "multispecies", "degenerate_cell"],
    },
}


def make_training_set() -> List[Atoms]:
    """The seeded synthetic set the trainable anchor is fitted on.

    Three isolated atoms carry the reference energies for H, C and O; the
    remaining configurations are rattled copies of a small periodic cell with
    synthetic labels. Nothing here is meant to be physical -- the anchor
    exists to pin an evaluation, not to model anything -- but it does have to
    be *stable*, which is why every number comes from one seeded generator
    consumed in a fixed order.
    """
    rng = np.random.default_rng(SEED + 1)
    configs: List[Atoms] = []
    for z in (1, 6, 8):
        atom = Atoms(numbers=[z], positions=[[0.0, 0.0, 0.0]], cell=[6.0] * 3)
        atom.info["REF_energy"] = float(rng.normal(scale=0.5))
        atom.info["config_type"] = "IsolatedAtom"
        atom.new_array("REF_forces", np.zeros((1, 3)))
        configs.append(atom)

    base = Atoms(
        numbers=[8, 6, 1, 1],
        positions=[[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.5, 0.9, 0.0], [1.9, 0.9, 0.6]],
        cell=[5.0, 5.0, 5.0],
        pbc=[True, True, True],
    )
    for _ in range(24):
        config = base.copy()
        config.positions += rng.normal(scale=0.12, size=config.positions.shape)
        config.info["REF_energy"] = float(rng.normal(scale=1.0))
        config.new_array("REF_forces", rng.normal(scale=0.5, size=(len(config), 3)))
        config.info["REF_stress"] = rng.normal(scale=0.05, size=6)
        configs.append(config)
    return configs


def write_fixtures(directory: Path = FIXTURES_DIR) -> Dict[str, Path]:
    """Write every fixture plus the manifest and the training set."""
    directory.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    manifest = {"seed": SEED, "r_max_hint": R_MAX, "fixtures": {}}
    for name, builder in BUILDERS.items():
        atoms = builder()
        path = directory / f"{name}.xyz"
        ase_write(path, atoms, format="extxyz")
        entry = dict(MANIFEST_ENTRIES[name])
        entry["file"] = path.name
        entry["n_atoms"] = len(atoms)
        entry["formula"] = atoms.get_chemical_formula()
        entry["pbc"] = [bool(p) for p in atoms.pbc]
        manifest["fixtures"][name] = entry
        written[name] = path

    train_path = directory / "tiny_train.xyz"
    ase_write(train_path, make_training_set(), format="extxyz")
    written["tiny_train"] = train_path

    manifest_path = directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    written["manifest"] = manifest_path
    return written
