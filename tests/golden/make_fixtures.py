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

Species in that first group are drawn from {H, C, O} only, so one
three-element anchor model covers all six.

A second group serves the magnetic family, and it is a different periodic
table: the magnetic models take a per-node magnetic moment as an *input* and
return ``dE/dm``, so their fixtures have to carry moments and to be built on
elements that have any (Fe, with O as the ligand). They are listed in
``MAGNETIC_ENTRIES`` below with their own physics rationale. One manifest
holds both groups and every consumer selects by chemistry --
``load_fixtures(elements=...)`` -- because handing an H/C/O anchor an iron
structure is not a tolerance failure, it is a missing z-table entry.
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


# ---------------------------------------------------------------------------
# The magnetic group
#
# These five exist because ``dE/dm`` is pinned nowhere else before MAG-1, and
# a derivative with respect to an input is only as meaningful as the input it
# is taken at. Every choice below is a physics decision rather than a
# convenience, so each is stated:
#
# * **the moment is an input, and it is written where the model reads it.**
#   ``REF_magmom`` (mace/tools/default_keys.py:18), an (n_atoms, 3) array --
#   not ase's initial magnetic moments, which no forward on this tree looks
#   at. The harness refuses a structure that carries the moments only in the
#   ase attribute, so this is enforced rather than remembered.
#
# * **the ligand's moment is small and not zero, and the reason is physics
#   rather than arithmetic.** The obvious worry is that
#   ``MagneticScaleShiftMACE.forward`` takes ``torch.norm(magmom)``
#   (mace/modules/extensions.py:1813) and that the gradient of a norm at the
#   origin is 0/0. Measured, it is not a problem: the norm enters only
#   *squared* (``1 - 2 * clamp(|m| / m_max)**2``), which is smooth at the
#   origin, and torch's convention of a zero gradient for the norm there is
#   exactly the right value for it. On the committed anchor a site at m = 0
#   gets a finite ``dE/dm`` agreeing with a central difference to eleven
#   digits, and a named test keeps that measurement rather than this sentence.
#   The bridging oxygens carry 0.3 muB because superexchange leaves them one.
#
# * **no moment saturates.** The radial magnetic basis reads
#   ``1 - 2 * clamp(|m| / m_max, 0, 1)**2`` (:1818-1824), so a moment at or
#   above the element's ``m_max`` sits on the flat side of a clamp and
#   contributes exactly zero to ``dE/dm`` through that path -- a structurally
#   zero derivative wearing the same shape as a computed one. The five
#   fixtures span |m| / m_max = 0.25, 0.58, 0.67, 0.78, 0.89 against the
#   anchor's ``m_max`` of 1.2 (O) and 4.5 (Fe), so the Chebyshev basis is
#   sampled across its argument range instead of at one point.
#
# * **two elements, not one.** ``m_max`` is indexed by the one-hot species
#   (:1815-1817), so a single-element fixture set cannot tell a correct
#   lookup from a transposed one. The Fe/O cluster can, and a named test uses
#   it for exactly that.
#
# The moments are collinear along z where the physics is collinear and
# genuinely three-dimensional in the frustrated trimer, because the spherical
# harmonics of the moment (``max_m_ell``) are identically trivial in the
# collinear case and a set with no non-collinear member would pin the l>0
# machinery not at all.
# ---------------------------------------------------------------------------

#: Fe-Fe in the free dimer; the experimental bond length is about 2.02 Ang.
_FE_DIMER_D = 2.02

#: Per-atom moments, in Bohr magnetons. The free atom's 4 muB is Hund's rule
#: on 3d^6 4s^2 (S = 2); the dimer sits between the atom and bcc iron's 2.2;
#: the oxo-bridged cluster's 3.5 is the high-spin Fe(III) value reduced by
#: covalency, and 0.3 on the bridging oxygen is the induced moment
#: superexchange leaves there.
_M_FE_ATOM = 4.0
_M_FE_DIMER = 3.0
_M_FE_TRIMER = 2.6
_M_FE_OXO = 3.5
_M_O_INDUCED = 0.3


def _mag_rng() -> np.random.Generator:
    """A stream of its own, so adding this group perturbs no existing fixture."""
    return np.random.default_rng(SEED + 2)


def _with_moments(atoms: Atoms, moments: np.ndarray) -> Atoms:
    """Attach the moments under the key the magnetic models read."""
    atoms.new_array("REF_magmom", np.asarray(moments, dtype=float))
    return atoms


def mag_fe_atom() -> Atoms:
    """One iron atom carrying its free-atom moment: zero edges.

    Nothing survives here but the species embedding, the E0 term and the
    one-body magnetic term, which is what makes it the fixture that pins
    ``--use_magmom_one_body`` on its own.
    """
    atoms = Atoms(numbers=[26], positions=[[0.0, 0.0, 0.0]], pbc=[False] * 3)
    return _with_moments(atoms, [[0.0, 0.0, _M_FE_ATOM]])


def mag_fe_dimer_fm() -> Atoms:
    """Fe2 with both moments parallel along z: the ferromagnetic state."""
    atoms = Atoms(
        numbers=[26, 26],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, _FE_DIMER_D]],
        pbc=[False] * 3,
    )
    return _with_moments(
        atoms, [[0.0, 0.0, _M_FE_DIMER], [0.0, 0.0, _M_FE_DIMER]]
    )


def mag_fe_dimer_afm() -> Atoms:
    """The same two atoms in the same places, one moment flipped.

    The pair is the point. Geometry, species and |m| are identical to
    ``mag_fe_dimer_fm`` down to the last digit of the committed file, so every
    difference between the two references is the spin state and nothing else.
    That difference -- the FM/AFM splitting -- is the observable a magnetic
    interatomic potential exists to reproduce, and a rewrite that dropped the
    moment from the message passing would make the two fixtures produce one
    number instead of two while every non-magnetic golden still passed.
    """
    atoms = Atoms(
        numbers=[26, 26],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, _FE_DIMER_D]],
        pbc=[False] * 3,
    )
    return _with_moments(
        atoms, [[0.0, 0.0, _M_FE_DIMER], [0.0, 0.0, -_M_FE_DIMER]]
    )


def mag_fe3_canted() -> Atoms:
    """An equilateral Fe3 triangle in the 120-degree Neel state, canted.

    The frustrated triangular antiferromagnet: three moments at 120 degrees to
    each other in the plane of the triangle, which is the classical ground
    state when every pair wants to be antiparallel and no arrangement can
    satisfy all three. A seeded out-of-plane canting is added and the moments
    are renormalised, so no component of any moment is zero and no two are
    parallel -- a symmetric arrangement would make components of ``dE/dm``
    vanish by symmetry and pin nothing.
    """
    side = 2.30
    radius = side / np.sqrt(3.0)
    angles = np.array([0.0, 2.0, 4.0]) * np.pi / 3.0
    positions = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(3)], axis=-1
    )
    moments = np.stack(
        [np.cos(angles + np.pi / 2), np.sin(angles + np.pi / 2), np.zeros(3)],
        axis=-1,
    )
    moments = moments + _mag_rng().normal(scale=0.15, size=moments.shape)
    moments *= _M_FE_TRIMER / np.linalg.norm(moments, axis=-1, keepdims=True)
    return _with_moments(
        Atoms(numbers=[26, 26, 26], positions=positions, pbc=[False] * 3), moments
    )


def mag_feo_cluster() -> Atoms:
    """A planar Fe2O2 diamond core: two elements, two very different moments.

    The oxo-bridged Fe2O2 rhombus is the recurring motif of di-iron sites, and
    its irons are antiferromagnetically coupled through the bridges rather
    than directly, so the fixture carries the physics of superexchange rather
    than of a bond. It is also the only fixture with more than one element,
    which makes it the only one that can distinguish the per-element ``m_max``
    lookup from a transposed one; the two elements' ``m_max`` differ by nearly
    a factor of four, so a transposition is not a rounding difference.
    """
    positions = np.array(
        [[-1.30, 0.00, 0.00], [1.30, 0.00, 0.00], [0.00, -1.32, 0.00], [0.00, 1.32, 0.00]]
    )
    rng = _mag_rng()
    o_moments = rng.normal(size=(2, 3))
    o_moments *= _M_O_INDUCED / np.linalg.norm(o_moments, axis=-1, keepdims=True)
    moments = np.concatenate(
        [
            np.array([[0.0, 0.0, _M_FE_OXO], [0.0, 0.0, -_M_FE_OXO]]),
            o_moments,
        ],
        axis=0,
    )
    return _with_moments(
        Atoms(numbers=[26, 26, 8, 8], positions=positions, pbc=[False] * 3), moments
    )


BUILDERS = {
    "triclinic_bulk": triclinic_bulk,
    "water_cluster": water_cluster,
    "isolated_atom": isolated_atom,
    "dimer_short": dimer_short,
    "slab_vacuum": slab_vacuum,
    "slab_zero_vacuum": slab_zero_vacuum,
    "mag_fe_atom": mag_fe_atom,
    "mag_fe_dimer_fm": mag_fe_dimer_fm,
    "mag_fe_dimer_afm": mag_fe_dimer_afm,
    "mag_fe3_canted": mag_fe3_canted,
    "mag_feo_cluster": mag_feo_cluster,
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

#: The magnetic group. Every entry carries `magmom`, so a golden selects the
#: whole group with that tag and the chemistry filter never has to name them.
MAGNETIC_ENTRIES = {
    "mag_fe_atom": {
        "description": "One Fe atom carrying its free-atom moment, 4.0 muB along z.",
        "regime": "zero edges; the one-body magnetic term alone",
        "tags": ["aperiodic", "molecular", "single_atom", "magmom", "collinear"],
    },
    "mag_fe_dimer_fm": {
        "description": "Fe2 at 2.02 Ang, both moments +3.0 muB along z.",
        "regime": "ferromagnetic dimer; the partner of mag_fe_dimer_afm",
        "tags": ["aperiodic", "molecular", "magmom", "collinear"],
    },
    "mag_fe_dimer_afm": {
        "description": "The same Fe2 geometry with the second moment reversed.",
        "regime": "antiferromagnetic dimer; identical geometry to the FM one, "
        "so the whole difference between the two is the spin state",
        "tags": ["aperiodic", "molecular", "magmom", "collinear"],
    },
    "mag_fe3_canted": {
        "description": "Equilateral Fe3 triangle in a canted 120-degree Neel state.",
        "regime": "non-collinear moments; the only fixture exercising the "
        "l > 0 spherical harmonics of the moment",
        "tags": ["aperiodic", "molecular", "magmom", "noncollinear", "frustrated"],
    },
    "mag_feo_cluster": {
        "description": "Planar Fe2O2 oxo-bridged core, Fe +-3.5 muB, O 0.3 muB.",
        "regime": "two elements with different m_max; the per-element lookup",
        "tags": ["aperiodic", "molecular", "magmom", "multispecies", "noncollinear"],
    },
}

MANIFEST_ENTRIES.update(MAGNETIC_ENTRIES)


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
        # Recorded for the reader; the `elements=` filter derives it from the
        # file instead, so the two cannot drift into disagreeing silently --
        # a guard test compares them.
        entry["species"] = sorted({int(z) for z in atoms.get_atomic_numbers()})
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
