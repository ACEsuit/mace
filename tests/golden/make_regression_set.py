"""Recipe for the committed end-to-end training regression set.

``regression_train.xyz`` is the dataset the legacy-versus-rewrite training
comparison is run on: small enough to train in seconds on one CPU core, and
deliberately *diverse* -- isolated atoms, molecules, slabs and bulk cells, all
three anchor elements, and both electrostatic labels (per-atom partial charges
and the graph dipole) so that a long-range model has something to be wrong
about. It is committed rather than generated at test time, because a
regression whose reference data is regenerated per run is not a regression.

Why the labels come from a closed-form potential rather than from noise
=======================================================================

The other fixtures in this directory carry random labels, which is right for
what they do: a golden compares two evaluations of one checkpoint, and the
labels never enter. A *training* regression is the opposite -- it asserts that
optimisation makes the error go down, and random labels make that assertion
false rather than sensitive. Measured on the 20 rattled waters the shared
``fitting_configs`` fixture builds, the validation loss of an L-BFGS run rises
monotonically (23.8 -> 47.0 -> 112.6 -> 130.1 over four epochs) because a
full-epoch quasi-Newton step fits the training noise exactly and the held-out
noise is uncorrelated with it. That is not a broken optimiser and no tolerance
can make it read as one.

So every label here is produced by one analytic function of the geometry:

    E = sum_Z E0[Z] + sum_{pairs r<rc} eps_ij * [ (s_ij/r)^4 - 2 (s_ij/r)^2 ]
                                       * (1 - (r/rc)^2)^3

with the forces, the stress and the coordination-dependent charges all
differentiated from that same expression in closed form. The function is
smooth, short-ranged and genuinely learnable by a two-layer MACE, so "the
validation loss decreased" is a statement about the optimiser and not about
which configurations happened to land in the validation split.

The analytic gradients are not trusted: :func:`self_check` differentiates the
energy numerically and compares, and it runs both here and as a test
(``tests/golden/test_regression_set.py``), so a committed dataset whose forces
do not match its energies fails in CI rather than teaching a rewrite the wrong
physics.

Regenerate with::

    python tests/golden/make_regression_set.py --i-know-what-i-am-doing
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import write as ase_write
from ase.neighborlist import neighbor_list

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "fixtures" / "regression_train.xyz"

#: The anchor elements, and nothing else: the committed tiny anchors carry the
#: z-table [1, 6, 8], so a set with a fourth element could not be fine-tuned
#: from them without rebuilding the anchors.
ELEMENTS = (1, 6, 8)

#: Range of the pair term. Chosen equal to the anchors' r_max so that every
#: labelled interaction is inside the model's receptive field -- a target the
#: model provably cannot see is a floor on the error that looks like
#: underfitting.
R_CUT = 3.5

SEED = 20260812

# Per-element isolated-atom energies, pair well depths and radii. The numbers
# are arbitrary but fixed; what matters is that the wells differ per element
# pair, so a model cannot fit the set with a single species-agnostic curve.
E0: Dict[int, float] = {1: -0.7, 6: -3.1, 8: -2.4}
EPS: Dict[int, float] = {1: 0.30, 6: 0.85, 8: 0.60}
SIGMA: Dict[int, float] = {1: 1.05, 6: 1.55, 8: 1.35}

#: Base partial charge per element, and how strongly the charge responds to
#: coordination. Both electrostatic labels come from these: the per-atom
#: charges directly, and the dipole as their first moment.
Q0: Dict[int, float] = {1: 0.34, 6: 0.12, 8: -0.62}
Q_COORDINATION = 0.08


# ---------------------------------------------------------------------------
# The closed-form potential
# ---------------------------------------------------------------------------


def _pair_parameters(z_i: np.ndarray, z_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lorentz-Berthelot mixing of the per-element constants."""
    eps = np.sqrt(
        np.array([EPS[int(z)] for z in z_i]) * np.array([EPS[int(z)] for z in z_j])
    )
    sigma = 0.5 * (
        np.array([SIGMA[int(z)] for z in z_i]) + np.array([SIGMA[int(z)] for z in z_j])
    )
    return eps, sigma


def _taper(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(1 - (r/rc)^2)^3`` and its derivative; both vanish at ``rc``."""
    x = 1.0 - (r / R_CUT) ** 2
    return x**3, 3.0 * x**2 * (-2.0 * r / R_CUT**2)


def _pair_energy(r: np.ndarray, eps: np.ndarray, sigma: np.ndarray):
    """The pair term and its radial derivative."""
    s2 = (sigma / r) ** 2
    u = eps * (s2**2 - 2.0 * s2)
    du = eps * (-4.0 * sigma**4 / r**5 + 4.0 * sigma**2 / r**3)
    fc, dfc = _taper(r)
    return u * fc, du * fc + u * dfc


def _pairs(atoms: Atoms):
    """Every directed neighbour pair inside ``R_CUT``, with its vector.

    ``ase.neighborlist`` rather than the package's own neighbour list on
    purpose: this recipe must not depend on the code the dataset is used to
    test, or a neighbour-list bug would relabel the data to agree with itself.
    """
    i, j, d = neighbor_list("ijD", atoms, R_CUT)
    return np.asarray(i), np.asarray(j), np.asarray(d, dtype=np.float64)


def label(atoms: Atoms) -> Atoms:
    """Attach energy, forces, stress, charges and dipole to a copy of ``atoms``.

    Every quantity is a derivative of the one energy expression, in the sign
    conventions the package uses (``mace/modules/utils.py``): forces are
    ``-dE/dx``, and the stress is ``(1/V) dE/d(strain)``, so a positive radial
    derivative (attraction) gives a negative diagonal stress.
    """
    out = atoms.copy()
    numbers = np.asarray(out.numbers)
    n_atoms = len(out)

    energy = float(sum(E0[int(z)] for z in numbers))
    forces = np.zeros((n_atoms, 3))
    virial = np.zeros((3, 3))
    coordination = np.zeros(n_atoms)

    if n_atoms > 1 or np.asarray(out.pbc).any():
        i, j, vec = _pairs(out)
        if len(i):
            r = np.linalg.norm(vec, axis=1)
            eps, sigma = _pair_parameters(numbers[i], numbers[j])
            v, dv = _pair_energy(r, eps, sigma)
            # The neighbour list is directed, so every physical pair appears
            # twice and each contribution is halved.
            energy += 0.5 * float(v.sum())
            # vec is r_j - r_i, so dE/dx_i = -dv * vec / r.
            grad = (dv / r)[:, None] * vec
            np.add.at(forces, i, grad)
            virial += 0.5 * np.einsum("p,pa,pb->ab", dv / r, vec, vec)
            fc, _ = _taper(r)
            np.add.at(coordination, i, fc)

    charges = np.array(
        [Q0[int(z)] for z in numbers]
    ) + Q_COORDINATION * coordination
    # Neutralise: a per-configuration net charge the model is never told about
    # would be an unlearnable offset in the dipole.
    charges -= charges.mean()

    out.info["REF_energy"] = energy
    out.arrays["REF_forces"] = forces
    out.arrays["REF_charges"] = charges
    # ``REF_dipole`` and not ``dipole``, which is the package's own default
    # key, because ``dipole`` is one of ase's recognised calculator
    # properties: a value written into ``atoms.info["dipole"]`` comes back
    # from an extxyz round trip inside ``atoms.calc.results`` with ``info``
    # empty, and the parser only reads ``info``. So a set that used the
    # default name would carry a dipole label no training run could see.
    # Consumers pass ``--dipole_key=REF_dipole``.
    out.info["REF_dipole"] = charges @ out.positions
    out.info["total_charge"] = 0.0
    volume = abs(np.linalg.det(np.asarray(out.cell)))
    if np.asarray(out.pbc).any() and volume > 0.0:
        # A degenerate cell carries no stress label at all rather than an
        # infinity: the slab built with no vacuum has det(cell) = 0, and the
        # only honest label for `(1/V) dE/d(strain)` there is none. The
        # configuration itself stays in the set -- it is a real user input and
        # a real place for the neighbour list to divide by zero -- so the
        # training path has to cope with a per-property label that is absent
        # on one configuration and present on the rest.
        stress = virial / volume
        out.info["REF_stress"] = np.array(
            [
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[1, 2],
                stress[0, 2],
                stress[0, 1],
            ]
        )
    return out


def self_check(atoms: Atoms, h: float = 1e-5) -> Tuple[float, float]:
    """Central-difference the labelled energy and return the worst errors.

    Returns ``(force_error, stress_error)`` in eV/Ang and eV/Ang^3. The stress
    error is ``0.0`` for an aperiodic configuration, which carries none.
    """
    labelled = label(atoms)
    numerical = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for k in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[a, k] += h
            minus.positions[a, k] -= h
            numerical[a, k] = -(
                label(plus).info["REF_energy"] - label(minus).info["REF_energy"]
            ) / (2 * h)
    force_error = float(np.abs(numerical - labelled.arrays["REF_forces"]).max())

    stress_error = 0.0
    if "REF_stress" in labelled.info:
        volume = abs(np.linalg.det(np.asarray(atoms.cell)))
        # One entry of the deformation gradient at a time, so there is no
        # symmetric-strain factor-of-two convention to get wrong: positions
        # and cell both transform as x -> F x, which is exactly what makes
        # every pair vector transform the same way.
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        numeric_voigt = np.zeros(6)
        for component, (a, b) in enumerate(voigt):
            deformed = []
            for sign in (+1, -1):
                gradient = np.eye(3)
                gradient[a, b] += sign * h
                probe = atoms.copy()
                probe.set_cell(np.asarray(atoms.cell) @ gradient.T, scale_atoms=False)
                probe.positions = np.asarray(atoms.positions) @ gradient.T
                deformed.append(label(probe).info["REF_energy"])
            numeric_voigt[component] = (deformed[0] - deformed[1]) / (2 * h) / volume
        stress_error = float(
            np.abs(numeric_voigt - labelled.info["REF_stress"]).max()
        )
    return force_error, stress_error


# ---------------------------------------------------------------------------
# The structures
#
# Four families, each present for a reason a rewrite could get wrong on its
# own: isolated atoms fix the E0 table, molecules are the aperiodic
# extended-cell neighbour regime, slabs are mixed per-axis pbc (where the
# stress divides by a volume the neighbour list had to patch), and bulk cells
# are the fully periodic minimum-image regime.
# ---------------------------------------------------------------------------


def _rattle(atoms: Atoms, rng: np.random.Generator, amplitude: float) -> Atoms:
    out = atoms.copy()
    out.positions += rng.normal(scale=amplitude, size=out.positions.shape)
    return out


def isolated_atoms() -> List[Atoms]:
    configs = []
    for z in ELEMENTS:
        atom = Atoms(numbers=[z], positions=[[0.0, 0.0, 0.0]], cell=[8.0] * 3)
        atom.info["config_type"] = "IsolatedAtom"
        configs.append(atom)
    return configs


def molecules(rng: np.random.Generator) -> List[Atoms]:
    """Water, formaldehyde and a methane-like CH4, rattled."""
    templates = [
        Atoms(
            numbers=[8, 1, 1],
            positions=[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
        ),
        Atoms(
            numbers=[6, 8, 1, 1],
            positions=[
                [0.0, 0.0, -0.53],
                [0.0, 0.0, 0.68],
                [0.0, 0.94, -1.11],
                [0.0, -0.94, -1.11],
            ],
        ),
        Atoms(
            numbers=[6, 1, 1, 1, 1],
            positions=[
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [0.63, -0.63, -0.63],
                [-0.63, 0.63, -0.63],
            ],
        ),
    ]
    configs = []
    for template in templates:
        for _ in range(4):
            configs.append(_rattle(template, rng, 0.08))
    # One two-molecule cluster, so the set contains an intermolecular
    # interaction rather than only intramolecular ones.
    for _ in range(2):
        dimer = templates[0].copy()
        partner = templates[0].copy()
        partner.positions += np.array([2.6, 0.3, 0.1])
        dimer += partner
        configs.append(_rattle(dimer, rng, 0.06))
    for config in configs:
        config.set_cell([12.0] * 3)
        config.set_pbc(False)
        config.info["config_type"] = "molecule"
    return configs


def slabs(rng: np.random.Generator) -> List[Atoms]:
    """Two-layer surfaces, periodic in x and y, vacuum along z."""
    configs = []
    for _ in range(5):
        base = Atoms(
            numbers=[6, 6, 8, 1],
            positions=[
                [0.0, 0.0, 0.0],
                [1.7, 0.1, 0.05],
                [0.85, 1.5, 1.35],
                [2.4, 1.6, 1.45],
            ],
            cell=[3.4, 3.2, 11.0],
            pbc=[True, True, False],
        )
        config = _rattle(base, rng, 0.07)
        config.info["config_type"] = "surface"
        configs.append(config)
    # One slab with an all-zero vacuum row: the degenerate cell the neighbour
    # list patches, which is a real user input and a real place to divide by
    # zero.
    zero_vacuum = configs[0].copy()
    cell = np.asarray(zero_vacuum.cell).copy()
    cell[2] = 0.0
    zero_vacuum.set_cell(cell)
    zero_vacuum.info["config_type"] = "surface_zero_vacuum"
    configs.append(zero_vacuum)
    return configs


def bulk(rng: np.random.Generator) -> List[Atoms]:
    """Fully periodic cells, one of them triclinic."""
    configs = []
    orthorhombic = Atoms(
        numbers=[6, 6, 8, 8, 1, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [2.1, 0.0, 0.0],
            [0.0, 2.0, 0.1],
            [2.1, 2.0, 0.1],
            [1.05, 1.0, 1.9],
            [3.15, 1.0, 1.9],
        ],
        cell=[4.2, 4.0, 3.9],
        pbc=True,
    )
    triclinic = orthorhombic.copy()
    triclinic.set_cell(
        [[4.2, 0.0, 0.0], [0.6, 4.0, 0.0], [0.4, 0.5, 3.9]], scale_atoms=True
    )
    for template, name in ((orthorhombic, "bulk"), (triclinic, "bulk_triclinic")):
        for _ in range(4):
            config = _rattle(template, rng, 0.06)
            config.info["config_type"] = name
            configs.append(config)
    return configs


def build() -> List[Atoms]:
    rng = np.random.default_rng(SEED)
    raw = list(
        itertools.chain(
            isolated_atoms(), molecules(rng), slabs(rng), bulk(rng)
        )
    )
    return [label(config) for config in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-know-what-i-am-doing",
        action="store_true",
        dest="acknowledged",
        help="required: this overwrites a committed dataset",
    )
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if not args.acknowledged:
        parser.error(
            "refusing to overwrite a committed dataset without "
            "--i-know-what-i-am-doing"
        )
    configs = build()
    worst_force, worst_stress = 0.0, 0.0
    for config in configs:
        if len(config) == 1:
            continue
        force_error, stress_error = self_check(config)
        worst_force = max(worst_force, force_error)
        worst_stress = max(worst_stress, stress_error)
    print(f"self-check: worst force {worst_force:.3e}, stress {worst_stress:.3e}")
    ase_write(args.output, configs, format="extxyz")
    print(f"wrote {len(configs)} configurations to {args.output}")


if __name__ == "__main__":
    main()
