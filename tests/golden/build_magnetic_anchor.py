"""Build the tiny ``MagneticScaleShiftMACE`` anchor by direct instantiation.

The magnetic family is a full parallel model stack -- its own models, its own
interaction and product blocks, its own derivative kernels
(``compute_forces_magforces``, ``compute_forces_virials_magforces`` in
``mace/modules/utils.py``), its own calculator and its own CLI surface -- and
it arrived after the last release with no numerical reference anywhere. The
behavioural suite in ``tests/extensions/magnetic`` is thorough about
*properties* (equivariance, parity, parameter registration, dtype scoping),
and none of that is a number a rewrite has to reproduce.

What makes this anchor different from the two in ``build_mace_anchor.py`` /
``train_anchor.py`` is the input side. The magnetic models take a per-node
magnetic moment as an **input** and differentiate the energy with respect to
it, so ``magforces`` is ``dE/dm`` and a reference that did not record the
moments it was taken at would not be reproducible at all. The harness records
and compares them, exactly; the fixtures carry them under ``REF_magmom``,
which is the array the model reads.

Built rather than trained, for the same reason as the plain-MACE anchor: a
seeded untrained network exercises the energy assembly and both derivative
paths exactly as a fitted one would, and a training recipe would make the
reference depend on an optimiser trajectory. Two choices below are not free,
though, and both would silently change every number in the reference:

* ``use_reduced_cg=False``. The default is ``True`` and
  ``mace/modules/wrapper_ops.py:428`` degrades it to ``False`` when
  cuequivariance is absent, so an anchor built with the default has different
  weights depending on what happens to be installed on the machine that built
  it. Same reasoning as the plain anchor.
* ``use_magmom_one_body=True``. This is the switch behind
  ``--use_magmom_one_body``, and it adds a per-atom energy term that depends
  on |m| alone through a Chebyshev basis plus a per-species constant
  correction (``mace/modules/extensions.py:1719-1737``, applied at
  ``:1866-1888``). Off, the term is not merely zero -- the parameters do not
  exist, and nothing in the reference would cover the path. On, the
  ``mag_fe_atom`` fixture pins it essentially on its own: with no edges,
  nothing else contributes.

This anchor needs the ``magnetic`` extra (``sphericart``): the moment's
spherical harmonics are computed by ``sphericart.torch.SolidHarmonics``
(``mace/modules/extensions.py:1351-1363``), so both building and *loading*
the checkpoint require it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.modules.extensions import MagneticScaleShiftMACE

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "tiny_magnetic.model"
SIDECAR_PATH = MODELS_DIR / "tiny_magnetic.build.json"

SEED = 20260810

#: Iron and its ligand, in z-table order -- which is the order ``m_max`` is
#: indexed in, since the lookup is by one-hot species position
#: (``mace/modules/extensions.py:1815-1817``).
ATOMIC_NUMBERS = [8, 26]

#: Fixed, not fitted. This anchor is never trained, so its isolated-atom
#: energies are part of the recipe rather than a dataset statistic.
ATOMIC_ENERGIES = [-4.25, -6.75]

AVG_NUM_NEIGHBORS = 3.0

#: The per-element ceiling the moment magnitude is scaled against, in the same
#: order as ``ATOMIC_NUMBERS``. Chosen so that no fixture saturates the clamp
#: at ``1 - 2 * clamp(|m| / m_max, 0, 1)**2``: the largest ratio in the set is
#: the free Fe atom's 4.0 / 4.5 = 0.89. The two entries differ by nearly a
#: factor of four, which is what makes a transposed lookup visible.
M_MAX = [1.2, 4.5]

ANCHOR_CONFIG: Dict[str, Any] = {
    "r_max": 4.0,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "num_interactions": 2,
    "num_elements": len(ATOMIC_NUMBERS),
    "hidden_irreps": "16x0e + 16x1o",
    "MLP_irreps": "8x0e",
    "atomic_numbers": ATOMIC_NUMBERS,
    "correlation": 2,
    "radial_type": "bessel",
    "distance_transform": "None",
    "pair_repulsion": True,
    "use_reduced_cg": False,
    # The residual variant for the later layer, as the plain anchor does: the
    # first layer has no self-connection to carry.
    "interaction_cls": (
        "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock"
    ),
    "interaction_cls_first": (
        "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
    ),
    "gate": "silu",
    # --- the magnetic hyperparameters -------------------------------------
    "m_max": M_MAX,
    # The radial basis over the transformed moment magnitude.
    "num_mag_radial_basis": 6,
    # The one-body basis, a separate count with a separate parameter block.
    "num_mag_radial_basis_one_body": 6,
    # l = 0, 1, 2 on the moment's solid harmonics. At `max_m_ell=1` only the
    # direction of m enters; the l=2 terms are what a non-collinear fixture
    # exists to reach, so the anchor has to carry them.
    "max_m_ell": 2,
    "use_magmom_one_body": True,
    "atomic_inter_scale": 0.7,
    "atomic_inter_shift": 0.1,
}

#: How the SCF wrapper is asked to relax the moments. Recorded here rather
#: than in the SCF golden, so the reference and its regeneration cannot
#: disagree about what was run.
#:
#: ``n_scf_step`` is LBFGS's ``max_iter``, and it is deliberately far larger
#: than the wrapper's own default of 10. At 10 every one of these structures
#: stops because the budget ran out, which means the recorded state is an
#: iterate and the reference would pin the optimiser's trajectory -- the one
#: thing an SCF golden must not do. Raised, LBFGS terminates on its own
#: (between 6 and 75 closure evaluations here) and the result stops depending
#: on the budget at all: measured identical, bit for bit, at ``n_scf_step``
#: 200 and 500 on all five moment-carrying fixtures.
#:
#: That still does not make every terminal point a fixed point, and the
#: difference is measured rather than assumed -- see
#: ``tests/golden/test_tiny_magnetic_scf.py``, which perturbs the initial
#: moments and keeps only the structures whose answer follows.
SCF_CONFIG: Dict[str, Any] = {
    "n_scf_step": 200,
    "scf_tol": 1e-10,
    "scf_step_size": 1.0,
    "use_scf": True,
    "use_collinear": False,
}


def build_model() -> torch.nn.Module:
    """Instantiate the magnetic anchor under a fixed seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return MagneticScaleShiftMACE(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
        m_max=ANCHOR_CONFIG["m_max"],
        num_mag_radial_basis=ANCHOR_CONFIG["num_mag_radial_basis"],
        num_mag_radial_basis_one_body=ANCHOR_CONFIG["num_mag_radial_basis_one_body"],
        max_m_ell=ANCHOR_CONFIG["max_m_ell"],
        interaction_cls=modules.interaction_classes[
            ANCHOR_CONFIG["interaction_cls"]
        ],
        interaction_cls_first=modules.interaction_classes[
            ANCHOR_CONFIG["interaction_cls_first"]
        ],
        num_interactions=ANCHOR_CONFIG["num_interactions"],
        num_elements=ANCHOR_CONFIG["num_elements"],
        hidden_irreps=o3.Irreps(ANCHOR_CONFIG["hidden_irreps"]),
        MLP_irreps=o3.Irreps(ANCHOR_CONFIG["MLP_irreps"]),
        atomic_energies=np.array(ATOMIC_ENERGIES, dtype=float),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=ANCHOR_CONFIG["correlation"],
        gate=modules.gate_dict[ANCHOR_CONFIG["gate"]],
        pair_repulsion=ANCHOR_CONFIG["pair_repulsion"],
        distance_transform=ANCHOR_CONFIG["distance_transform"],
        radial_type=ANCHOR_CONFIG["radial_type"],
        use_reduced_cg=ANCHOR_CONFIG["use_reduced_cg"],
        use_magmom_one_body=ANCHOR_CONFIG["use_magmom_one_body"],
        atomic_inter_scale=ANCHOR_CONFIG["atomic_inter_scale"],
        atomic_inter_shift=ANCHOR_CONFIG["atomic_inter_shift"],
    )


def build_anchor(model_path: Path = MODEL_PATH) -> Path:
    """Build, save and document the anchor. Returns the model path."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_model().to(torch.float64)
    finally:
        torch.set_default_dtype(previous)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path)

    sidecar = {
        "model": model_path.name,
        "class": type(model).__name__,
        "recipe": "tests/golden/build_magnetic_anchor.py",
        "command": (
            "python -c \"from tests.golden.build_magnetic_anchor import "
            'build_anchor; build_anchor()"'
        ),
        "regenerate_with": (
            "python tests/golden/regenerate.py --target magnetic "
            "--i-know-what-i-am-doing"
        ),
        "built_by": (
            "direct instantiation; a seeded untrained network exercises the "
            "energy assembly and both derivative paths exactly as a fitted one"
        ),
        "requires": "the `magnetic` extra (sphericart) to build and to load",
        "seed": SEED,
        "dtype": "float64",
        "atomic_energies": ATOMIC_ENERGIES,
        "avg_num_neighbors": AVG_NUM_NEIGHBORS,
        "config": ANCHOR_CONFIG,
        "scf_config": SCF_CONFIG,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    with SIDECAR_PATH.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(build_anchor())
