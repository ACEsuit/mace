"""Build the tiny ``AtomicDielectricMACE`` anchor by direct instantiation.

Same discipline as ``build_dipole_anchor.py``: a seeded untrained network
exercises the readout assembly exactly as a trained one does, so the anchor is
initialised and saved rather than fitted. This is the class behind MACE-MDP,
and it shares less with the dipole anchor than its name suggests:

* every readout is a ``LinearDipolePolarReadoutBlock`` /
  ``NonLinearDipolePolarReadoutBlock`` emitting ``2x0e + 1x1o + 1x2e`` per atom
  (``mace/modules/blocks.py:244``):
  a charge, the isotropic part of the polarizability, a dipole and the
  anisotropic part (``mace/modules/models.py:1110-1118``). The last layer
  keeps the whole of ``hidden_irreps`` instead of narrowing to ``l=1``
  (``:1007``), so its features carry the ``2e`` block the polarizability
  needs. ``hidden_irreps`` therefore includes ``2e``: without it the
  anisotropic polarizability is exactly zero, and a reference taken on it would
  pin nothing.
* the charges are predicted, not read from the structure. Each structure's
  charges are shifted by one constant so they sum to ``data["total_charge"]``
  (``:1132-1135``), and the graph dipole adds ``sum_i q_i r_i`` in e*Angstrom,
  without the Debye conversion the dipole class applies
  (``compute_fixed_charge_dipole_polar``, ``mace/modules/utils.py:667-676``).
* ``forward(..., compute_dielectric_derivatives=True)`` returns the full
  position Jacobians ``dmu_dr`` and ``dalpha_dr`` (``:1168-1176``), which is
  what an infrared or Raman intensity is computed from.

Two constructor arguments are worth stating:

* ``atomic_energies`` **must** be ``None``; the class asserts it (``:925``).
* ``use_reduced_cg`` is accepted and ignored (``:870``), so the product basis
  builds the plain e3nn Clebsch-Gordan basis whether or not cuequivariance is
  installed. It is left out of the recipe for that reason, and
  ``tests/golden/test_tiny_dielectric.py`` measures it rather than assuming it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from e3nn import o3

from mace import modules

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "tiny_dielectric.model"
SIDECAR_PATH = MODELS_DIR / "tiny_dielectric.build.json"

SEED = 20261105

#: Three elements, matching the fixture set.
ATOMIC_NUMBERS = [1, 6, 8]

AVG_NUM_NEIGHBORS = 3.0

ANCHOR_CONFIG: dict[str, Any] = {
    "r_max": 3.5,
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "RealAgnosticInteractionBlock",
    "num_interactions": 2,
    "num_elements": len(ATOMIC_NUMBERS),
    "hidden_irreps": "4x0e + 4x1o + 4x2e",
    "MLP_irreps": "4x0e + 4x1o + 4x2e",
    "atomic_numbers": ATOMIC_NUMBERS,
    "correlation": 2,
    "radial_type": "bessel",
    "radial_MLP": [64, 64, 64],
    "gate": "silu",
}


def build_model() -> torch.nn.Module:
    """Instantiate the ``AtomicDielectricMACE`` anchor under a fixed seed.

    The seed is set inside a forked generator, so the caller's random state is
    the same afterwards as before."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        return _instantiate()


def _instantiate() -> torch.nn.Module:
    return modules.AtomicDielectricMACE(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
        interaction_cls=modules.interaction_classes[ANCHOR_CONFIG["interaction_cls"]],
        interaction_cls_first=modules.interaction_classes[
            ANCHOR_CONFIG["interaction_cls_first"]
        ],
        num_interactions=ANCHOR_CONFIG["num_interactions"],
        num_elements=ANCHOR_CONFIG["num_elements"],
        hidden_irreps=o3.Irreps(ANCHOR_CONFIG["hidden_irreps"]),
        MLP_irreps=o3.Irreps(ANCHOR_CONFIG["MLP_irreps"]),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=ANCHOR_CONFIG["correlation"],
        gate=modules.gate_dict[ANCHOR_CONFIG["gate"]],
        radial_type=ANCHOR_CONFIG["radial_type"],
        radial_MLP=ANCHOR_CONFIG["radial_MLP"],
        # Asserted None by the class: this is not an energy model.
        atomic_energies=None,
    )


def build_anchor(model_path: Path) -> Path:
    """Build, save and document the anchor. Returns the model path.

    `model_path` is required rather than defaulted to `MODEL_PATH`, because
    this writes over whatever it is given and the committed anchor is what the
    reference in this directory was recorded against."""
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
        "recipe": "tests/golden/build_dielectric_anchor.py",
        "command": (
            'python -c "from tests.golden.build_dielectric_anchor import '
            'build_anchor, MODEL_PATH; build_anchor(MODEL_PATH)"'
        ),
        "regenerate_with": (
            "python tests/golden/regenerate.py --target dielectric "
            "--i-know-what-i-am-doing"
        ),
        "built_by": (
            "direct instantiation; the training CLI can emit this class "
            "(--model AtomicDielectricMACE) but only against a dataset labelled "
            "with dipoles and polarizabilities, and an untrained network "
            "exercises the same assembly"
        ),
        "seed": SEED,
        "dtype": "float64",
        "atomic_energies": None,
        "avg_num_neighbors": AVG_NUM_NEIGHBORS,
        "config": ANCHOR_CONFIG,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    sidecar_path = model_path.with_suffix(".build.json")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(build_anchor(MODEL_PATH))
