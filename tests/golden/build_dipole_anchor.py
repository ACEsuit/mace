"""Build the tiny ``AtomicDipolesMACE`` anchor by direct instantiation.

Same discipline as ``build_mace_anchor.py`` and for the same reason: a seeded
untrained network exercises the readout assembly exactly as a trained one
does, so the anchor is initialised and saved rather than fitted. What it
exists to pin is the dipole assembly, which is not a rescaled energy and
shares no code with one:

* the readouts are ``LinearDipoleReadoutBlock`` / ``NonLinearDipoleReadoutBlock``
  with ``dipole_only=True``, and only the *last* interaction is narrowed to
  the ``l=1`` block of ``hidden_irreps`` (``mace/modules/models.py:723-731``),
  so the per-layer contributions this anchor sums are not all the same shape;
* the graph dipole is the scatter-sum of the per-atom dipoles **plus** a
  fixed-charge baseline computed from ``data["charges"]``
  (``compute_fixed_charge_dipole``, ``mace/modules/models.py:825-831``). The
  baseline is the term a rewrite is most likely to drop or to re-origin, and
  it is invisible in any energy anchor.

Two constructor arguments are worth stating rather than leaving to be
rediscovered:

* ``atomic_energies`` **must** be ``None`` -- the class asserts it (``:664``).
  This is not an energy model, and there is no E0 table to record.
* ``use_reduced_cg`` is accepted and ignored by this class (it is one of the
  ``pylint: disable=unused-argument`` parameters at ``:648``), so nothing is
  forwarded to ``EquivariantProductBasisBlock`` and its ``None`` default
  reaches ``SymmetricContractionWrapper``, where ``use_reduced_cg and
  CUET_AVAILABLE`` is falsy either way (``mace/modules/wrapper_ops.py:428``).
  The plain-``MACE`` anchor has to pin the flag to ``False`` because its
  ``True`` default is silently degraded when cuequivariance is absent; here
  the reduced path is unreachable, so the weights do not depend on what
  happens to be installed. That is asserted, not assumed, in
  ``tests/golden/test_tiny_dipoles.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from e3nn import o3

from mace import modules

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "tiny_dipoles.model"
SIDECAR_PATH = MODELS_DIR / "tiny_dipoles.build.json"

#: Deliberately the same seed as the energy anchors: the three checkpoints are
#: one committed set and there is nothing to gain from three seeds.
SEED = 20260810

#: Three elements, matching the fixture set.
ATOMIC_NUMBERS = [1, 6, 8]

AVG_NUM_NEIGHBORS = 8.0

ANCHOR_CONFIG: Dict[str, Any] = {
    "r_max": 3.5,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "num_interactions": 2,
    "num_elements": len(ATOMIC_NUMBERS),
    # At least one l=1 block is mandatory: the class asserts
    # len(hidden_irreps) > 1 before narrowing the last layer to hidden_irreps[1]
    # (mace/modules/models.py:723-728). A scalar-only anchor cannot exist.
    "hidden_irreps": "16x0e + 16x1o",
    "MLP_irreps": "8x0e",
    "atomic_numbers": ATOMIC_NUMBERS,
    "correlation": 3,
    "radial_type": "bessel",
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "RealAgnosticInteractionBlock",
    "gate": "silu",
}


def build_model() -> torch.nn.Module:
    """Instantiate the ``AtomicDipolesMACE`` anchor under a fixed seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return modules.AtomicDipolesMACE(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
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
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=ANCHOR_CONFIG["correlation"],
        gate=modules.gate_dict[ANCHOR_CONFIG["gate"]],
        radial_type=ANCHOR_CONFIG["radial_type"],
        # Asserted None by the class: this is not an energy model.
        atomic_energies=None,
    )


def build_anchor(model_path: Path) -> Path:
    """Build, save and document the anchor. Returns the model path.

    `model_path` is required rather than defaulted to `MODEL_PATH`: this writes
    over whatever it is given, and the committed anchor is what every reference
    in this directory was recorded against. A default meant that calling this
    with no argument, from a REPL or a half-remembered one-liner, silently
    replaced the checkpoint with a fresh one -- same recipe, different bytes."""
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
        "recipe": "tests/golden/build_dipole_anchor.py",
        "command": (
            "python -c \"from tests.golden.build_dipole_anchor import "
            'build_anchor, MODEL_PATH; build_anchor(MODEL_PATH)"'
        ),
        "regenerate_with": (
            "python tests/golden/regenerate.py --target dipoles "
            "--i-know-what-i-am-doing"
        ),
        "built_by": (
            "direct instantiation; the training CLI can emit this class "
            "(--model AtomicDipolesMACE) but only against a dipole-labelled "
            "dataset, and an untrained network exercises the same assembly"
        ),
        "seed": SEED,
        "dtype": "float64",
        "atomic_energies": None,
        "avg_num_neighbors": AVG_NUM_NEIGHBORS,
        "config": ANCHOR_CONFIG,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    # Beside the model that was actually written, not beside the committed
    # one: a build into a temporary directory used to leave the checkpoint
    # there and overwrite the committed sidecar in place.
    sidecar_path = model_path.with_suffix(".build.json")
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(build_anchor(MODEL_PATH))
