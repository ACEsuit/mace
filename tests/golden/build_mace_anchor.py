"""Build the plain-``MACE`` parity anchor by direct instantiation.

This anchor is deliberately *not* produced by ``mace_run_train``, and that is
not a matter of convenience. ``--model MACE`` returns a ``ScaleShiftMACE``
with ``atomic_inter_scale=args.std`` and the shift zeroed
(``mace/tools/model_script_utils.py:279-296``), so the CLI cannot emit a
plain ``MACE`` at all; a CLI recipe would silently anchor the wrong class.

What the anchor exists to pin is the plain-``MACE`` energy assembly, and in
particular where the short-range repulsion term enters. ``MACE`` appends the
pair term to ``energies`` / ``node_energies_list`` next to ``e0``
(``mace/modules/models.py:359-361``) and never scales it, while
``ScaleShiftMACE`` seeds its readout sum with ``[pair_node_energy]``
(``:539``) and puts the whole sum through ``scale_shift`` (``:579``) -- so
the same term is scaled in one class and raw in the other. Only anchoring
both classes turns that divergence into a number a rewrite has to reproduce.
A seeded untrained network exercises that assembly exactly as a trained one
does, so this anchor is initialised and saved, not fitted.

Note for the GPU parity work: this anchor is not convertible.
``extract_config_mace_model`` whitelists ``ScaleShiftMACE`` and the extension
classes, and returns an ``{"error": ...}`` payload for a plain ``MACE``, so
backend parity runs on the trained ``ScaleShiftMACE`` anchor and pins the
refusal here as a contract instead.
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
MODEL_PATH = MODELS_DIR / "tiny_mace.model"
SIDECAR_PATH = MODELS_DIR / "tiny_mace.build.json"

SEED = 20260810

#: Three elements, matching the fixture set.
ATOMIC_NUMBERS = [1, 6, 8]

#: Fixed, not fitted: this anchor is never trained, so its reference energies
#: are part of the recipe rather than a dataset statistic.
ATOMIC_ENERGIES = [-1.5, -5.25, -8.75]

AVG_NUM_NEIGHBORS = 8.0

ANCHOR_CONFIG: Dict[str, Any] = {
    "r_max": 3.5,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "num_interactions": 2,
    "num_elements": len(ATOMIC_NUMBERS),
    "hidden_irreps": "16x0e + 16x1o",
    "MLP_irreps": "8x0e",
    "atomic_numbers": ATOMIC_NUMBERS,
    "correlation": 3,
    "radial_type": "bessel",
    "distance_transform": "None",
    "pair_repulsion": True,
    # Pinned to False on purpose. The default is True, and
    # `mace/modules/wrapper_ops.py:428` then silently degrades it to False
    # when cuequivariance is absent -- so an anchor built with the default
    # would have different weights, and different outputs, depending on what
    # happens to be installed on the machine that built it. A cross-machine
    # golden cannot depend on an optional import.
    "use_reduced_cg": False,
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "RealAgnosticInteractionBlock",
    "gate": "silu",
}


def build_model() -> torch.nn.Module:
    """Instantiate the plain ``MACE`` anchor under a fixed seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return modules.MACE(
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
        atomic_energies=np.array(ATOMIC_ENERGIES, dtype=float),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=ANCHOR_CONFIG["correlation"],
        gate=modules.gate_dict[ANCHOR_CONFIG["gate"]],
        pair_repulsion=ANCHOR_CONFIG["pair_repulsion"],
        distance_transform=ANCHOR_CONFIG["distance_transform"],
        radial_type=ANCHOR_CONFIG["radial_type"],
        use_reduced_cg=ANCHOR_CONFIG["use_reduced_cg"],
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
        "recipe": "tests/golden/build_mace_anchor.py",
        "command": (
            "python -c \"from tests.golden.build_mace_anchor import "
            'build_anchor; build_anchor()"'
        ),
        "regenerate_with": (
            "python tests/golden/regenerate.py --target anchors "
            "--i-know-what-i-am-doing"
        ),
        "built_by": "direct instantiation (the CLI cannot emit a plain MACE)",
        "seed": SEED,
        "dtype": "float64",
        "atomic_energies": ATOMIC_ENERGIES,
        "avg_num_neighbors": AVG_NUM_NEIGHBORS,
        "config": ANCHOR_CONFIG,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    with SIDECAR_PATH.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(build_anchor())
