"""Build the ``MACELES`` long-range-electrostatics anchor by instantiation.

Like the plain-``MACE`` anchor and unlike the trained one, this checkpoint is
initialised under a fixed seed and saved, not fitted. Training it would add a
dependency on the optimiser and on eight epochs of arithmetic without
exercising one extra line of the thing being pinned: what this anchor exists
to hold down is the *assembly* -- which latent readouts exist, what shape each
latent quantity comes back in, where the long-range energy enters the total,
and what the Ewald sum makes of it -- and a seeded untrained network runs all
of that exactly as a trained one does.

The architecture is the ``tiny_scaleshift`` anchor's, deliberately: same
cutoff, same irreps, same three elements, same ZBL. ``MACELES`` is a
``ScaleShiftMACE`` subclass, so the only intended difference between the two
checkpoints is the LES head -- plus one thing the class forces and the recipe
cannot: ``keep_last_layer_irreps=True`` (``mace/modules/extensions.py:146``),
without which the last layer would drop its vector features and the dipole,
quadrupole and polarizability readouts would have nothing to read.

The LES configuration is **not** written here. It lives in the committed
``models/tiny_maceles.les_arguments.yaml`` and is read through the same
``read_yaml`` the CLI's ``--les_arguments`` uses, so the file next to the
checkpoint is the file that built it rather than a transcription of it.

Two properties of the checkpoint are worth stating because they are load
bearing for the reference:

* the latent polarizability is **isotropic** (``[n_atoms]``, not
  ``[n_atoms, 3, 3]``), which is the layout ``make_alpha_positive`` used to
  skip -- the flag tested ``dim() == 2`` and the scalar readout emits
  ``dim() == 1`` after its advanced-index collapse, so negative alphas went
  into the Ewald sum with the flag on. Fixed on develop; the anchor is
  configured so the golden fails again if it regresses.
* ``BEC`` comes back ``[n_atoms, 2, 3, 3]``, not ``[n_atoms, 3, 3]``: with
  dipoles on, ``les`` stacks the charge and dipole contributions
  (``les/module/bec.py:100-104``). That is why the shared schema declares
  ``BEC`` per-atom-matrix rather than per-atom-tensor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.tools.arg_parser import read_yaml

GOLDEN_ROOT = Path(__file__).resolve().parent
MODELS_DIR = GOLDEN_ROOT / "models"
MODEL_PATH = MODELS_DIR / "tiny_maceles.model"
SIDECAR_PATH = MODELS_DIR / "tiny_maceles.build.json"
LES_ARGUMENTS_PATH = MODELS_DIR / "tiny_maceles.les_arguments.yaml"

SEED = 20260810

#: Three elements, matching the fixture set and the other anchors.
ATOMIC_NUMBERS = [1, 6, 8]

#: Fixed, not fitted, for the same reason as in build_mace_anchor.py.
ATOMIC_ENERGIES = [-1.5, -5.25, -8.75]

AVG_NUM_NEIGHBORS = 8.0

#: Not 1.0 on purpose. `MACELES` puts the short-range readout sum through
#: `scale_shift` and adds the LES energy *outside* it
#: (mace/modules/extensions.py:517-522, :604), so a unit scale would make the
#: reference unable to tell the two placements apart.
ATOMIC_INTER_SCALE = 1.3

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
    # Pinned to False for the reason build_mace_anchor.py records: the default
    # is True and degrades silently to False when cuequivariance is absent, so
    # an anchor built with the default would carry different weights depending
    # on what happened to be installed on the machine that built it.
    "use_reduced_cg": False,
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "RealAgnosticInteractionBlock",
    "gate": "silu",
    "atomic_inter_scale": ATOMIC_INTER_SCALE,
    "atomic_inter_shift": 0.0,
}


def load_les_arguments(path: Path = LES_ARGUMENTS_PATH) -> Dict[str, Any]:
    """The committed LES configuration, read the way the CLI reads it."""
    return read_yaml(str(path))


def build_model(les_arguments: Dict[str, Any] | None = None) -> torch.nn.Module:
    """Instantiate the ``MACELES`` anchor under a fixed seed."""
    from mace.modules.extensions import MACELES  # noqa: PLC0415

    if les_arguments is None:
        les_arguments = load_les_arguments()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return MACELES(
        les_arguments=les_arguments,
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
        atomic_inter_scale=ANCHOR_CONFIG["atomic_inter_scale"],
        atomic_inter_shift=ANCHOR_CONFIG["atomic_inter_shift"],
    )


def build_anchor(model_path: Path) -> Path:
    """Build, save and document the anchor. Returns the model path.

    `model_path` is required rather than defaulted to `MODEL_PATH`: this writes
    over whatever it is given, and the committed anchor is what every reference
    in this directory was recorded against. A default meant that calling this
    with no argument, from a REPL or a half-remembered one-liner, silently
    replaced the checkpoint with a fresh one -- same recipe, different bytes."""
    from tests.golden.les_pin import installed_les_commit  # noqa: PLC0415

    les_arguments = load_les_arguments()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = build_model(les_arguments).to(torch.float64)
    finally:
        torch.set_default_dtype(previous)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path)

    sidecar = {
        "model": model_path.name,
        "class": type(model).__name__,
        "recipe": "tests/golden/build_maceles_anchor.py",
        "command": (
            "python -c \"from tests.golden.build_maceles_anchor import "
            'build_anchor, MODEL_PATH; build_anchor(MODEL_PATH)"'
        ),
        "regenerate_with": (
            "python tests/golden/regenerate.py --target les "
            "--i-know-what-i-am-doing"
        ),
        "built_by": (
            "direct instantiation, under the seed below. Not trained: this "
            "anchor pins the LES assembly and the Ewald sum, both of which a "
            "seeded untrained network exercises identically."
        ),
        "seed": SEED,
        "dtype": "float64",
        "atomic_energies": ATOMIC_ENERGIES,
        "avg_num_neighbors": AVG_NUM_NEIGHBORS,
        "config": ANCHOR_CONFIG,
        "les_arguments_file": LES_ARGUMENTS_PATH.name,
        "les_arguments": les_arguments,
        "les_commit": installed_les_commit(),
        "keep_last_layer_irreps": True,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "note": (
            "MACELES forces keep_last_layer_irreps=True on its base class "
            "(mace/modules/extensions.py:146), so the last layer keeps its "
            "vector features and the LES readouts have something to read. "
            "The architecture is otherwise the tiny_scaleshift anchor's."
        ),
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
