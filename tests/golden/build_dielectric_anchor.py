"""Seeded legacy-only AtomicDielectricMACE anchor for FM-5/ARCH-3.

The recipe follows build_dipole_anchor.py. No native model or conversion code
participates in building the checkpoint or recording its reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from e3nn import o3

from mace import modules

MODEL_PATH = Path(__file__).resolve().parent / "models/tiny_dielectric.model"
SEED = 20261105
ANCHOR_CONFIG = {
    "r_max": 3.5,
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "interaction_cls": "RealAgnosticResidualInteractionBlock",
    "interaction_cls_first": "RealAgnosticInteractionBlock",
    "num_interactions": 2,
    "num_elements": 3,
    "hidden_irreps": "4x0e + 4x1o + 4x2e",
    "MLP_irreps": "4x0e + 4x1o + 4x2e",
    "avg_num_neighbors": 3.0,
    "atomic_numbers": [1, 6, 8],
    "correlation": 2,
    "gate": "silu",
    "radial_MLP": [64, 64, 64],
    "use_polarizability": True,
    # This legacy class always constructs the full e3nn contraction basis.
    "use_reduced_cg": False,
}


def build_model():
    """Construct in fp64 without changing the caller's RNG or default dtype."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            config = dict(ANCHOR_CONFIG)
            for key in ("interaction_cls", "interaction_cls_first"):
                config[key] = modules.interaction_classes[config[key]]
            for key in ("hidden_irreps", "MLP_irreps"):
                config[key] = o3.Irreps(config[key])
            config["gate"] = modules.gate_dict[config["gate"]]
            return modules.AtomicDielectricMACE(**config).eval()
    finally:
        torch.set_default_dtype(previous)


def build_anchor(model_path: Path) -> Path:
    """Save only to the explicitly supplied destination."""
    model = build_model()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path)
    sidecar = {
        "model": model_path.name,
        "class": type(model).__name__,
        "recipe": "tests/golden/build_dielectric_anchor.py",
        "regenerate_with": "python tests/golden/regenerate.py --target dielectric --i-know-what-i-am-doing",
        "seed": SEED,
        "dtype": "float64",
        "config": ANCHOR_CONFIG,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    model_path.with_suffix(".build.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path
