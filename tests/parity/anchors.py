"""Load the committed legacy anchors without leaking their environment flag."""

import os
from pathlib import Path

import torch

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def load_anchor(name: str):
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        return torch.load(
            GOLDEN / "models" / name, map_location="cpu", weights_only=False
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
