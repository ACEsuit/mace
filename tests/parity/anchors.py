"""Load the committed legacy anchors without leaking their environment flag."""

from pathlib import Path

import torch

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def load_anchor_weights(path: Path, model: torch.nn.Module) -> None:
    """Install frozen numerical-reference tensors in their canonical roles.

    These edit-locked fixtures are not native release checkpoints. Their old
    container headers do not define a format that the product must accept.
    """
    from mace_torch.serialization import load_canonical_state
    from safetensors.torch import load_file

    path.resolve().relative_to((GOLDEN / "models").resolve())
    state = {}
    for key, value in load_file(str(path.with_suffix(".safetensors"))).items():
        module, name = key.split("::", 1)
        # Match numerical roles after naming changes, without accepting these
        # edit-locked oracle containers as product checkpoints.
        module = module.replace("outputs.heads.", "outputs.readouts.").replace(
            "outputs.energy_head", "outputs.energy_readout"
        )
        state.setdefault(module, {})[name] = value
    load_canonical_state(model, state)
