"""The two tiny checkpoints every other golden is measured against."""

from __future__ import annotations

from typing import Dict

from tests.golden.paths import REPO_ROOT

ORDER = 20
HELP = "the tiny anchor checkpoints and their build sidecars"

#: anchor name -> (checkpoint, reference file, what the anchor is for)
ANCHORS: Dict[str, tuple] = {
    "tiny_scaleshift": (
        "tests/golden/models/tiny_scaleshift.model",
        "tiny_scaleshift_e3nn_cpu_fp64.json",
        "Trained ScaleShiftMACE anchor: the class the training CLI emits, "
        "with the short-range repulsion term inside the scale-shift.",
    ),
    "tiny_mace": (
        "tests/golden/models/tiny_mace.model",
        "tiny_mace_e3nn_cpu_fp64.json",
        "Directly instantiated plain MACE anchor: the class the CLI cannot "
        "produce, with the short-range repulsion term outside any scaling.",
    ),
}

#: The recipe that produces each one, recorded in the reference provenance.
RECIPES = {
    "tiny_scaleshift": "tests/golden/train_anchor.py",
    "tiny_mace": "tests/golden/build_mace_anchor.py",
}


def checkpoint(name: str):
    """The absolute path of an anchor checkpoint."""
    return REPO_ROOT / ANCHORS[name][0]


def run() -> None:
    # Late import for the reason given in the package docstring: --help must
    # work without torch.
    from tests.golden.build_mace_anchor import (  # pylint: disable=import-outside-toplevel
        build_anchor,
    )
    from tests.golden.train_anchor import (  # pylint: disable=import-outside-toplevel
        train_anchor,
    )

    print("  training the ScaleShiftMACE anchor (a few seconds)...")
    path = train_anchor()
    print(f"  anchor   tiny_scaleshift  -> {path.relative_to(REPO_ROOT)}")
    path = build_anchor()
    print(f"  anchor   tiny_mace        -> {path.relative_to(REPO_ROOT)}")
