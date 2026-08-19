"""One training step per anchor, reduced to a per-parameter gradient digest."""

from __future__ import annotations

from pathlib import Path

from tests.golden import harness, targets
from tests.golden.paths import REPO_ROOT

ORDER = 35
HELP = "the one-training-step gradient digests"


def run() -> None:
    # Late import: see the package docstring. A gradient step needs the whole
    # framework.
    from tests.golden.train_step import (  # pylint: disable=import-outside-toplevel
        GRADIENT_REFERENCES,
        snapshot,
    )

    # Read through the shared registry rather than importing the anchors
    # module: this family consumes checkpoints it does not own, and going
    # through all_anchors() is what keeps that a lookup instead of a
    # dependency between two families.
    anchors = targets.all_anchors()
    for name, reference_name in GRADIENT_REFERENCES.items():
        payload = snapshot(name)
        path = harness.write_reference(
            harness.REFERENCES_DIR / reference_name,
            payload,
            provenance={
                "source": f"tests/golden/models/{Path(anchors[name][0]).name}",
                "recipe": "tests/golden/train_step.py",
                "description": (
                    "One forward+backward of an energy+forces loss on the "
                    f"{name} anchor, reduced to a per-parameter gradient "
                    "digest. A rewrite of the training path has to reproduce "
                    "these, and a rewrite that changes them has to say which "
                    "gradient moved and why."
                ),
                "evaluated_with": "e3nn, CPU, float64",
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            allow_overwrite=True,
        )
        print(f"  gradient  {name:15s} -> {path.relative_to(REPO_ROOT)}")
