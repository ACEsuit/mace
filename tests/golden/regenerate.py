"""Regenerate the committed goldens.

Goldens are edit-locked. Every artifact under ``tests/golden/`` -- the
fixtures, the anchor checkpoints, the reference JSONs -- is only rewritten by
this script, and only in a change that does nothing else and explains the
physics of what moved. That is why it refuses to run without
``--i-know-what-i-am-doing``: a regeneration that happens as a side effect of
a feature change destroys the only evidence that the feature changed nothing
it should not have.

Usage::

    python tests/golden/regenerate.py --target all --i-know-what-i-am-doing

Targets, in dependency order: ``fixtures`` (structures + training set),
``anchors`` (both checkpoints and their sidecars), ``references`` (the
snapshot JSONs). ``all`` runs the three in that order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

GOLDEN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from tests.golden import harness  # noqa: E402
from tests.golden.build_mace_anchor import MODEL_PATH as MACE_ANCHOR_PATH  # noqa: E402
from tests.golden.build_mace_anchor import build_anchor  # noqa: E402
from tests.golden.make_fixtures import write_fixtures  # noqa: E402
from tests.golden.train_anchor import MODEL_PATH as SCALESHIFT_ANCHOR_PATH  # noqa: E402
from tests.golden.train_anchor import train_anchor  # noqa: E402

#: anchor name -> (checkpoint, reference file, what the anchor is for)
ANCHORS: Dict[str, tuple] = {
    "tiny_scaleshift": (
        SCALESHIFT_ANCHOR_PATH,
        "tiny_scaleshift_e3nn_cpu_fp64.json",
        "Trained ScaleShiftMACE anchor: the class the training CLI emits, "
        "with the short-range repulsion term inside the scale-shift.",
    ),
    "tiny_mace": (
        MACE_ANCHOR_PATH,
        "tiny_mace_e3nn_cpu_fp64.json",
        "Directly instantiated plain MACE anchor: the class the CLI cannot "
        "produce, with the short-range repulsion term outside any scaling.",
    ),
}


def regenerate_fixtures() -> None:
    written = write_fixtures()
    for name, path in written.items():
        print(f"  fixture  {name:16s} -> {path.relative_to(REPO_ROOT)}")


def regenerate_anchors() -> None:
    print("  training the ScaleShiftMACE anchor (a few seconds)...")
    path = train_anchor()
    print(f"  anchor   tiny_scaleshift  -> {path.relative_to(REPO_ROOT)}")
    path = build_anchor()
    print(f"  anchor   tiny_mace        -> {path.relative_to(REPO_ROOT)}")


def regenerate_references() -> None:
    # Imported here, not at module scope: the fixture and reference targets
    # must stay runnable in an environment where the framework is importable
    # but heavy, and nothing above this point needs torch.
    import torch  # pylint: disable=import-outside-toplevel

    from mace.calculators import MACECalculator  # pylint: disable=import-outside-toplevel

    fixtures = harness.load_fixtures()
    for name, (model_path, reference_name, description) in ANCHORS.items():
        model = torch.load(model_path, weights_only=False, map_location="cpu")
        calc = MACECalculator(models=[model], device="cpu", default_dtype="float64")
        snapshot = harness.snapshot_outputs(
            calc,
            fixtures,
            dtype="float64",
            device="cpu",
            backend="e3nn",
            metadata={"model_class": type(model).__name__},
        )
        path = harness.write_reference(
            harness.REFERENCES_DIR / reference_name,
            snapshot,
            provenance={
                "source": f"tests/golden/models/{model_path.name}",
                "recipe": (
                    "tests/golden/train_anchor.py"
                    if name == "tiny_scaleshift"
                    else "tests/golden/build_mace_anchor.py"
                ),
                "description": description,
                "evaluated_with": "mace.calculators.MACECalculator, e3nn, CPU, float64",
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            allow_overwrite=True,
        )
        print(f"  reference {name:15s} -> {path.relative_to(REPO_ROOT)}")


TARGETS = {
    "fixtures": regenerate_fixtures,
    "anchors": regenerate_anchors,
    "references": regenerate_references,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["all", *TARGETS],
        default="all",
        help="which golden artifacts to rewrite",
    )
    parser.add_argument(
        "--i-know-what-i-am-doing",
        action="store_true",
        help=(
            "required. Regenerating a golden discards the evidence it was "
            "collected to provide; do it in its own reviewed change."
        ),
    )
    args = parser.parse_args(argv)
    if not args.i_know_what_i_am_doing:
        parser.error(
            "refusing to rewrite committed goldens without "
            "--i-know-what-i-am-doing. These files are the reference the "
            "rewrite is measured against; regenerating one turns a failing "
            "test into a passing one without anyone deciding that the new "
            "numbers are correct."
        )
    names = list(TARGETS) if args.target == "all" else [args.target]
    for name in names:
        print(f"[{name}]")
        TARGETS[name]()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
