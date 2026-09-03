"""The tiny AtomicDipolesMACE anchor and its reference.

Its own family, not part of ``references``: the dipole anchor emits no
energy, so an ase calculator cannot be driven for it at all, and its channels
reach no calculator's results dict. It is snapshotted through the model
forward instead.
"""

from __future__ import annotations

from typing import Dict

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

ORDER = 36
HELP = "the dipole anchor and its reference"

#: Only the molecular fixtures. This is an organic-chemistry model, and the
#: dipole of a periodic cell is origin-dependent, so a golden taken on the
#: slabs or the triclinic cell would pin a number that has no physical
#: meaning while looking exactly like one that does.
MOLECULAR_TAGS = ("molecular",)

#: ...and by chemistry, not by tag alone. The manifest is shared with every
#: other family, so a selection that names only a tag starts evaluating
#: whatever the next family adds to it. This anchor knows H/C/O; an iron
#: fixture is a missing z-table entry, not a tolerance failure.
DIPOLE_ELEMENTS = (1, 6, 8)

#: anchor name -> (checkpoint, reference file, what the anchor is for)
ANCHORS: Dict[str, tuple] = {
    "tiny_dipoles": (
        "tests/golden/models/tiny_dipoles.model",
        "tiny_dipoles_e3nn_cpu_fp64.json",
        "Directly instantiated AtomicDipolesMACE anchor: the graph dipole is "
        "the scatter-sum of the per-atom dipoles plus the fixed-charge "
        "baseline, on the molecular fixtures only.",
    ),
}


def _dipole_projection(out: dict) -> dict:
    """The forward's dipole dict in the snapshot's vocabulary.

    Graph-level channels are declared per graph, so the single graph is
    indexed out here; see tests/golden/model_keys.py, note 4.
    """
    from tests.golden.routes import as_numpy  # pylint: disable=import-outside-toplevel

    return {
        "dipole": as_numpy(out["dipole"][0]),
        "atomic_dipoles": as_numpy(out["atomic_dipoles"]),
    }


def run() -> None:
    # Late imports: the builder pulls in the framework, and --help has to
    # work where torch cannot be loaded.
    import torch  # pylint: disable=import-outside-toplevel

    from tests.golden.build_dipole_anchor import (  # pylint: disable=import-outside-toplevel
        MODEL_PATH as DIPOLE_ANCHOR_PATH,
    )

    from tests.golden.build_dipole_anchor import (  # pylint: disable=import-outside-toplevel
        build_anchor as build_dipole_anchor,
    )

    path = build_dipole_anchor()
    print(f"  anchor   tiny_dipoles     -> {path.relative_to(REPO_ROOT)}")

    # Snapshotted through the forward, not a calculator: see the module
    # docstring and tests/golden/routes.py.
    from tests.golden.routes import ForwardRoute  # pylint: disable=import-outside-toplevel

    model = torch.load(DIPOLE_ANCHOR_PATH, weights_only=False, map_location="cpu").to(
        torch.float64
    )
    snapshot = harness.snapshot_outputs(
        ForwardRoute(model, _dipole_projection),
        harness.load_fixtures(tags=MOLECULAR_TAGS, elements=DIPOLE_ELEMENTS),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={"model_class": type(model).__name__},
    )
    path = harness.write_reference(
        harness.REFERENCES_DIR / "tiny_dipoles_e3nn_cpu_fp64.json",
        snapshot,
        provenance={
            "source": f"tests/golden/models/{DIPOLE_ANCHOR_PATH.name}",
            "recipe": "tests/golden/build_dipole_anchor.py",
            "description": (
                "Directly instantiated AtomicDipolesMACE anchor: the graph "
                "dipole is the scatter-sum of the per-atom dipoles plus the "
                "fixed-charge baseline, on the molecular fixtures only."
            ),
            "evaluated_with": (
                "AtomicDipolesMACE.forward via tests/golden/routes.ForwardRoute, "
                "e3nn, CPU, float64"
            ),
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    print(f"  reference {'tiny_dipoles':15s} -> {path.relative_to(REPO_ROOT)}")
