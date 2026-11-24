"""The tiny AtomicDielectricMACE anchor and its reference.

Its own family, for the reason the dipole anchor is: the class emits no
energy, so no ase calculator can be driven for it, and it is snapshotted
through the model forward. The forward is asked for the position Jacobians as
well, since they are a separate code path from the outputs they differentiate.
"""

from __future__ import annotations

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

ORDER = 39
HELP = "the dielectric anchor and its reference"

#: Named rather than selected by tag and element. They are today exactly the
#: molecular H/C/O fixtures, and naming them keeps a fixture added to the
#: shared manifest later from widening this reference without a regeneration.
#: Molecular only, because a dipole under periodic boundaries depends on the
#: choice of cell.
FIXTURE_NAMES = ("dimer_short", "isolated_atom", "water_cluster")

REFERENCE_NAME = "tiny_dielectric_e3nn_cpu_fp64.json"

#: The channels the forward returns, all of them.
CHANNELS = (
    "charges",
    "dipole",
    "atomic_dipoles",
    "polarizability",
    "polarizability_sh",
    "dmu_dr",
    "dalpha_dr",
)

#: Declared per graph, so the single graph is indexed out; see
#: tests/golden/model_keys.py, note 4.
GRAPH_CHANNELS = ("dipole", "polarizability", "polarizability_sh")

#: The Jacobians are taken only when the forward is asked for them.
FORWARD_KWARGS = {"compute_dielectric_derivatives": True}


def projection(out: dict) -> dict:
    """The forward's output dict in the snapshot's vocabulary."""
    from tests.golden.routes import as_numpy  # pylint: disable=import-outside-toplevel

    return {
        name: as_numpy(out[name][0] if name in GRAPH_CHANNELS else out[name])
        for name in CHANNELS
    }


def run() -> None:
    # Late imports: the builder pulls in the framework, and --help has to
    # work where torch cannot be loaded.
    from tests.golden.anchors import (  # pylint: disable=import-outside-toplevel
        load_checkpoint,
    )
    from tests.golden.build_dielectric_anchor import (  # pylint: disable=import-outside-toplevel
        MODEL_PATH,
        build_anchor,
    )
    from tests.golden.routes import (  # pylint: disable=import-outside-toplevel
        ForwardRoute,
    )

    path = build_anchor(MODEL_PATH)
    print(f"  anchor   tiny_dielectric  -> {path.relative_to(REPO_ROOT)}")

    model = load_checkpoint(MODEL_PATH)
    snapshot = harness.snapshot_outputs(
        ForwardRoute(model, projection, forward_kwargs=FORWARD_KWARGS),
        harness.load_fixtures(names=FIXTURE_NAMES),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={"model_class": type(model).__name__},
    )
    path = harness.write_reference(
        harness.REFERENCES_DIR / REFERENCE_NAME,
        snapshot,
        provenance={
            "source": f"tests/golden/models/{MODEL_PATH.name}",
            "recipe": "tests/golden/build_dielectric_anchor.py",
            "description": (
                "Directly instantiated AtomicDielectricMACE anchor: predicted "
                "charges, the graph dipole with its charge baseline, the "
                "Cartesian and spherical polarizability, and the position "
                "Jacobians of both, on the molecular H/C/O fixtures."
            ),
            "evaluated_with": (
                "AtomicDielectricMACE.forward(compute_dielectric_derivatives="
                "True) via tests/golden/routes.ForwardRoute, e3nn, CPU, float64"
            ),
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    print(f"  reference {'tiny_dielectric':15s} -> {path.relative_to(REPO_ROOT)}")
