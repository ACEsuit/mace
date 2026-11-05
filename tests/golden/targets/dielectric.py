"""The tiny dielectric anchor, including both full position Jacobians."""

from tests.golden import harness

ORDER = 38
HELP = "the tiny dielectric anchor and its response/derivative reference"
FIXTURE_NAMES = ("dimer_short", "isolated_atom", "water_cluster")
REFERENCE_PATH = harness.REFERENCES_DIR / "tiny_dielectric_e3nn_cpu_fp64.json"
QUANTITIES = (
    "charges",
    "dipole",
    "atomic_dipoles",
    "polarizability",
    "polarizability_sh",
    "dmu_dr",
    "dalpha_dr",
)


def projection(output):
    from tests.golden.routes import as_numpy

    graph_outputs = {"dipole", "polarizability", "polarizability_sh"}
    return {
        name: as_numpy(output[name][0] if name in graph_outputs else output[name])
        for name in QUANTITIES
    }


def snapshot(model):
    from tests.golden.routes import ForwardRoute

    return harness.snapshot_outputs(
        ForwardRoute(
            model, projection, forward_kwargs={"compute_dielectric_derivatives": True}
        ),
        harness.load_fixtures(names=FIXTURE_NAMES),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={"model_class": type(model).__name__},
    )


def run():
    import torch

    from tests.golden.build_dielectric_anchor import MODEL_PATH, build_anchor

    build_anchor(MODEL_PATH)
    model = torch.load(MODEL_PATH, weights_only=False, map_location="cpu").eval()
    harness.write_reference(
        REFERENCE_PATH,
        snapshot(model),
        provenance={
            "source": "tests/golden/models/tiny_dielectric.model",
            "recipe": "tests/golden/build_dielectric_anchor.py",
            "description": "Seeded legacy AtomicDielectricMACE: scalar, vector and tensor readouts and their position Jacobians on the three H/C/O molecular fixtures.",
            "evaluated_with": "AtomicDielectricMACE.forward, e3nn, CPU, float64",
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    print(f"  anchor: {MODEL_PATH}\n  reference: {REFERENCE_PATH}")
