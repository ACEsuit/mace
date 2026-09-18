"""The magnetic anchor and its two references.

Not part of ``--target all``: regenerating these needs the ``magnetic``
extra (sphericart). Naming the target without it stops with that sentence
rather than with an ImportError from three frames down.
"""

from __future__ import annotations

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

ORDER = 37
HELP = "the magnetic anchor and its two references"
IN_ALL = False


#: The magnetic reference files, and what each one is for.
MAGNETIC_REFERENCES = {
    "tiny_magnetic": (
        "tiny_magnetic_e3nn_cpu_fp64.json",
        "Tiny MagneticScaleShiftMACE: energy, the per-atom energies, forces "
        "and magforces (dE/dm) on the five moment-carrying fixtures. The "
        "moments are recorded as inputs and compared exactly, because a "
        "derivative with respect to an input is only meaningful at the value "
        "it was taken at.",
    ),
    "tiny_magnetic_scf": (
        "tiny_magnetic_scf_e3nn_cpu_fp64.json",
        "The same model wrapped in MagneticSCFMACE: the converged fixed "
        "point. equilibrated_magmom, the energy and the forces are pinned; "
        "scf_steps and scf_energy_history are recorded as metadata, since "
        "they describe how LBFGS got there rather than where it is.",
    ),
}


def run() -> None:
    """The magnetic anchor and its two references.

    Imported inside the function, not at module scope: the targets package is
    imported to build ``--help``, and a module-scope import of the builder
    would make every other target refuse to run on a machine with no
    sphericart.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from tests.golden import magnetic_surfaces
        from tests.golden.build_magnetic_anchor import MODEL_PATH, build_anchor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "the magnetic goldens need the `magnetic` extra (sphericart): "
            f"{exc}. Install it, or run --target fixtures/anchors/references."
        ) from exc

    path = build_anchor()
    print(f"  anchor   tiny_magnetic    -> {path.relative_to(REPO_ROOT)}")

    sources = {
        "tiny_magnetic": (
            magnetic_surfaces.MagneticForward(),
            magnetic_surfaces.magnetic_fixtures(),
        ),
        "tiny_magnetic_scf": (
            magnetic_surfaces.MagneticSCFForward(),
            # A subset, for the measured reason in magnetic_surfaces.py.
            magnetic_surfaces.scf_fixtures(),
        ),
    }
    for name, (reference_name, description) in MAGNETIC_REFERENCES.items():
        source, fixtures = sources[name]
        metadata: dict = {"model_class": "MagneticScaleShiftMACE"}
        if name == "tiny_magnetic_scf":
            metadata = {
                "model_class": "MagneticSCFMACE",
                "scf_config": dict(magnetic_surfaces.SCF_CONFIG),
            }
        snapshot = harness.snapshot_outputs(
            source,
            fixtures,
            dtype="float64",
            device="cpu",
            backend="e3nn",
            metadata=metadata,
        )
        written = harness.write_reference(
            harness.REFERENCES_DIR / reference_name,
            snapshot,
            provenance={
                "source": f"tests/golden/models/{MODEL_PATH.name}",
                "recipe": "tests/golden/build_magnetic_anchor.py",
                "description": description,
                "evaluated_with": (
                    "tests/golden/magnetic_surfaces.py, the model forward, "
                    "e3nn, CPU, float64"
                ),
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
                "requires": "the `magnetic` extra (sphericart)",
            },
            allow_overwrite=True,
        )
        print(f"  reference {name:15s} -> {written.relative_to(REPO_ROOT)}")
