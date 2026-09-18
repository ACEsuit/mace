"""The LAMMPS export goldens: the libtorch numbers and the ML-IAP interface."""

from __future__ import annotations

from pathlib import Path

from tests.golden import harness
from tests.golden.paths import REPO_ROOT
from tests.golden.targets import anchors as anchors_target

ORDER = 60
HELP = "the LAMMPS export goldens (libtorch numbers, ML-IAP interface)"


def run() -> None:
    """The LAMMPS export goldens: the libtorch numbers and the ML-IAP interface.

    This is the only place the export golden's *input* is built, and the only
    place in the golden machinery that reaches into the package's neighbour
    list -- see tests/integrations/lammps/export_golden.py for why the input
    is committed rather than rebuilt at test time.
    """
    import json  # pylint: disable=import-outside-toplevel
    import shutil  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel
    import sys as _sys  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    import torch  # pylint: disable=import-outside-toplevel

    from tests.integrations.lammps.export_golden import (  # pylint: disable=import-outside-toplevel
        FIXTURE,
        N_REPEAT,
        build_input,
        mliap_interface,
        replay,
    )

    anchor = anchors_target.checkpoint("tiny_scaleshift")
    with tempfile.TemporaryDirectory() as work:
        work_path = Path(work)
        model_copy = work_path / anchor.name
        shutil.copy(anchor, model_copy)
        create = REPO_ROOT / "mace" / "cli" / "create_lammps_model.py"
        for extra, suffix in ((["--format=mliap"], "-mliap_lammps.pt"), ([], "-lammps.pt")):
            subprocess.run(
                [_sys.executable, str(create), *extra, str(model_copy)],
                check=True,
                cwd=work,
            )
            assert (work_path / (model_copy.name + suffix)).exists()

        torch.set_default_dtype(torch.float64)
        model = torch.load(
            anchor, map_location="cpu", weights_only=False
        ).to(torch.float64)
        atoms = harness.load_fixtures([FIXTURE])[FIXTURE]
        recorded = build_input(model, atoms)
        outputs = replay(work_path / (model_copy.name + "-lammps.pt"), recorded)

        libtorch = {
            "schema_version": harness.SCHEMA_VERSION,
            "dtype": "float64",
            "device": "cpu",
            "backend": "e3nn",
            "units": {"length": "Ang", "energy": "eV"},
            "provenance": {
                "source": f"tests/golden/models/{anchor.name}",
                "recipe": "tests/golden/regenerate.py --target lammps",
                "description": (
                    "mace_create_lammps_model (libtorch format) on the "
                    "ScaleShiftMACE anchor, evaluated on an open cluster of "
                    f"{N_REPEAT}^3 replicas of the {FIXTURE} fixture with the "
                    "central replica as the LOCAL atoms."
                ),
                "fixture": FIXTURE,
                "n_repeat": N_REPEAT,
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            "input": recorded,
            "outputs": {
                key: value.tolist() for key, value in outputs.items()
            },
        }
        path = harness.REFERENCES_DIR / "lammps_export_libtorch_fp64.json"
        # Compact rather than indented: this reference is mostly a few
        # thousand edge indices, and one number per line would triple a file
        # nobody reads by eye.
        path.write_text(json.dumps(libtorch, sort_keys=True) + "\n")
        print(f"  reference lammps libtorch -> {path.relative_to(REPO_ROOT)}")

        interface = {
            "schema_version": harness.SCHEMA_VERSION,
            "provenance": {
                "source": f"tests/golden/models/{anchor.name}",
                "recipe": "tests/golden/regenerate.py --target lammps",
                "description": (
                    "What LAMMPS reads off the ML-IAP artefact before it "
                    "calls the model. The numerics are deliberately not "
                    "pinned -- see tests/integrations/lammps/"
                    "export_golden.py."
                ),
            },
            "interface": mliap_interface(
                work_path / (model_copy.name + "-mliap_lammps.pt")
            ),
        }
        path = harness.REFERENCES_DIR / "lammps_export_mliap_interface.json"
        path.write_text(json.dumps(interface, indent=2, sort_keys=True) + "\n")
        print(f"  reference lammps mliap    -> {path.relative_to(REPO_ROOT)}")
