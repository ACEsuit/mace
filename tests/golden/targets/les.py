"""The MACELES anchor and its two references, model and calculator.

Not part of ``--target all``: it needs the optional ``les`` package at the
commit ``requirements/les.txt`` pins, and the numbers it writes are a
property of that library as much as of this repository. Folding it into
``all`` would mean a regeneration on a machine without ``les`` either crashes
halfway through the others or, worse, silently leaves the LES references
describing an older solver than the sidecar claims.
"""

from __future__ import annotations

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

ORDER = 38

#: The manifest is shared with every other family, so a bare load_fixtures()
#: picks up whatever the next one adds. This anchor is built on H/C/O and
#: cannot evaluate an iron structure at all.
ANCHOR_ELEMENTS = (1, 6, 8)
HELP = "the MACELES anchor and its model and calculator references"
IN_ALL = False


def run() -> None:
    """The MACELES anchor and its two references, model and calculator.

    Both files record the ``les`` commit they were taken with, and this
    refuses to write either one when that commit cannot be established. A
    reference whose provenance says "unknown" is the exact artifact the two
    xfails in ``tests/extensions/les/test_maceles.py`` are made of: numbers
    nobody can reproduce, and no way to tell whether the model or the solver
    moved.
    """
    import torch  # pylint: disable=import-outside-toplevel

    from tests.golden import maceles_surfaces  # pylint: disable=import-outside-toplevel
    from tests.golden.build_maceles_anchor import (  # pylint: disable=import-outside-toplevel
        LES_ARGUMENTS_PATH,
        MODEL_PATH,
        build_anchor,
        load_les_arguments,
    )
    from tests.golden.les_pin import (  # pylint: disable=import-outside-toplevel
        describe_les,
        installed_les_commit,
    )

    commit = installed_les_commit()
    if commit is None:
        raise SystemExit(
            "refusing to regenerate the LES goldens: the installed les "
            f"records no VCS provenance ({describe_les()}). Install it the "
            "way CI does -- pip install -r requirements/les.txt -- so the "
            "reference can record which solver produced its numbers."
        )

    path = build_anchor()
    print(f"  anchor   tiny_maceles     -> {path.relative_to(REPO_ROOT)}")

    model = torch.load(MODEL_PATH, weights_only=False, map_location="cpu").to(
        torch.float64
    )
    les_arguments = load_les_arguments()
    provenance_base = {
        "source": f"tests/golden/models/{MODEL_PATH.name}",
        "recipe": "tests/golden/build_maceles_anchor.py",
        "les_commit": commit,
        "les_arguments_file": (
            f"tests/golden/models/{LES_ARGUMENTS_PATH.name}"
        ),
        "tolerance_row": harness.FP64_CPU_REFERENCE.name,
    }

    snapshot = harness.snapshot_outputs(
        maceles_surfaces.ModelSurface(model),
        harness.load_fixtures(elements=ANCHOR_ELEMENTS),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={
            "model_class": type(model).__name__,
            "les_commit": commit,
            "les_arguments": les_arguments,
        },
    )
    written = harness.write_reference(
        harness.REFERENCES_DIR / "tiny_maceles_e3nn_cpu_fp64.json",
        snapshot,
        provenance={
            **provenance_base,
            "description": (
                "Tiny MACELES anchor through its forward: the long-range "
                "energy, all five latent quantities and the Born effective "
                "charges, none of which the ase calculator exposes in full."
            ),
            "evaluated_with": (
                "MACELES.forward via tests/golden/maceles_surfaces.ModelSurface, "
                "e3nn, CPU, float64"
            ),
        },
        allow_overwrite=True,
    )
    print(f"  reference tiny_maceles     -> {written.relative_to(REPO_ROOT)}")

    field_snapshot = harness.snapshot_outputs(
        maceles_surfaces.field_calculator(model),
        harness.load_fixtures(list(maceles_surfaces.FIELD_FIXTURES)),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={
            "model_class": type(model).__name__,
            "les_commit": commit,
            "les_arguments": les_arguments,
            # eps_infty, keep_neutral and electric_field_unit are evaluation
            # configuration rather than inputs to the graph, so they travel
            # here; external_field is both, and the harness records it as an
            # input channel as well.
            "field_settings": maceles_surfaces.FIELD_SETTINGS,
        },
    )
    written = harness.write_reference(
        harness.REFERENCES_DIR / "tiny_maceles_field_cpu_fp64.json",
        field_snapshot,
        provenance={
            **provenance_base,
            "description": (
                "The same anchor through MACECalculator with an external "
                "field, eps_infty and keep_neutral: the field reaches the "
                "Ewald sum through the batch, and the Born charges are turned "
                "into a force correction outside the model."
            ),
            "evaluated_with": (
                "mace.calculators.MACECalculator(compute_bec=True, "
                "external_field=..., eps_infty=..., keep_neutral=...), e3nn, "
                "CPU, float64"
            ),
            "field_settings": maceles_surfaces.FIELD_SETTINGS,
        },
        allow_overwrite=True,
    )
    print(f"  reference tiny_maceles_field -> {written.relative_to(REPO_ROOT)}")
