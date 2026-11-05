"""First references for the fifteen published artifacts not previously pinned."""

import hashlib

from tests.golden import foundation_artifacts as fa
from tests.golden import harness
from tests.golden.targets.foundation_references import MDP_FIXTURES, POLAR_FIXTURES

ORDER = 43
HELP = "references for the remaining published foundation roster (downloads)"
IN_ALL = False

# Explicit identities: extending a loader's table must not silently expand a
# frozen reference family. Existing references belong to their original targets.
MODELS = {
    **{
        f"mp-{name}": ("mace_mp", name, "mace_mp_urls")
        for name in (
            "medium",
            "large",
            "small-0b",
            "medium-0b",
            "small-0b2",
            "medium-0b2",
            "large-0b2",
            "medium-0b3",
            "small-omat-0",
            "medium-omat-0",
            "mace-matpes-pbe-0",
            "mace-matpes-r2scan-0",
        )
    },
    "off-large": ("mace_off", "large", "mace_off_urls"),
    "polar-1-m": ("mace_polar", "polar-1-m", "polar_model_urls"),
    "polar-1-l": ("mace_polar", "polar-1-l", "polar_model_urls"),
}


def reference_path(name):
    return harness.REFERENCES_DIR / f"roster_{name.replace('-', '_')}_cpu_fp64.json"


def specification(name):
    from mace.calculators import foundations_models

    loader, alias, table = MODELS[name]
    return fa.FoundationArtifact(
        name=name,
        loader=loader,
        loader_kwargs={"model": alias, "device": "cpu", "default_dtype": "float64"},
        reference=reference_path(name).name,
        sha256="",  # The first capture records the actual source bytes below.
        origin=getattr(foundations_models, table)[alias],
        network=True,
        fixture_tags=(),
        fixture_names=MDP_FIXTURES if loader == "mace_off" else POLAR_FIXTURES,
        description=f"Published {loader}({alias!r}), full calculator surface.",
        url_table=(table, alias),
    )


def run():
    for name in MODELS:
        spec = specification(name)
        calculator, source = fa.load_calculator(spec)
        snapshot = harness.snapshot_outputs(
            calculator,
            harness.load_fixtures(names=spec.fixture_names),
            dtype="float64",
            device="cpu",
            backend="e3nn",
        )
        harness.write_reference(
            reference_path(name),
            snapshot,
            provenance={
                "source": spec.origin,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "recipe": "python tests/golden/regenerate.py --target foundation-roster --i-know-what-i-am-doing",
                "loader": spec.loader,
                "loader_kwargs": spec.loader_kwargs,
                "description": spec.description,
                "fixture_names": list(spec.fixture_names),
                "evaluated_with": "legacy MACECalculator, e3nn, CPU, float64",
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            allow_overwrite=True,
        )
        print(f"  reference {name}: {reference_path(name)}", flush=True)
