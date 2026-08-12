"""Shared body of the two foundation targets.

Underscore-prefixed, so the package skips it during discovery: it is what
``foundations`` and ``foundations_network`` both do, not a target of its own.
The two differ only in whether running them downloads anything, and that
difference is the whole reason they are separate targets.
"""

from __future__ import annotations

from typing import Dict

from tests.golden import harness
from tests.golden.paths import REPO_ROOT


def _kwargs_repr(kwargs: Dict[str, object]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


def regenerate(network: bool) -> None:
    """Snapshot the published foundation checkpoints of one tier.

    The loader call, the fixture selection and the expected digest all come
    out of ``foundation_artifacts.ARTIFACTS``, which the tests read too: a
    reference generated with different arguments than the test replays is a
    reference that pins the arguments rather than the model.
    """
    from tests.golden import (  # pylint: disable=import-outside-toplevel
        foundation_artifacts as fa,
    )

    for spec in fa.ARTIFACTS.values():
        if spec.network is not network:
            continue
        calc, checkpoint = fa.load_calculator(spec)
        digest = fa.sha256_of(checkpoint)
        if digest != spec.sha256:
            raise SystemExit(
                f"{spec.name}: the artifact at {checkpoint} has digest "
                f"{digest}, the registry expects {spec.sha256}. Regenerating "
                f"against a different file than the one the registry names "
                f"would produce a reference nobody can reproduce; update "
                f"tests/golden/foundation_artifacts.py first, deliberately."
            )
        fixtures = harness.load_fixtures(tags=spec.fixture_tags or None)
        snapshot = harness.snapshot_outputs(
            calc,
            fixtures,
            dtype="float64",
            device="cpu",
            backend="e3nn",
            metadata={
                "model_class": type(calc.models[0]).__name__,
                "r_max": float(calc.models[0].r_max),
            },
        )
        provenance = {
            "source": spec.origin,
            "sha256": spec.sha256,
        }
        if spec.release_url:
            provenance["release_url"] = spec.release_url
        path = harness.write_reference(
            harness.REFERENCES_DIR / spec.reference,
            snapshot,
            provenance={
                **provenance,
                "recipe": (
                    "mace.calculators.foundations_models."
                    f"{spec.loader}({_kwargs_repr(spec.loader_kwargs)})"
                ),
                "description": spec.description,
                "fixture_tags": list(spec.fixture_tags),
                "evaluated_with": "mace.calculators.MACECalculator, e3nn, CPU, float64",
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            allow_overwrite=True,
        )
        print(f"  reference {spec.name:15s} -> {path.relative_to(REPO_ROOT)}")
