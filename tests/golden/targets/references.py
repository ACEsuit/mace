"""The snapshot JSONs for the two tiny anchors.

Only for those two. Other families snapshot their own checkpoints, because
the calculator arguments and the channels worth recording differ per model
class; this target would otherwise evaluate, say, a dipole model as if it
were an energy model and record a reference that pins the wrong thing.
"""

from __future__ import annotations

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

# Imported as a module, not as names: ``all_anchors()`` reads ``ANCHORS`` off
# every target module, so binding that name here would make this module look
# like a second owner of the two anchors it merely reads.
from tests.golden.targets import anchors as anchors_target

ORDER = 30
HELP = "the reference JSONs for the tiny anchors"


def run() -> None:
    # Late import: see the package docstring.
    import torch  # pylint: disable=import-outside-toplevel

    from mace.calculators import (  # pylint: disable=import-outside-toplevel
        MACECalculator,
    )

    fixtures = harness.load_fixtures()
    for name, (_, reference_name, description) in anchors_target.ANCHORS.items():
        model_path = anchors_target.checkpoint(name)
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
                "recipe": anchors_target.RECIPES[name],
                "description": description,
                "evaluated_with": "mace.calculators.MACECalculator, e3nn, CPU, float64",
                "tolerance_row": harness.FP64_CPU_REFERENCE.name,
            },
            allow_overwrite=True,
        )
        print(f"  reference {name:15s} -> {path.relative_to(REPO_ROOT)}")
