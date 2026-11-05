"""First periodic stress references for the existing OFF23-medium artifact."""

import hashlib

from tests.golden import foundation_artifacts as fa
from tests.golden import harness

ORDER = 44
HELP = "OFF23 medium on the three existing periodic fixtures (download/cache)"
IN_ALL = False

FIXTURES = ("slab_vacuum", "slab_zero_vacuum", "triclinic_bulk")
REFERENCE = "off_medium_periodic_e3nn_cpu_fp64.json"


def run():
    import torch

    torch.set_default_dtype(torch.float64)
    spec = fa.ARTIFACTS["off_medium"]
    calculator, source = fa.load_calculator(spec)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == spec.sha256, "OFF23 medium source artifact changed"
    snapshot = harness.snapshot_outputs(
        calculator, harness.load_fixtures(names=FIXTURES),
        dtype="float64", device="cpu", backend="e3nn",
    )
    harness.write_reference(
        harness.REFERENCES_DIR / REFERENCE, snapshot,
        provenance={
            "source": spec.origin,
            "sha256": digest,
            "recipe": "python tests/golden/regenerate.py --target off-medium-periodic --i-know-what-i-am-doing",
            "loader": spec.loader,
            "loader_kwargs": spec.loader_kwargs,
            "fixture_names": list(FIXTURES),
            "evaluated_with": "frozen legacy calculator, e3nn, CPU, float64",
            "description": "First periodic stress capture for JAX-1; existing molecular references unchanged.",
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
