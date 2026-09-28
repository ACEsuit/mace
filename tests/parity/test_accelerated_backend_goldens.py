"""The committed anchor reference, reproduced by v1 on each kernel backend.

The same comparison the frozen tree's accelerated backends are held to
(``tests/golden/test_backend_parity_golden.py``): a committed checkpoint,
evaluated on the accelerator, against the numbers a CPU e3nn run committed to
this repository, at the ``fp64_accelerated_backend`` row. The route is the
user's: the anchor becomes a v1 checkpoint whose record names the backend, and
the v1 calculator reads it and evaluates the golden fixtures.

A value comparison cannot tell whether a vendor kernel ran, since the
reference backend reproduces the reference too. So every case also counts the
calls into the backend's own ops, and each has to have been called.

The reference backend runs the same comparison on the CPU at the
``fp64_cpu_reference`` row, which is what keeps the path itself honest on a
host with no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden import harness
from tests.parity.test_anchor_as_foundation import ANCHOR, write_anchor_checkpoint
from tests.parity.test_fm00_training_step import load_anchor

REFERENCE = harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"

CASES = [
    pytest.param("reference", "cpu", harness.FP64_CPU_REFERENCE.name, id="reference"),
    pytest.param(
        "cueq",
        "cuda",
        harness.FP64_ACCELERATED_BACKEND.name,
        marks=[pytest.mark.gpu, pytest.mark.cueq],
        id="cueq",
    ),
    pytest.param(
        "oeq",
        "cuda",
        harness.FP64_ACCELERATED_BACKEND.name,
        marks=[pytest.mark.gpu, pytest.mark.oeq],
        id="oeq",
    ),
]

#: The class-name prefix of each backend's own ops.
OPS = {"cueq": "CuEq", "oeq": "Oeq"}


def on_backend(checkpoint: Path, backend: str) -> Path:
    """The same checkpoint, recording that it runs on ``backend``.

    The weights are canonical, so they are the same file whichever backend
    reads them; only the record's backend name changes.
    """
    sidecar = checkpoint.with_suffix(".json")
    document = json.loads(sidecar.read_text())
    document["config"]["config"]["resolved"]["model"]["backend"] = backend
    sidecar.write_text(json.dumps(document))
    return checkpoint


@pytest.mark.parametrize("backend, device, row", CASES)
def test_the_committed_cpu_reference_is_reproduced(
    fp64, tmp_path, backend, device, row
):
    from mace_torch.calculators import MACECalculator

    legacy = load_anchor(ANCHOR)
    checkpoint = on_backend(Path(write_anchor_checkpoint(legacy, tmp_path)), backend)
    calculator = MACECalculator(model_paths=checkpoint, device=device)
    engine = calculator.models[0].engine
    assert next(engine.parameters()).device.type == device

    calls: dict[str, int] = {}
    hooks = []
    if backend in OPS:
        for name, module in engine.named_modules():
            if type(module).__name__.startswith(OPS[backend]):
                calls[name] = 0

                def count(_module, _inputs, _output, name=name):
                    calls[name] += 1

                hooks.append(module.register_forward_hook(count))
        assert calls, f"no {backend} op was built into the model"
    try:
        snapshot = harness.snapshot_outputs(
            calculator,
            harness.load_fixtures(elements=[int(z) for z in legacy.atomic_numbers]),
            dtype="float64",
            device=device,
            backend=backend,
            metadata={"route": "v1 checkpoint through the v1 calculator"},
        )
    finally:
        for hook in hooks:
            hook.remove()
    idle = sorted(name for name, count in calls.items() if count == 0)
    assert not idle, f"{backend} ops that never ran: {idle}"

    harness.compare_to_reference(snapshot, harness.load_reference(REFERENCE), row=row)
    if backend == "cueq":
        from mace_torch.backends.cueq import compiled_operations_available

        assert compiled_operations_available(), (
            "cuEquivariance ran its pure-torch path, so this matched without "
            "running a vendor kernel"
        )
