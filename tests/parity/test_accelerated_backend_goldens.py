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

import hashlib
import json
from pathlib import Path

import pytest

from tests.golden import harness
from tests.parity.anchors import load_anchor
from tests.parity.test_anchor_as_foundation import ANCHOR, write_anchor_checkpoint

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
    checkpoint = Path(write_anchor_checkpoint(legacy, tmp_path))
    weights = checkpoint.with_suffix(".safetensors")
    original_weights = hashlib.sha256(weights.read_bytes()).hexdigest()
    checkpoint = on_backend(checkpoint, backend)
    calculator = MACECalculator(
        model_paths=checkpoint, device=device, enable_cueq=False, enable_oeq=False
    )
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
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == original_weights
    if backend == "cueq":
        from mace_torch.backends.cueq import compiled_operations_available

        assert compiled_operations_available(), (
            "cuEquivariance ran its pure-torch path, so this matched without "
            "running a vendor kernel"
        )


@pytest.mark.parametrize("backend,device,row", CASES)
def test_eval_uses_recorded_backend_and_preserves_canonical_weights(
    fp64, tmp_path, monkeypatch, backend, device, row
):
    import numpy as np
    from ase.io import read, write
    from ase.stress import full_3x3_to_voigt_6_stress
    from mace_torch.calculators import MACECalculator
    from mace_torch.cli.eval_configs import main
    from mace_torch.deploy import loader

    legacy = load_anchor(ANCHOR)
    checkpoint = on_backend(Path(write_anchor_checkpoint(legacy, tmp_path)), backend)
    weights = checkpoint.with_suffix(".safetensors")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    frames = list(
        harness.load_fixtures(elements=[int(z) for z in legacy.atomic_numbers]).values()
    )
    source, output = tmp_path / "input.xyz", tmp_path / "output.xyz"
    write(source, frames)
    arguments = [
        "--configs",
        str(source),
        "--model",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        device,
        "--compute_stress",
        "--batch_size",
        "2",
    ]
    with pytest.raises(ValueError, match=r"enable_cueq.*checkpoint"):
        main([*arguments, "--enable_cueq"])
    assert not output.exists()
    for flag in ("enable_cueq", "enable_oeq"):
        with pytest.raises(TypeError, match=flag + ".*model.backend"):
            MACECalculator(checkpoint, **{flag: True})

    original_load = loader.load_deployed
    calls, hooks = {}, []

    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        assert model.config.model.backend == backend
        assert next(model.engine.parameters()).device.type == device
        for name, module in model.engine.named_modules():
            if backend in OPS and type(module).__name__.startswith(OPS[backend]):
                calls[name] = 0

                def count(_module, _inputs, _output, name=name):
                    calls[name] += 1

                hooks.append(module.register_forward_hook(count))
        return model

    monkeypatch.setattr(loader, "load_deployed", load)
    try:
        assert main(arguments) == 0
    finally:
        for hook in hooks:
            hook.remove()
    if backend in OPS:
        assert calls and all(calls.values()), calls
    calculator = MACECalculator(checkpoint, device=device)
    results = read(output, index=":")
    assert len(results) == len(frames)
    tolerance = harness.tolerance(row)
    for expected_atoms, actual in zip(frames, results, strict=True):
        calculator.calculate(expected_atoms)
        for name in ("energy", "forces", "stress"):
            written = (actual.arrays if name == "forces" else actual.info)[
                "MACE_" + name
            ]
            if name == "stress":
                written = full_3x3_to_voigt_6_stress(written)
            np.testing.assert_allclose(
                written,
                calculator.results[name],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                err_msg=name,
            )
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
