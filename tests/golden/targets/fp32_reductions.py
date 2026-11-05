"""Initial capture of PAR-2's deterministic fp32 reductions, never an overwrite."""

ORDER = 75
HELP = "capture the fixed-order fp32 reduction fixture (refuses existing files)"
IN_ALL = False


def run() -> None:
    import json
    import platform
    from pathlib import Path

    import numpy as np
    import torch

    from tests.parity.fp32_reductions import evaluate, fixture, sequential_reference

    root = Path(__file__).resolve().parents[1] / "references"
    reference = root / "fp32_reductions.json"
    sidecar = root / "fp32_reductions.capture.json"
    if reference.exists() or sidecar.exists():
        raise FileExistsError("PAR-2 initial capture refuses to overwrite its frozen reference")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inputs = fixture()
    outputs = evaluate(inputs)
    independent = sequential_reference(inputs)
    for name, tensor in outputs.items():
        np.testing.assert_array_equal(tensor.detach().numpy(), independent[name])
    reference.write_text(json.dumps({
        "schema_version": 1, "dtype": "float32", "device": "cpu",
        "inputs": inputs,
        "outputs": {name: value.tolist() for name, value in outputs.items()},
    }, indent=2) + "\n")
    sidecar.write_text(json.dumps({
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/golden/regenerate.py --target fp32-reductions --i-know-what-i-am-doing",
        "seed": inputs["seed"], "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "threads": 1, "deterministic_algorithms": True,
        "scope": "CPU segment_reduce and scalar-channel channelwise_tp_conv accumulation order",
        "max_abs_error_limit": float(np.finfo(np.float32).eps),
        "independent_oracle": "sequential IEEE float32 addition of stored dyadic inputs",
    }, indent=2) + "\n")
