"""The v1 stack on the same inference cases the legacy baselines were taken on.

``test_inference_cpu.py`` measures the frozen tree on fixed cases so that "the
rewrite is not slower" can be judged against an old number. This is the other
half: the same committed anchor, converted into a v1 checkpoint, on the same
diamond supercells at the same three sizes and two precisions, evaluated
through the v1 model on each kernel backend. The nightly ``benchmarks`` job
runs both files in one session and ``compare_v1.py`` pairs them in its summary.

The backends pair with the frozen tree's as ``reference`` with ``e3nn`` (both
the plain-torch path on the CPU), ``cueq`` with ``cueq`` and ``oeq`` with
``oeq``. The GPU cases run in no CI job, for the reason the legacy file gives;
they are the recipe for a quiet GPU host:

    pytest tests/benchmarks/test_inference_v1.py -m benchmark -p no:randomly \\
        --benchmark-json=benchmark.json

Nothing here gates. The foundation model is not converted here: that needs the
network and the production converter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import build

from tests.benchmarks.test_inference_cpu import (
    DTYPES,
    SYSTEM_SIZES,
    _device_label,
    _record_throughput,
)

#: The v1 kernel backend each legacy backend is compared with.
LEGACY_COUNTERPART = {"reference": "e3nn", "cueq": "cueq", "oeq": "oeq"}


def backend_params() -> list:
    return [
        pytest.param("reference", id="reference"),
        pytest.param("cueq", marks=[pytest.mark.gpu, pytest.mark.cueq], id="cueq"),
        pytest.param("oeq", marks=[pytest.mark.gpu, pytest.mark.oeq], id="oeq"),
    ]


def _v1_anchor(directory: Path, backend: str, dtype: str, device: str):
    """The committed anchor as a v1 checkpoint that records ``backend``, read
    back the way a user's model is."""
    from mace_core.kernels.precision import PrecisionConfig
    from mace_torch.deploy.loader import load_deployed

    from tests.parity.test_anchor_as_foundation import write_anchor_checkpoint
    from tests.parity.test_fm00_training_step import load_anchor

    checkpoint = Path(
        write_anchor_checkpoint(load_anchor("tiny_scaleshift.model"), directory)
    )
    sidecar = checkpoint.with_suffix(".json")
    document = json.loads(sidecar.read_text())
    document["config"]["config"]["resolved"]["model"]["backend"] = backend
    sidecar.write_text(json.dumps(document))
    return load_deployed(
        checkpoint, device=device, precision=PrecisionConfig(model=dtype)
    )


def _graph(deployed, repeat: int, dtype: str, device: str) -> dict:
    from mace_core.data.configuration import Configuration
    from mace_torch.data import collate_training
    from mace_torch.data.batch import GraphDataset

    atoms = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((repeat,) * 3)
    item = Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        cell=np.asarray(atoms.cell),
        pbc=(True, True, True),
    )
    dataset = GraphDataset(
        [item], cutoff=deployed.r_max, z_table=deployed.z_table, targets=()
    )
    collated = collate_training(
        [dataset[0]], z_table=deployed.z_table, float_dtype=dtype
    )
    return dict(collated.to(device).graph)


@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=5)
@pytest.mark.parametrize("regime", list(SYSTEM_SIZES))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", backend_params())
def test_v1_inference_latency(
    benchmark, tmp_path, backend: str, dtype: str, regime: str
):
    """One energy and forces evaluation at a fixed size, on the v1 stack."""
    device = "cpu" if backend == "reference" else "cuda"
    repeat, expected_atoms = SYSTEM_SIZES[regime]
    previous = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, dtype))
    try:
        deployed = _v1_anchor(tmp_path, backend, dtype, device)
        graph = _graph(deployed, repeat, dtype, device)
        num_atoms = int(graph["positions"].shape[0])
        assert num_atoms == expected_atoms

        benchmark.extra_info.update(
            stack="v1",
            model="anchor",
            backend=backend,
            legacy_backend=LEGACY_COUNTERPART[backend],
            regime=regime,
            num_atoms=num_atoms,
            num_edges=int(graph["edge_index"].shape[1]),
            dtype=dtype,
            device=device,
            device_name=_device_label(device),
            torch_version=torch.__version__,
            r_max=float(deployed.r_max),
        )
        engine = deployed.engine

        def evaluate():
            if device == "cuda":
                torch.cuda.synchronize()
            engine(dict(graph), compute=("forces",), training=False)
            if device == "cuda":
                torch.cuda.synchronize()

        benchmark(evaluate)
        _record_throughput(benchmark, num_atoms)
    finally:
        torch.set_default_dtype(previous)
