"""The converted foundation models on a GPU.

Two claims, both about what changes when the same model moves to an
accelerator:

* **The committed goldens still hold**, at each golden's own row. A CPU
  reference crosses a device here, which is what the rows are for.
* **Float32 is reproducible when the sums are.** Every reduction in the model
  is an ``index_add``, which on a GPU accumulates with atomics, so the order
  of a float32 sum and with it the last bits change from run to run. With
  deterministic algorithms enforced the same evaluation twice is identical
  bit for bit, and the atomic path stays inside the float32 row of it.

Measured on MACE-MP-0 small on an A100, in units of float32 epsilon times the
quantity's largest component: 0 between two deterministic runs, up to 5
between two atomic ones, 13 between the A100 and a Xeon, 26 between the A100
and an Apple CPU. A frozen float32 golden at machine epsilon therefore
reproduces only on the device and torch build that wrote it, so the
reproducibility claim is made within one process instead, where it holds on
any accelerator.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest
import torch

from tests.golden import foundation_artifacts as fa
from tests.golden.harness import load_fixtures, tolerance
from tests.parity.test_foundation_roster import (
    FLOAT32_PRECISION,
    GOLDEN_PARAMS,
    GOLDENS,
    artifact_path,
    assert_reproduces_golden,
    converted,
)
from tests.parity.test_foundation_roster import v1_outputs as outputs_on

# cuBLAS is deterministic only with a fixed workspace, and reads the setting
# when its handle is created, so it has to be in the environment before the
# first CUDA call of the session. Collection imports this before any test runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

pytestmark = pytest.mark.gpu

EXACT = tolerance("exact")
FLOAT32 = tolerance("fp32")

#: The artifacts whose float32 reproducibility is gated: the first instance
#: the conversion was validated on, and the default model, which needs no
#: download.
REPRODUCIBLE = [
    pytest.param("mp_small", marks=pytest.mark.network, id="mp_small"),
    pytest.param("mpa0_medium", id="mpa0_medium"),
]


@pytest.mark.parametrize("golden", GOLDEN_PARAMS)
def test_the_conversion_reproduces_the_committed_golden_on_gpu(golden, fp64):
    assert_reproduces_golden(golden, "cuda")


@contextlib.contextmanager
def deterministic(enabled: bool):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def float32_outputs(name: str, enabled: bool) -> dict[tuple[str, str], np.ndarray]:
    """Every fixture the golden pins, evaluated in float32 on the GPU."""
    spec = fa.ARTIFACTS[name]
    imported, _ = converted(artifact_path(GOLDENS[name]), FLOAT32_PRECISION)
    imported.engine.to("cuda")
    found = {}
    try:
        with deterministic(enabled):
            for fixture, atoms in load_fixtures(names=list(spec.fixture_names)).items():
                for quantity, value in outputs_on(imported, atoms, 0, "cuda").items():
                    found[(fixture, quantity)] = value
    finally:
        imported.engine.to("cpu")
    return found


def exceeded(got, expected, row) -> list[str]:
    """Each quantity outside ``row``, with its largest difference."""
    found = []
    for (fixture, quantity), value in expected.items():
        if not np.allclose(got[(fixture, quantity)], value, atol=row.atol, rtol=row.rtol):
            difference = float(np.abs(got[(fixture, quantity)] - value).max())
            found.append(f"{fixture} {quantity}: {difference:.3g}")
    return found


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_deterministic_float32_reproduces_bit_for_bit(name):
    first = float32_outputs(name, enabled=True)
    second = float32_outputs(name, enabled=True)
    assert not exceeded(second, first, EXACT), exceeded(second, first, EXACT)


@pytest.mark.parametrize("name", REPRODUCIBLE)
def test_atomic_float32_sums_stay_inside_the_float32_row(name):
    reference = float32_outputs(name, enabled=True)
    atomic = float32_outputs(name, enabled=False)
    assert not exceeded(atomic, reference, FLOAT32), exceeded(
        atomic, reference, FLOAT32
    )
