"""A committed fp32 reduction-order gate; never recapture during tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.parity.fp32_reductions import assert_golden, evaluate, sequential_reference

pytestmark = [pytest.mark.nightly, pytest.mark.slow]
REFERENCE = (
    Path(__file__).resolve().parents[1] / "golden/references/fp32_reductions.json"
)


@pytest.fixture
def golden():
    return json.loads(REFERENCE.read_text())


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def test_reference_backend_reproduces_frozen_fp32_reductions(golden):
    assert_golden(evaluate(golden["inputs"]), golden["outputs"])


def test_golden_agrees_with_independent_sequential_float32(golden):
    for name, expected in sequential_reference(golden["inputs"]).items():
        np.testing.assert_array_equal(expected, golden["outputs"][name])


def test_machine_epsilon_gate_rejects_reordered_accumulation(golden):
    changed = evaluate(golden["inputs"], reverse=True)
    # Both reductions individually must be sensitive; one failure cannot hide
    # a convolution fixture that never exercised accumulation order.
    for name in changed:
        outputs = evaluate(golden["inputs"])
        outputs[name] = changed[name]
        with pytest.raises(AssertionError, match=name):
            assert_golden(outputs, golden["outputs"])


def test_fixed_order_matches_the_frozen_cpu_scatter(isolated, golden):
    from mace.tools.scatter import scatter_sum

    inputs = golden["inputs"]
    values = torch.tensor(inputs["values"], dtype=torch.float32)
    receiver = torch.tensor(inputs["receiver"], dtype=torch.int64)
    expected = scatter_sum(values, receiver, dim=0, dim_size=inputs["num_nodes"])
    assert torch.equal(evaluate(inputs)["segment_reduce"], expected)
