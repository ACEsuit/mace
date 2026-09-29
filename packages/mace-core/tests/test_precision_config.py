"""The precision of each dispatched op, as the configuration answers it."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest
from mace_core.config.precision import OP_KINDS, PrecisionConfig, PrecisionPolicy
from mace_core.kernels import descriptors
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import widest

ALL_DESCRIPTORS = [
    cls
    for _, cls in inspect.getmembers(descriptors, inspect.isclass)
    if issubclass(cls, Descriptor) and cls is not Descriptor
]


def test_the_presets():
    assert PrecisionConfig.float64() == PrecisionConfig()
    assert PrecisionConfig.float64().interface_dtype == "float64"
    assert PrecisionConfig.float32().interface_dtype == "float32"
    assert PrecisionConfig.float32().accumulate == "float64"
    mixed = PrecisionConfig.mixed()
    assert mixed.interface_dtype == "float32"
    assert mixed.policy_for(LinearDescriptor()) == PrecisionPolicy.uniform("float32")
    assert mixed.policy_for(RadialBasisDescriptor()) == PrecisionPolicy.uniform(
        "float64"
    )
    assert mixed.policy_for(SymmetricContractionDescriptor()) == PrecisionPolicy(
        "float32", "float64", "float32"
    )


@pytest.mark.parametrize("descriptor", ALL_DESCRIPTORS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("preset", ["float64", "float32"])
def test_a_uniform_preset_gives_every_op_the_interface_dtype(descriptor, preset):
    config = getattr(PrecisionConfig, preset)()
    assert config.policy_for(descriptor()) == PrecisionPolicy.uniform(preset)


@pytest.mark.parametrize("descriptor", ALL_DESCRIPTORS, ids=lambda c: c.__name__)
def test_every_descriptor_has_an_op_kind(descriptor):
    """A descriptor without one would take the default silently, so a
    configuration naming its kind could never reach it."""
    policy = PrecisionPolicy.uniform("float64")
    matched = [
        kind
        for kind in OP_KINDS
        if PrecisionConfig(model="float32", ops=((kind, policy),)).policy_for(
            descriptor()
        )
        == policy
    ]
    assert len(matched) == 1, matched


def test_a_span_takes_the_widest_of_its_members_in_each_field():
    config = PrecisionConfig(
        model="float32",
        ops=(
            ("linear", PrecisionPolicy("float32", "float32", "float32")),
            (
                "channelwise_tp_conv",
                PrecisionPolicy("float32", "float64", "float32"),
            ),
            ("segment_reduce", PrecisionPolicy("float64", "float64", "float64")),
        ),
    )
    span = (LinearDescriptor(), ChannelwiseTPConvDescriptor())
    assert config.policy_for(span) == PrecisionPolicy("float32", "float64", "float32")
    wider = (*span, SegmentReduceDescriptor())
    assert config.policy_for(wider) == PrecisionPolicy.uniform("float64")


def test_a_span_of_one_is_its_member():
    config = PrecisionConfig.mixed()
    member = SymmetricContractionDescriptor()
    assert config.policy_for([member]) == config.policy_for(member)


def test_an_empty_span_is_refused():
    with pytest.raises(ValueError, match="at least one op"):
        PrecisionConfig().policy_for(())


def test_a_floor_narrower_than_the_compute_dtype_is_refused():
    with pytest.raises(ValueError, match="can only widen"):
        PrecisionPolicy("float64", "float32", "float64")


def test_an_unknown_op_kind_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="symmetric_contraction"):
        PrecisionConfig(ops=(("contraction", PrecisionPolicy.uniform("float64")),))


def test_a_device_with_float64_changes_nothing():
    config = PrecisionConfig.mixed()
    assert config.for_device(True) == (config, [])


def test_a_float64_floor_degrades_to_the_model_dtype_and_says_so():
    """What the frozen tree's ``safe_double`` does on a device without
    float64: fall back to the tensor's own dtype rather than raise."""
    resolved, degraded = PrecisionConfig.mixed().for_device(False)
    assert resolved.accumulate == "float32"
    assert resolved.resolved_accumulate(False) == "float32"
    for descriptor in ALL_DESCRIPTORS:
        assert resolved.policy_for(descriptor()) == PrecisionPolicy.uniform("float32")
    assert "the energy reduction" in degraded
    assert "radial_basis compute" in degraded
    assert "symmetric_contraction accumulate" in degraded


def test_a_float32_run_reports_only_its_energy_reduction():
    resolved, degraded = PrecisionConfig.float32().for_device(False)
    assert resolved == PrecisionConfig(model="float32", accumulate="float32")
    assert degraded == ["the energy reduction"]


def test_widest():
    assert widest("float32", "bfloat16") == "float32"
    assert widest("float32", "float64", "bfloat16") == "float64"


def test_the_module_imports_no_framework():
    probe = (
        "import sys, mace_core.config.precision; "
        "print(sorted(m for m in ('torch', 'jax') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"
