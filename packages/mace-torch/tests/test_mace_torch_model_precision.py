"""A model's constants follow its precision, not the process that built it.

The radial basis, the cutoff envelope and the repulsion make their constants
in the process default at construction. A fresh interpreter's default is
float32, so a float64 model built there held float32 constants and computed
different numbers from the same model built after a caller had changed the
default, which is how one checkpoint gave two answers.
"""

from __future__ import annotations

import pytest
import torch
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch_engine_fixtures import ATOMIC_NUMBERS, ENERGY


def build(default: torch.dtype) -> MACEModel:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(default)
    try:
        head = EnergyOutputHead(
            ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
            ["default"],
            AtomicNumberTable(ATOMIC_NUMBERS),
            ScaleShiftSpec("std", (1.0,), (0.0,)),
            PrecisionConfig(),
        )
        return MACEModel(
            ReferenceBackend(),
            atomic_numbers=ATOMIC_NUMBERS,
            observables=[ENERGY],
            energy_head=head,
            num_features=4,
            lmax=1,
            cutoff=3.5,
            pair_repulsion=True,
            cutoff_order=5,
            precision="float64",
        )
    finally:
        torch.set_default_dtype(previous)


def floating_buffers(model: MACEModel) -> dict[str, torch.Tensor]:
    return {
        name: buffer
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point()
    }


@pytest.mark.parametrize("default", [torch.float32, torch.float64])
def test_a_float64_model_holds_float64_constants_whatever_the_default(default):
    narrow = {
        name: buffer.dtype
        for name, buffer in floating_buffers(build(default)).items()
        if buffer.dtype != torch.float64
    }
    assert not narrow, narrow


def test_the_constants_are_the_same_numbers_from_either_default():
    """Not only the dtype: built under float32 and cast afterwards, a constant
    would be float64 and still rounded."""
    from_narrow = floating_buffers(build(torch.float32))
    from_wide = floating_buffers(build(torch.float64))
    assert from_narrow.keys() == from_wide.keys()
    for name, value in from_narrow.items():
        assert torch.equal(value, from_wide[name]), name


def test_building_leaves_the_process_default_alone():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        build(torch.float32)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(previous)
