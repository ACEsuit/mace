"""The saved-state walker and the base class of trainable constants.

The walker's round trip through a whole backbone is in the backbone's suite;
here are what holds for any module tree.
"""

import re

import pytest
import torch
from mace_torch.kernels import TrainableConstants, canonical_state
from mace_torch.nn.radial import ZBLBasis


@pytest.mark.parametrize("trainable", [False, True])
def test_trainable_constants_are_saved_only_when_learned(trainable):
    repulsion = ZBLBasis(trainable=trainable)
    state = repulsion.to_canonical()
    assert set(state) == (
        {"screening_length_exponent", "screening_length_prefactor"}
        if trainable
        else set()
    )


def test_trainable_constants_load_what_they_saved():
    repulsion = ZBLBasis(trainable=True)
    moved = {name: value + 0.1 for name, value in repulsion.to_canonical().items()}
    repulsion.load_canonical(moved)
    for name, value in repulsion.to_canonical().items():
        assert torch.equal(value, moved[name])


def test_a_tensor_no_module_would_save_is_refused():
    """A backend's own trainable basis without the canonical pair, for one."""

    class Basis(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frequencies = torch.nn.Parameter(torch.ones(4))

    with pytest.raises(ValueError, match=re.escape("['radial.frequencies']")):
        canonical_state(torch.nn.ModuleDict({"radial": Basis()}))


def test_a_module_built_on_trainable_constants_is_saved():
    class Basis(TrainableConstants):
        def __init__(self) -> None:
            super().__init__()
            self.register_constant("frequencies", torch.ones(4), trainable=True)

    state = canonical_state(torch.nn.ModuleDict({"radial": Basis()}))
    assert set(state) == {"radial"}
    assert torch.equal(state["radial"]["frequencies"], torch.ones(4))
