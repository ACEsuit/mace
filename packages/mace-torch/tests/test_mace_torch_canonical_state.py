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


class _Scaled(torch.nn.Module):
    """A block that saves its own scalar and leaves its op to save its own."""

    def __init__(self, op: torch.nn.Module) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.op = op

    def to_canonical(self) -> dict[str, torch.Tensor]:
        return {"scale": self.scale.detach()}

    def load_canonical(self, state) -> None:
        with torch.no_grad():
            self.scale.copy_(state["scale"])


def test_a_tensor_under_a_saving_module_is_refused_when_its_own_does_not_save():
    """A backend's linear op without the canonical pair, inside a block that
    saves only its own scalar, as the convolution saves its neighbour count."""
    model = torch.nn.ModuleDict({"block": _Scaled(torch.nn.Linear(2, 2, bias=False))})
    with pytest.raises(ValueError, match=re.escape("['block.op.weight']")):
        canonical_state(model)


def test_a_tensor_in_a_parameter_list_is_its_holders():
    """The reference ops keep their weights in parameter lists."""

    class Op(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weights = torch.nn.ParameterList([torch.ones(2), torch.ones(3)])

        def to_canonical(self) -> dict[str, torch.Tensor]:
            return {f"w{i}": w.detach() for i, w in enumerate(self.weights)}

        def load_canonical(self, state) -> None:
            pass

    state = canonical_state(torch.nn.ModuleDict({"block": _Scaled(Op())}))
    assert set(state) == {"block", "block.op"}
    assert set(state["block.op"]) == {"w0", "w1"}


def test_a_module_built_on_trainable_constants_is_saved():
    class Basis(TrainableConstants):
        def __init__(self) -> None:
            super().__init__()
            self.register_constant("frequencies", torch.ones(4), trainable=True)

    state = canonical_state(torch.nn.ModuleDict({"radial": Basis()}))
    assert set(state) == {"radial"}
    assert torch.equal(state["radial"]["frequencies"], torch.ones(4))
