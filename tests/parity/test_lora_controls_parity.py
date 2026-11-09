"""LoRA factors compared in shared coordinates, including the legacy fan-in."""

import copy
import math

import pytest
import torch
from e3nn import o3
from mace_core.kernels.specs import LinearSpec
from mace_torch.backends.reference import ReferenceLinear
from mace_torch.finetune.lora import DenseLowRank, inject_lora, merge_lora
from torch.func import functional_call

from mace.modules.lora import LoRADenseLinear, LoRAO3Linear
from tests.golden.harness import CLOSED_FORM_FP64
from tests.parity.legacy_weights import linear_permutation

pytestmark = pytest.mark.usefixtures("isolated", "fp64")


def close(actual, expected):
    torch.testing.assert_close(
        actual, expected, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
    )


@pytest.mark.parametrize("rank,alpha", [(1, 0.0), (3, 2.5), (2, -0.5)])
@pytest.mark.parametrize("irreps_in", ["3x0e+2x1o", "2x0e+1x0e+2x1o"])
def test_equivariant_adapters_match_values_gradients_and_merge(rank, alpha, irreps_in):
    torch.manual_seed(71)
    irreps_out = "4x0e+3x1o"
    base = o3.Linear(irreps_in, irreps_out, biases=True)
    legacy = LoRAO3Linear(base, rank=rank, alpha=alpha)
    native = ReferenceLinear(
        LinearSpec(irreps_in=irreps_in, irreps_out=irreps_out, has_bias=True)
    )
    permutation = linear_permutation(base)
    with torch.no_grad():
        native.weight[permutation] = base.weight
        native.bias.copy_(base.bias)
    plain = copy.deepcopy(native)
    inject_lora(native, rank=rank, alpha=alpha)
    adapter = native.parametrizations.weight[0]
    first = [torch.randn_like(value, requires_grad=True) for value in adapter.first]
    second = [torch.randn_like(value, requires_grad=True) for value in adapter.second]
    features = torch.randn(5, base.irreps_in.dim, requires_grad=True)
    # A carries the same input fan-in as the base map. B's extra 1/sqrt(rank)
    # is absorbed into its canonical factor; the shared variables below make
    # gradient comparison independent of the two parameter storage orders.
    legacy_parameters = {
        "lora_A.weight": torch.cat([value.flatten() for value in first])[
            linear_permutation(legacy.lora_A)
        ],
        "lora_B.weight": math.sqrt(rank)
        * torch.cat([value.flatten() for value in second])[
            linear_permutation(legacy.lora_B)
        ],
    }
    native_parameters = {
        f"parametrizations.weight.0.{name}.{index}": value
        for name, values in (("first", first), ("second", second))
        for index, value in enumerate(values)
    }
    actual = functional_call(native, native_parameters, (features,))
    expected = functional_call(legacy, legacy_parameters, (features,))
    close(actual, expected)
    variables = (features, *first, *second)
    direction = torch.randn_like(actual)
    gradients = [
        torch.autograd.grad((value * direction).sum(), variables, create_graph=True)
        for value in (actual, expected)
    ]
    for left, right in zip(*gradients, strict=True):
        close(left, right)
    curvature = [
        torch.autograd.grad(sum(value.square().sum() for value in gradient), variables)
        for gradient in gradients
    ]
    for left, right in zip(*curvature, strict=True):
        close(left, right)
    with torch.no_grad():
        for name, value in native_parameters.items():
            native.get_parameter(name).copy_(value)
        for name, value in legacy_parameters.items():
            legacy.get_parameter(name).copy_(value)
    before = native(features).detach()
    if alpha:
        assert not torch.allclose(before, plain(features))
    else:
        close(before, plain(features))
    merge_lora(native)
    merged_legacy = legacy.merge_into_base()
    close(native(features), before)
    close(native(features), merged_legacy(features))
    assert not any("parametrizations" in name for name, _ in native.named_parameters())
    for matrix in (o3.rand_matrix(), -torch.eye(3)):
        close(
            native(features @ base.irreps_in.D_from_matrix(matrix).T),
            before @ base.irreps_out.D_from_matrix(matrix).T,
        )


@pytest.mark.parametrize("rank,alpha", [(1, 0.0), (3, 2.5), (2, -0.5)])
def test_dense_adapter_matches_legacy_values_and_factor_gradients(rank, alpha):
    torch.manual_seed(17)
    legacy = LoRADenseLinear(torch.nn.Linear(4, 5), rank=rank, alpha=alpha)
    weight = legacy.base.weight.detach().T.clone()
    native = DenseLowRank(weight, rank=rank, alpha=alpha)
    first = torch.randn(4, rank, requires_grad=True)
    second = torch.randn(rank, 5, requires_grad=True)
    features = torch.randn(7, 4, requires_grad=True)
    effective = functional_call(native, {"first": first, "second": second}, (weight,))
    actual = features @ effective + legacy.base.bias
    expected = functional_call(
        legacy, {"lora_A.weight": first.T, "lora_B.weight": second.T}, (features,)
    )
    close(actual, expected)
    gradients = [
        torch.autograd.grad(value.square().sum(), (features, first, second))
        for value in (actual, expected)
    ]
    for left, right in zip(*gradients, strict=True):
        close(left, right)


@pytest.mark.parametrize(
    "flags",
    [[], ["--foundation_model_readout"], ["--foundation_filter_elements", "False"]],
)
def test_readout_transfer_flag_matches_legacy_function_and_forces(tmp_path, flags):
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.cli.run_train import parse
    from mace_torch.finetune.foundation import read_foundation
    from mace_torch.finetune.transfer import transfer_foundation

    from mace.tools import AtomicNumberTable
    from mace.tools.finetuning_utils import load_foundations_elements
    from tests.parity.anchors import load_anchor
    from tests.parity.test_anchor_as_foundation import write_anchor_checkpoint
    from tests.parity.test_element_table_parity import (
        forces_and_energies,
        legacy_forces_and_energies,
        water_structures,
    )

    torch.manual_seed(11)
    source = load_anchor("tiny_scaleshift.model")
    target = copy.deepcopy(source)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.normal_(std=0.15)
    # Align the fresh draws before either transfer, so keeping a new readout
    # compares the same function rather than two initializers' random streams.
    source_native = read_foundation(
        write_anchor_checkpoint(source, tmp_path / "source"), DEFAULT_CATALOGUE
    )
    target_native = read_foundation(
        write_anchor_checkpoint(target, tmp_path / "target"), DEFAULT_CATALOGUE
    )
    enabled = not flags
    assert parse(flags).finetune.transfer_readout is enabled
    before = {key: value.clone() for key, value in target.readouts.state_dict().items()}
    numbers = source.atomic_numbers.tolist()
    load_foundations_elements(
        target,
        source,
        AtomicNumberTable(numbers),
        load_readout=enabled,
        use_shift=True,
        use_scale=True,
        max_L=1,
    )
    transfer_foundation(
        source_native.engine.model,
        target_native.engine.model,
        foundation_elements=numbers,
        elements=numbers,
        foundation_heads=("default",),
        heads=("default",),
        readout_from={"default": "default"},
        transfer_readout=enabled,
    )
    expected_readout = source.readouts.state_dict() if enabled else before
    for key, value in target.readouts.state_dict().items():
        torch.testing.assert_close(value, expected_readout[key], atol=0, rtol=0)
    structures = water_structures()
    for actual, expected in zip(
        forces_and_energies(target_native.engine, target, structures),
        legacy_forces_and_energies(target, structures),
        strict=True,
    ):
        close(actual, expected)
