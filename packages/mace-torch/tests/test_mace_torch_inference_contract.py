"""Prepared inference changes execution state, never canonical model state."""

from dataclasses import replace

import pytest
import torch
from conftest import assert_close
from mace_core.config.precision import PrecisionConfig
from mace_torch.backends.composite import CompositeBackend
from mace_torch.backends.inference import (
    InferencePreparation,
    PreparedInference,
    prepare_inference,
)
from mace_torch.backends.inference_conformance import run_inference_conformance
from mace_torch.backends.precision import PrecisionBackend
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import SpanCapabilities
from mace_torch.serialization import (
    CheckpointError,
    canonical_state,
    load_checkpoint,
    save_checkpoint,
)
from mace_torch_inference_fixtures import (
    CompositionBackend,
    composition,
    inputs,
    interaction_spec,
    product_spec,
    region_weights,
)
from torch import nn


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("readout", [False, True])
def test_product_snapshot_parity_and_canonical_roundtrip(
    layout, residual, readout, tmp_path
):
    spec = product_spec(layout, residual, readout)
    source = composition(spec)
    original = {
        path: {name: value.clone() for name, value in state.items()}
        for path, state in canonical_state(source).items()
    }
    save_checkpoint(tmp_path / "before", source, {})
    backend = CompositionBackend()
    prepared = prepare_inference(backend, spec, region_weights(source))
    assert prepared is not None
    run_inference_conformance(
        prepared,
        source,
        [inputs(spec, 3), inputs(spec, 5)],
        precision=spec.linear.precision,
    )
    assert backend.calls == 1
    assert not list(prepared.parameters())
    assert prepared.state_dict() == {}
    save_checkpoint(tmp_path / "after", source, {})
    # Compare tensors and metadata exactly, independent of header key order.
    restored = composition(spec)
    load_checkpoint(tmp_path / "after.json", restored)
    for path, state in canonical_state(restored).items():
        for name, value in state.items():
            assert torch.equal(value, original[path][name])
    assert (tmp_path / "before.json").read_bytes() == (
        tmp_path / "after.json"
    ).read_bytes()
    for actual, expected in zip(
        prepared(inputs(spec)), restored(inputs(spec)), strict=True
    ):
        assert_close(actual, expected.detach())
    with pytest.raises(ValueError, match="cannot train"):
        prepared.train()
    with pytest.raises(ValueError, match="reprepare"):
        prepared.to(torch.float32)
    with pytest.raises(CheckpointError, match="source model"):
        save_checkpoint(tmp_path / "invalid", prepared, {})


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize(
    "residual,density,cutoff",
    [
        (False, False, True),
        (True, False, True),
        (False, True, False),
        (True, True, False),
    ],
)
def test_full_interaction_live_geometry_and_input_adjoints(
    layout, residual, density, cutoff
):
    spec = interaction_spec(layout, residual, density, cutoff)
    source = composition(spec)
    prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))
    assert prepared is not None
    cases = [inputs(spec, 3), inputs(spec, 5)]
    changed = list(cases[0])
    changed[1] = changed[1] * 7  # leave the cutoff without changing the edge list
    cases.append(tuple(changed))
    empty = list(cases[0])
    empty[1], empty[5], empty[6] = empty[1][:0], empty[5][:0], empty[6][:0]
    cases.append(tuple(empty))
    run_inference_conformance(
        prepared, source, cases, precision=spec.linear_up.precision
    )


def test_snapshot_is_independent_and_repreparation_uses_new_weights():
    spec = product_spec()
    source = composition(spec)
    backend = CompositionBackend()
    old = prepare_inference(backend, spec, region_weights(source))
    arguments = inputs(spec)
    before = old(arguments)
    with torch.no_grad():
        source.linear.weight.add_(0.8)
    for actual, expected in zip(old(arguments), before, strict=True):
        assert torch.equal(actual, expected)
    new = prepare_inference(backend, spec, region_weights(source))
    assert backend.calls == 2
    assert not torch.equal(new(arguments)[0], before[0])
    run_inference_conformance(new, source, [arguments], precision=spec.linear.precision)
    # The canonical model still trains with its original parameter identities.
    source.train()
    source(arguments)[0].square().sum().backward()
    assert source.linear.weight.grad is not None


def test_terminal_readout_can_omit_unused_features():
    spec = replace(product_spec(), return_features=False)
    source = composition(spec)
    prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))
    assert prepared.spec.output_names == ("readout",)
    run_inference_conformance(
        prepared, source, [inputs(spec)], precision=spec.interface_dtype
    )
    with pytest.raises(ValueError, match="requires a readout"):
        replace(spec, readout=None)


@pytest.mark.parametrize("transform", ["agnesi", "soft"])
def test_complete_region_preserves_distance_transform_state(transform):
    spec = interaction_spec(transform=transform)
    source = composition(spec)
    prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))
    assert prepared is not None
    run_inference_conformance(
        prepared,
        source,
        [inputs(spec, 3), inputs(spec, 5)],
        precision=spec.interface_dtype,
    )


def test_optional_extension_and_decline_do_not_change_existing_backends():
    spec = product_spec()
    source = composition(spec)
    weights = region_weights(source)
    assert prepare_inference(ReferenceBackend(), spec, weights) is None

    class Declining:
        def prepare_inference(self, spec, weights):
            return None

    assert prepare_inference(Declining(), spec, weights) is None
    with pytest.raises(ValueError, match="exactly"):
        prepare_inference(Declining(), spec, {**weights, "typo": {}})


def test_composite_and_precision_wrappers_preserve_the_optional_factory():
    spec = product_spec()
    source = composition(spec)
    primary = CompositionBackend()
    backend = CompositeBackend(
        PrecisionBackend(primary, PrecisionConfig()), ReferenceBackend()
    )
    prepared = prepare_inference(backend, spec, region_weights(source))
    assert prepared is not None and primary.calls == 1
    assert prepared.spec is spec
    run_inference_conformance(
        prepared, source, [inputs(spec)], precision=spec.interface_dtype
    )
    assert (
        prepare_inference(
            CompositeBackend(ReferenceBackend(), ReferenceBackend()),
            spec,
            region_weights(source),
        )
        is None
    )


def test_preparer_cannot_mutate_the_canonical_source():
    spec = product_spec()
    source = composition(spec)
    original = source.linear.weight.clone()

    class Mutating:
        def prepare_inference(self, spec, weights):
            weights["linear"][""]["weight"].zero_()
            return None

    assert prepare_inference(Mutating(), spec, region_weights(source)) is None
    assert torch.equal(source.linear.weight, original)


def test_preparation_in_inference_mode_preserves_later_input_derivatives():
    spec = product_spec()
    source = composition(spec)
    with torch.inference_mode():
        prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))
    assert not any(tensor.is_inference() for tensor in prepared.buffers())
    run_inference_conformance(
        prepared, source, [inputs(spec)], precision=spec.interface_dtype
    )


@pytest.mark.parametrize(
    "bad", ["buffer", "parameter", "weight_gradients", "return_type"]
)
def test_invalid_preparation_is_an_error(bad):
    spec = product_spec()
    source = composition(spec)

    class Invalid(CompositionBackend):
        def prepare_inference(self, spec, weights):
            result = super().prepare_inference(spec, weights)
            if bad == "return_type":
                return result.operation
            if bad == "buffer":
                result.operation.register_buffer("hidden", torch.ones(1))
            if bad == "parameter":
                result.operation.register_parameter(
                    "hidden", nn.Parameter(torch.ones(1))
                )
            if bad == "weight_gradients":
                result.operation.capabilities = SpanCapabilities(True, True)
            return result

    with pytest.raises((TypeError, ValueError)):
        prepare_inference(Invalid(), spec, region_weights(source))


def test_input_gradients_are_required_even_for_inference_only():
    spec = product_spec()
    source = composition(spec)
    prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))

    class Detached(nn.Module):
        capabilities = SpanCapabilities(False, False)

        def forward(self, arguments, tensors):
            return tuple(
                value.detach() for value in prepared.operation(arguments, tensors)
            )

    broken = PreparedInference(
        spec, InferencePreparation(Detached(), tuple(prepared.buffers()))
    )
    with pytest.raises(RuntimeError, match="does not require grad"):
        run_inference_conformance(
            broken, source, [inputs(spec)], precision=spec.linear.precision
        )


@pytest.mark.parametrize("make_spec", [product_spec, interaction_spec])
def test_prepared_region_compiles_and_exports_with_live_inputs(tmp_path, make_spec):
    spec = make_spec()
    source = composition(spec)
    prepared = prepare_inference(CompositionBackend(), spec, region_weights(source))
    compiled = torch.compile(prepared, backend="eager", fullgraph=True)
    for count in (3, 5):
        arguments = inputs(spec, count)
        for actual, expected in zip(
            compiled(arguments), source(arguments), strict=True
        ):
            assert_close(actual, expected.detach())
    arguments = inputs(spec, 3)
    program = torch.export.export(prepared, (arguments,), strict=True)
    path = tmp_path / "prepared.pt2"
    torch.export.save(program, path)
    deployed = torch.export.load(path).module()
    moved = tuple(
        tensor * 1.1 if tensor.is_floating_point() else tensor for tensor in arguments
    )
    for actual, expected in zip(deployed(moved), source(moved), strict=True):
        assert_close(actual, expected.detach())
