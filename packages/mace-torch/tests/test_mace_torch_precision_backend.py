"""Ops built at the precision the configuration gives each one."""

from __future__ import annotations

import pickle
from dataclasses import replace

import pytest
import torch
from conftest import fp64_only
from mace_core.config.precision import PrecisionConfig, PrecisionPolicy
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.canonical import KERNEL_SPEC_VERSION
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends.composite import CompositeBackend
from mace_torch.backends.precision import (
    PrecisionBackend,
    boundary_precision,
    interface_dtype,
)
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models.build import build_model
from mace_torch_engine_fixtures import ATOMIC_NUMBERS, build_graph, molecule
from torch import nn

CONFIG = ResolvedConfig.model_validate(
    {
        "model": {
            "model": "scale_shift",
            "observables": ["energy"],
            "r_max": 5.0,
            "num_channels": 4,
            "max_ell": 1,
            "hidden_irreps": "0e+1o",
        }
    }
)


def engine(precision: PrecisionConfig):
    built, _ = build_model(
        CONFIG.model_copy(update={"precision": precision}),
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(ATOMIC_NUMBERS),
        heads=("default",),
        e0s=ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        statistics=DatasetStatistics(avg_num_neighbors=3.0),
    )
    return built


def graph(dtype: torch.dtype) -> dict:
    return {
        key: value.to(dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in build_graph(*molecule()).items()
    }


def cast_ops(model: nn.Module) -> dict[str, tuple[str, str]]:
    return {
        path: cast
        for path, module in model.named_modules()
        if (cast := boundary_precision(module)) is not None
    }


@pytest.mark.parametrize("preset", ["float64", "float32"])
def test_a_uniform_preset_casts_nothing(preset):
    model = engine(getattr(PrecisionConfig, preset)())
    assert interface_dtype(model) == getattr(torch, preset)
    assert cast_ops(model) == {}


def test_mixed_casts_exactly_the_ops_it_holds_wider():
    ops = cast_ops(engine(PrecisionConfig.mixed()))
    assert ops == {
        "backbone.backbone.radial_embedding": ("float64", "float32"),
        "backbone.backbone.radial_embedding.radial": ("float64", "float32"),
        "backbone.backbone.products.0.contraction": ("float64", "float32"),
        "backbone.backbone.products.1.contraction": ("float64", "float32"),
    }


def test_mixed_hands_out_the_interface_dtype():
    out = engine(PrecisionConfig.mixed())(graph(torch.float32), compute=("forces",))
    assert out.total_energy.dtype is torch.float32
    assert out.forces.dtype is torch.float32


def test_a_checkpoint_does_not_depend_on_the_preset():
    """Same names, and the same draw rounded to the narrower dtype: the casts
    live on the op, so they add no module to the tree and no path to seed."""
    narrow = engine(PrecisionConfig.float32()).state_dict()
    mixed = engine(PrecisionConfig.mixed()).state_dict()
    assert list(narrow) == list(mixed)
    for name, tensor in narrow.items():
        torch.testing.assert_close(
            mixed[name].to(tensor.dtype), tensor, rtol=0, atol=0, msg=name
        )


def test_mixed_agrees_with_float32_to_its_tolerance():
    positions = graph(torch.float32)
    narrow = engine(PrecisionConfig.float32())(positions, compute=("forces",))
    mixed = engine(PrecisionConfig.mixed())(positions, compute=("forces",))
    torch.testing.assert_close(mixed.total_energy, narrow.total_energy)
    torch.testing.assert_close(mixed.forces, narrow.forces, rtol=1e-3, atol=5e-5)


@fp64_only
def test_the_casts_are_differentiable_twice():
    """Training on forces takes a derivative of the forces."""
    model = engine(PrecisionConfig.mixed())
    model.train()
    out = model(graph(torch.float32), compute=("forces",), training=True)
    parameter = next(p for p in model.parameters() if p.dtype is torch.float64)
    (gradient,) = torch.autograd.grad(out.forces.square().sum(), parameter)
    assert gradient.dtype is torch.float64
    assert torch.count_nonzero(gradient) > 0


def test_a_casting_model_pickles():
    pickle.loads(pickle.dumps(engine(PrecisionConfig.mixed())))


class Recording:
    """A backend that keeps what it was asked for."""

    name = "recording"
    solver: object | None = None

    def __init__(self, wide_accumulation: bool = False) -> None:
        self.wide_accumulation = wide_accumulation
        self.seen: list = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            wide_accumulation=self.wide_accumulation, spec_version=KERNEL_SPEC_VERSION
        )

    def make_linear(self, descriptor):
        self.seen.append(descriptor)
        return nn.Identity()

    def make_symmetric_contraction(self, descriptor):
        self.seen.append(descriptor)
        return nn.Identity()

    def make_interaction_layer(self, descriptors):
        self.seen.extend(descriptors)
        return nn.Identity()


FLOOR = PrecisionConfig(
    model="float32",
    ops=(("symmetric_contraction", PrecisionPolicy("float32", "float64", "float32")),),
)


def test_a_backend_without_its_own_accumulator_computes_in_the_floor():
    inner = Recording()
    PrecisionBackend(inner, FLOOR).make_symmetric_contraction(
        SymmetricContractionDescriptor()
    )
    assert (inner.seen[0].precision, inner.seen[0].accumulate) == ("float64", None)


def test_a_backend_with_its_own_accumulator_is_handed_both():
    inner = Recording(wide_accumulation=True)
    op = PrecisionBackend(inner, FLOOR).make_symmetric_contraction(
        SymmetricContractionDescriptor()
    )
    assert (inner.seen[0].precision, inner.seen[0].accumulate) == (
        "float32",
        "float64",
    )
    assert boundary_precision(op) is None


def test_an_op_at_the_interface_dtype_is_stamped_and_left_alone():
    inner = Recording()
    op = PrecisionBackend(inner, FLOOR).make_linear(
        LinearDescriptor(precision="float64")
    )
    assert inner.seen[0].precision == "float32"
    assert boundary_precision(op) is None


def test_a_span_is_stamped_with_one_policy():
    inner = Recording()
    span = PrecisionBackend(inner, FLOOR).make_interaction_layer(
        (LinearDescriptor(), SymmetricContractionDescriptor())
    )
    assert {member.precision for member in inner.seen} == {"float64"}
    assert span is not None
    assert boundary_precision(span) == ("float64", "float32")


def test_a_backend_without_spans_declines_one():
    assert (
        PrecisionBackend(ReferenceBackend(), FLOOR).make_interaction_layer(
            (ChannelwiseTPConvDescriptor(),)
        )
        is None
    )


def test_weights_held_apart_from_the_compute_dtype_are_refused():
    config = PrecisionConfig(
        model="float32",
        ops=(("linear", PrecisionPolicy("float32", "float32", "float64")),),
    )
    with pytest.raises(ValueError, match="param_dtype equal to compute_dtype"):
        PrecisionBackend(Recording(), config).make_linear(LinearDescriptor())


def test_without_float64_the_wider_ops_degrade_and_the_report_names_them():
    backend = PrecisionBackend(Recording(), PrecisionConfig.mixed(), False)
    op = backend.make_symmetric_contraction(SymmetricContractionDescriptor())
    assert boundary_precision(op) is None
    report = backend.build_report()
    assert report is not None
    assert "symmetric_contraction accumulate" in report
    assert PrecisionBackend(Recording(), PrecisionConfig.mixed()).build_report() is None


def test_tf32_degradation_is_recorded_without_global_math_switches(caplog):
    inner = Recording()
    backend = PrecisionBackend(inner, PrecisionConfig.tf32())
    previous = torch.get_float32_matmul_precision()
    backend.make_linear(LinearDescriptor())
    backend.make_linear(LinearDescriptor())
    assert inner.seen[0].product == "ieee" and inner.seen[0].precision == "float32"
    assert backend.degraded == ["LinearDescriptor tf32 products"]
    report = backend.build_report()
    assert report is not None and "LinearDescriptor tf32 products" in report
    assert "uses IEEE products" in caplog.text
    assert torch.get_float32_matmul_precision() == previous


def test_dispatch_checks_the_effective_precision_and_product():
    class TensorCoreRecording(Recording):
        def capabilities(self):
            return replace(
                super().capabilities(),
                ops=frozenset({"linear"}),
                dtypes=frozenset({"float32"}),
                products=frozenset({"ieee", "tf32"}),
            )

    inner = TensorCoreRecording()
    primary = PrecisionBackend(inner, PrecisionConfig.tf32())
    fallback = PrecisionBackend(ReferenceBackend(), PrecisionConfig.tf32())
    backend = CompositeBackend(primary, fallback)
    backend.make_linear(LinearDescriptor())
    assert inner.seen[0].product == "tf32"
    assert inner.seen[0].precision == "float32"
    assert backend.decisions[-1].descriptor == inner.seen[0]
    assert backend.decisions[-1].backend == inner.name
    assert primary.build_report() is None
    assert backend.capabilities().products == frozenset({"ieee", "tf32"})


def test_widening_the_compute_dtype_also_resolves_tf32():
    policy = PrecisionConfig(
        model="float32",
        ops=(("linear", PrecisionPolicy("float32", "float64", "float32", "tf32")),),
    )
    inner = Recording()
    backend = PrecisionBackend(inner, policy)
    backend.make_linear(LinearDescriptor())
    assert inner.seen[0].precision == "float64"
    assert inner.seen[0].product == "ieee"
    report = backend.build_report()
    assert report is not None and "tf32 products" in report


def test_what_it_does_not_stamp_is_the_wrapped_backends():
    inner = Recording()
    inner.solver = object()
    assert PrecisionBackend(inner, FLOOR).solver is inner.solver
    assert PrecisionBackend(inner, FLOOR).name == "recording"
