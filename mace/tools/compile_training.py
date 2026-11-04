"""Compiler-friendly conservative force and stress training.

The important transform in this module is ``make_fx``: it executes MACE's
internal energy derivative while tracing and records that derivative as ATen
operations.  AOTAutograd therefore sees a normal first-order backward through
the lowered graph instead of a runtime double-backward through
``torch.autograd.grad``.
"""

from __future__ import annotations

import contextlib
import difflib
import math
import types
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from e3nn import o3
from torch._decomp import core_aten_decompositions
from torch.func import functional_call
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx

_SUPPORTED_OUTPUT_FLAGS = frozenset(("energy", "forces", "virials", "stress"))

# Inductor releases before 2.6 cannot lower the floor-division shape
# cascades that dynamic-shape tracing of the force/stress path produces.
_MIN_TORCH_VERSION = (2, 6)
_TORCH_TOO_OLD_FOR_DYNAMIC_SHAPES = (
    tuple(int(part) for part in torch.__version__.split(".")[:2]) < _MIN_TORCH_VERSION
)
_NON_MODEL_INPUT_KEYS = frozenset(
    (
        "energy",
        "forces",
        "virials",
        "stress",
        "dipole",
        "polarizability",
        "magforces",
        "weight",
        "energy_weight",
        "forces_weight",
        "virials_weight",
        "stress_weight",
        "dipole_weight",
        "charges_weight",
        "polarizability_weight",
        "magforces_weight",
    )
)


def _stable_silu_backward(
    grad_output: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    sigmoid = torch.sigmoid(value)
    return grad_output * sigmoid * (1 + value * (1 - sigmoid))


def mace_decomp_table() -> dict[Any, Any]:
    """Return core ATen decompositions with a stable SiLU double derivative."""

    table = dict(core_aten_decompositions())
    table[torch.ops.aten.silu_backward.default] = _stable_silu_backward
    return table


@contextlib.contextmanager
def _fx_duck_shape(enabled: bool):
    config = torch.fx.experimental._config  # pylint: disable=protected-access
    previous = config.use_duck_shape
    config.use_duck_shape = enabled
    try:
        yield
    finally:
        config.use_duck_shape = previous


def _legacy_spherical_harmonics(
    lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Eager form of the e3nn 0.4.4 polynomial used by MACE models."""

    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack((sh_0_0,), dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack((sh_0_0, sh_1_0, sh_1_1, sh_1_2), dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
    if lmax == 2:
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
            ),
            dim=-1,
        )

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ),
            dim=-1,
        )
    sh_4_0 = 0.935414346693485 * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = (
        0.661437827766148 * sh_3_0 * y
        + 0.810092587300982 * sh_3_1 * z
        + 0.810092587300983 * sh_3_5 * x
    )
    sh_4_2 = (
        -0.176776695296637 * sh_3_0 * z
        + 0.866025403784439 * sh_3_1 * y
        + 0.684653196881458 * sh_3_2 * z
        + 0.684653196881457 * sh_3_4 * x
        + 0.176776695296637 * sh_3_6 * x
    )
    sh_4_3 = (
        -0.306186217847897 * sh_3_1 * z
        + 0.968245836551855 * sh_3_2 * y
        + 0.790569415042095 * sh_3_3 * x
        + 0.306186217847897 * sh_3_5 * x
    )
    sh_4_4 = -0.612372435695795 * (sh_3_2 * x + sh_3_4 * z) + sh_3_3 * y
    sh_4_5 = (
        -0.306186217847897 * sh_3_1 * x
        + 0.790569415042096 * sh_3_3 * z
        + 0.968245836551854 * sh_3_4 * y
        - 0.306186217847897 * sh_3_5 * z
    )
    sh_4_6 = (
        -0.176776695296637 * sh_3_0 * x
        - 0.684653196881457 * sh_3_2 * x
        + 0.684653196881457 * sh_3_4 * z
        + 0.866025403784439 * sh_3_5 * y
        - 0.176776695296637 * sh_3_6 * z
    )
    sh_4_7 = (
        -0.810092587300982 * sh_3_1 * x
        + 0.810092587300982 * sh_3_5 * z
        + 0.661437827766148 * sh_3_6 * y
    )
    sh_4_8 = 0.935414346693485 * (sh_3_6 * z - sh_3_0 * x)
    if lmax == 4:
        return torch.stack(
            (
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
            ),
            dim=-1,
        )
    raise RuntimeError(
        "--compile_training with e3nn 0.4.4 supports spherical harmonics through l=4"
    )


def _legacy_spherical_harmonics_forward(
    module: torch.nn.Module, vectors: torch.Tensor
) -> torch.Tensor:
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    harmonics = _legacy_spherical_harmonics(
        module._lmax,  # pylint: disable=protected-access
        vectors[..., 0],
        vectors[..., 1],
        vectors[..., 2],
    )
    chunks = tuple(
        harmonics[..., ell * ell : (ell + 1) * (ell + 1)]
        for ell in module._ls_list  # pylint: disable=protected-access
    )
    if module.normalization == "integral":
        chunks = tuple(
            chunk * (math.sqrt(2 * ell + 1) / math.sqrt(4 * math.pi))
            for ell, chunk in zip(
                module._ls_list, chunks  # pylint: disable=protected-access
            )
        )
    elif module.normalization == "component":
        chunks = tuple(
            chunk * math.sqrt(2 * ell + 1)
            for ell, chunk in zip(
                module._ls_list, chunks  # pylint: disable=protected-access
            )
        )
    return torch.cat(chunks, dim=-1)


@contextlib.contextmanager
def _traceable_legacy_spherical_harmonics(model: torch.nn.Module):
    """Temporarily bypass e3nn 0.4.4's shape-specializing ScriptFunction."""

    patched = []
    try:
        for module in model.modules():
            if not isinstance(module, o3.SphericalHarmonics):
                continue
            # e3nn 0.6 gives each module a selectable eager ``sph_func``. The
            # legacy release instead calls a module-global TorchScript function.
            if hasattr(module, "sph_func"):
                continue
            if getattr(module, "_lmax") > 4:
                raise RuntimeError(
                    "--compile_training with e3nn 0.4.4 supports spherical "
                    f"harmonics through l=4, but this model uses "
                    f"l={getattr(module, '_lmax')}"
                )
            had_instance_forward = "forward" in module.__dict__
            previous = module.__dict__.get("forward")
            module.forward = types.MethodType(  # type: ignore[method-assign]
                _legacy_spherical_harmonics_forward, module
            )
            patched.append((module, had_instance_forward, previous))
        yield
    finally:
        for module, had_instance_forward, previous in reversed(patched):
            if had_instance_forward:
                module.forward = previous
            else:
                del module.forward


def _requested_output_keys(output_args: Mapping[str, bool]) -> tuple[str, ...]:
    keys = ["energy"]
    keys.extend(key for key in ("forces", "virials", "stress") if output_args.get(key))
    return tuple(keys)


def _model_kwargs(output_args: Mapping[str, bool]) -> dict[str, bool]:
    compute_stress = bool(output_args.get("stress", False))
    return {
        "training": True,
        "compute_force": bool(output_args.get("forces", False)),
        "compute_virials": bool(output_args.get("virials", False) or compute_stress),
        "compute_stress": compute_stress,
    }


def _input_keys(batch: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key not in _NON_MODEL_INPUT_KEYS
        )
    )


def _prepare_batch(
    batch: Mapping[str, torch.Tensor], output_args: Mapping[str, bool]
) -> dict[str, torch.Tensor]:
    prepared = dict(batch)
    prepared["positions"] = batch["positions"].detach().clone().requires_grad_(True)
    if output_args.get("virials", False) or output_args.get("stress", False):
        num_graphs = batch["ptr"].numel() - 1
        displacement = batch.get("displacement")
        if displacement is None:
            displacement = torch.zeros(
                (num_graphs, 3, 3),
                dtype=prepared["positions"].dtype,
                device=prepared["positions"].device,
            )
        else:
            displacement = displacement.detach().clone()
        prepared["displacement"] = displacement.requires_grad_(True)
    return prepared


class _FlatFunctionalModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        parameter_names: Sequence[str],
        buffer_names: Sequence[str],
        output_args: Mapping[str, bool],
    ) -> None:
        super().__init__()
        self.model = model
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.parameter_names = tuple(parameter_names)
        self.buffer_names = tuple(buffer_names)
        self.model_kwargs = _model_kwargs(output_args)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        num_inputs = len(self.input_keys)
        num_parameters = len(self.parameter_names)
        data = dict(zip(self.input_keys, args[:num_inputs]))
        parameters = args[num_inputs : num_inputs + num_parameters]
        buffers = args[num_inputs + num_parameters :]
        state = dict(zip(self.parameter_names, parameters))
        state.update(zip(self.buffer_names, buffers))
        outputs = functional_call(
            self.model,
            state,
            args=(data,),
            kwargs=self.model_kwargs,
            # FX simplification can remove derived e3nn cache buffers while
            # preserving all optimizer-owned parameters. Extra live buffers
            # are harmless and must remain explicit for unsimplified models.
            strict=False,
        )
        return tuple(outputs[key] for key in self.output_keys)


def _trace_inputs(
    model: torch.nn.Module,
    example_batch: Mapping[str, torch.Tensor],
    output_args: Mapping[str, bool],
) -> tuple[
    _FlatFunctionalModel,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    list[torch.Tensor],
]:
    prepared = _prepare_batch(example_batch, output_args)
    input_keys = _input_keys(prepared)
    output_keys = _requested_output_keys(output_args)
    named_parameters = tuple(model.named_parameters())
    named_buffers = tuple(model.named_buffers())
    parameter_names = tuple(name for name, _ in named_parameters)
    buffer_names = tuple(name for name, _ in named_buffers)
    flat_model = _FlatFunctionalModel(
        model=model,
        input_keys=input_keys,
        output_keys=output_keys,
        parameter_names=parameter_names,
        buffer_names=buffer_names,
        output_args=output_args,
    )
    inputs = [prepared[key] for key in input_keys]
    inputs.extend(parameter for _, parameter in named_parameters)
    inputs.extend(buffer for _, buffer in named_buffers)
    return flat_model, input_keys, parameter_names, buffer_names, inputs


def _same_values_ignoring_layout(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.numel() != right.numel():
        return False
    left_values = torch.sort(left.detach().flatten().cpu()).values
    right_values = torch.sort(right.detach().flatten().cpu()).values
    return torch.equal(left_values, right_values)


def _architecture_source_shape_matches(
    fx_value: torch.Tensor,
    source_name: str,
    source_value: torch.Tensor,
) -> bool:
    if fx_value.dtype != source_value.dtype or fx_value.numel() != source_value.numel():
        return False
    if ".U_matrix_" not in source_name:
        return fx_value.shape == source_value.shape

    # e3nn code generation permutes U-matrix axes and inserts broadcast-only
    # singleton dimensions.  No other dimension change is an audited lowering.
    fx_dimensions = sorted(dimension for dimension in fx_value.shape if dimension != 1)
    source_dimensions = sorted(
        dimension for dimension in source_value.shape if dimension != 1
    )
    return fx_dimensions == source_dimensions


def _architecture_source_matches(
    fx_value: torch.Tensor,
    source_name: str,
    source_value: torch.Tensor,
) -> bool:
    return _architecture_source_shape_matches(
        fx_value, source_name, source_value
    ) and _same_values_ignoring_layout(fx_value, source_value)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left is right:
        return True
    if left.numel() == 0 or right.numel() == 0 or left.device != right.device:
        return False
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _common_path_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_part, right_part in zip(left.split("."), right.split(".")):
        if left_part != right_part:
            break
        length += 1
    return length


def _generated_graph_constant_source(trace_name: str) -> str | None:
    parent, separator, leaf = trace_name.rpartition(".")
    prefix = "_tensor_constant"
    index = leaf.removeprefix(prefix)
    if (
        not separator
        or not leaf.startswith(prefix)
        or not index.isdigit()
        or ".graphs." not in trace_name
        or ".graph." not in trace_name
    ):
        return None
    return f"{parent}.c{index}"


def _is_immutable_architecture_buffer(name: str) -> bool:
    """Identify generated algebra constants that cannot change during training."""

    return ".U_matrix_" in name or (".graphs." in name and ".graph.c" in name)


def _classify_architecture_constants(
    fx_model: GraphModule,
    model: torch.nn.Module,
    trace_model: torch.nn.Module | None = None,
) -> tuple[tuple[str, str], ...]:
    if tuple(fx_model.named_parameters()):
        raise RuntimeError("lowered training graph unexpectedly owns parameters")

    model_buffers = tuple(model.named_buffers())
    trace_buffers = (
        tuple(trace_model.named_buffers()) if trace_model is not None else ()
    )
    classifications = []
    for fx_name, fx_value in fx_model.named_buffers():
        immutable_matches = tuple(
            name
            for name, value in model_buffers
            if _is_immutable_architecture_buffer(name)
            if _architecture_source_matches(fx_value, name, value)
        )
        if not immutable_matches:
            raise RuntimeError(
                "lowered training graph owns an unaudited tensor constant "
                f"{fx_name!r} with shape={tuple(fx_value.shape)}, "
                f"dtype={fx_value.dtype}"
            )

        candidates = immutable_matches
        trace_name = None
        if trace_model is not None:
            trace_aliases = tuple(
                name
                for name, value in trace_buffers
                if _shares_storage(fx_value, value)
            )
            if len(trace_aliases) != 1:
                raise RuntimeError(
                    "lowered training graph tensor constant "
                    f"{fx_name!r} has ambiguous trace-buffer provenance="
                    f"{trace_aliases}"
                )
            trace_name = trace_aliases[0]
            generated_source = _generated_graph_constant_source(trace_name)
            if generated_source is not None:
                candidates = tuple(
                    name for name in immutable_matches if name == generated_source
                )
            else:
                prefix_lengths = {
                    name: _common_path_prefix_length(trace_name, name)
                    for name in immutable_matches
                }
                best_prefix = max(prefix_lengths.values())
                candidates = tuple(
                    name
                    for name, prefix_length in prefix_lengths.items()
                    if prefix_length == best_prefix
                )

        if len(candidates) != 1:
            raise RuntimeError(
                "lowered training graph tensor constant "
                f"{fx_name!r} has ambiguous immutable sources={candidates}"
            )
        source_name = candidates[0]

        mutable_matches = tuple(
            name
            for name, value in model_buffers
            if not _is_immutable_architecture_buffer(name)
            if _architecture_source_shape_matches(fx_value, source_name, value)
            if _same_values_ignoring_layout(fx_value, value)
        )
        if trace_name is None:
            plausible_mutable_matches = mutable_matches
        else:
            source_prefix = _common_path_prefix_length(trace_name, source_name)
            plausible_mutable_matches = tuple(
                name
                for name in mutable_matches
                if _common_path_prefix_length(trace_name, name) >= source_prefix
            )
        if plausible_mutable_matches:
            raise RuntimeError(
                "lowered training graph tensor constant "
                f"{fx_name!r} collides with mutable model buffers="
                f"{plausible_mutable_matches}"
            )

        classifications.append((fx_name, source_name))
    return tuple(classifications)


def _prune_unused_placeholders(
    fx_model: GraphModule,
    input_keys: Sequence[str],
    parameter_names: Sequence[str],
    buffer_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    names = tuple(input_keys) + tuple(parameter_names) + tuple(buffer_names)
    placeholders = tuple(
        node for node in fx_model.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(names):
        raise RuntimeError(
            "lowered training graph input count does not match its live-state "
            f"metadata: {len(placeholders)} != {len(names)}"
        )

    keep = tuple(bool(node.users) for node in placeholders)
    input_end = len(input_keys)
    parameter_end = input_end + len(parameter_names)
    for name, node, is_live in zip(
        parameter_names,
        placeholders[input_end:parameter_end],
        keep[input_end:parameter_end],
    ):
        if is_live:
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                "cannot validate whether unused parameter placeholder "
                f"{name!r} is trainable"
            )
        if value.requires_grad:
            raise RuntimeError(
                f"lowered training graph has unused trainable parameter {name!r}"
            )

    for name, node, is_live in zip(
        buffer_names,
        placeholders[parameter_end:],
        keep[parameter_end:],
    ):
        if is_live:
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"cannot validate unused buffer placeholder {name!r}")
        if value.requires_grad:
            raise RuntimeError(
                f"lowered training graph has unused differentiable buffer {name!r}"
            )

    for node, is_live in zip(placeholders, keep):
        if not is_live:
            fx_model.graph.erase_node(node)
    fx_model.graph.lint()
    fx_model.recompile()

    live_input_keys = tuple(
        name for name, is_live in zip(input_keys, keep[:input_end]) if is_live
    )
    live_parameter_names = tuple(
        name
        for name, is_live in zip(parameter_names, keep[input_end:parameter_end])
        if is_live
    )
    live_buffer_names = tuple(
        name for name, is_live in zip(buffer_names, keep[parameter_end:]) if is_live
    )
    return live_input_keys, live_parameter_names, live_buffer_names


def _lower_training_fx(
    model: torch.nn.Module,
    example_batch: Mapping[str, torch.Tensor],
    output_args: Mapping[str, bool],
) -> GraphModule:
    flat_model, input_keys, parameter_names, buffer_names, inputs = _trace_inputs(
        model, example_batch, output_args
    )
    trace_inputs = tuple(inputs)
    with _fx_duck_shape(False), _traceable_legacy_spherical_harmonics(model):
        fx_model = make_fx(
            flat_model,
            decomposition_table=mace_decomp_table(),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*trace_inputs)
    del inputs, trace_inputs
    architecture_constants = _classify_architecture_constants(
        fx_model, model, flat_model.model
    )
    del flat_model
    input_keys, parameter_names, buffer_names = _prune_unused_placeholders(
        fx_model, input_keys, parameter_names, buffer_names
    )
    fx_model.meta["mace_input_keys"] = input_keys
    fx_model.meta["mace_output_keys"] = _requested_output_keys(output_args)
    fx_model.meta["mace_parameter_names"] = parameter_names
    fx_model.meta["mace_buffer_names"] = buffer_names
    fx_model.meta["mace_architecture_constants"] = architecture_constants
    return fx_model


def _batch_shape_signature(
    batch: Mapping[str, torch.Tensor],
) -> tuple[int, int, int]:
    return (
        int(batch["ptr"].numel() - 1),
        int(batch["positions"].shape[0]),
        int(batch["edge_index"].shape[1]),
    )


def _first_code_difference(left: GraphModule, right: GraphModule) -> str:
    differing_lines = []
    for line in difflib.unified_diff(
        left.code.splitlines(),
        right.code.splitlines(),
        n=1,
        lineterm="",
    ):
        if line.startswith(("-", "+")) and not line.startswith(("---", "+++")):
            differing_lines.append(line[:500])
            if len(differing_lines) == 2:
                return "\n".join(differing_lines)
    if differing_lines:
        return differing_lines[0]
    return "FX programs differ without a printable line difference"


def make_training_fx(
    model: torch.nn.Module,
    example_batch: Mapping[str, torch.Tensor],
    output_args: Mapping[str, bool],
    validation_batch: Mapping[str, torch.Tensor] | None = None,
) -> GraphModule:
    """Lower MACE's internal conservative derivative to a functional FX graph."""

    fx_model = _lower_training_fx(model, example_batch, output_args)
    if validation_batch is None:
        return fx_model

    first_shape = _batch_shape_signature(example_batch)
    second_shape = _batch_shape_signature(validation_batch)
    if first_shape == second_shape:
        raise RuntimeError(
            "dynamic training compilation requires two batches with different "
            f"graph-derived dimensions; both were {first_shape}"
        )

    validation_fx = _lower_training_fx(model, validation_batch, output_args)
    if fx_model.code != validation_fx.code:
        difference = _first_code_difference(fx_model, validation_fx)
        raise RuntimeError(
            "dynamic training FX specialized graph-derived dimensions: "
            f"{first_shape} versus {second_shape}; first FX divergence:\n"
            f"{difference}"
        )
    fx_model.meta["mace_dynamic_shape_signatures"] = (first_shape, second_shape)
    return fx_model


def _has_small_graph_dimension(batch: Mapping[str, torch.Tensor]) -> bool:
    return (
        batch["positions"].shape[0] < 2
        or batch["edge_index"].shape[1] < 2
        or batch["ptr"].numel() - 1 < 2
    )


class CompiledTrainingModel(torch.nn.Module):
    """Dictionary-compatible facade over a lowered, compiled MACE graph."""

    def __init__(
        self,
        model: torch.nn.Module,
        example_batch: Mapping[str, torch.Tensor],
        output_args: Mapping[str, bool],
        mode: str,
        dynamic: bool,
        validation_batch: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.output_args = dict(output_args)
        self.output_keys = _requested_output_keys(output_args)
        self.input_keys: tuple[str, ...] = ()
        self.parameter_names = tuple(name for name, _ in model.named_parameters())
        self.buffer_names = tuple(name for name, _ in model.named_buffers())
        self.dynamic_shape_signatures: tuple[tuple[int, int, int], ...] | None = None
        self._compiled_ref: tuple[Any | None] = (None,)
        self.compilation_error: str | None = None

        unsupported = {
            key
            for key, requested in output_args.items()
            if requested and key not in _SUPPORTED_OUTPUT_FLAGS
        }
        if unsupported:
            raise ValueError(
                "compile_training_model does not support output objectives: "
                + ", ".join(sorted(unsupported))
            )
        if _has_small_graph_dimension(example_batch):
            warnings.warn(
                "MACE compiler-friendly training skipped: the example batch "
                "has fewer than two graphs, nodes, or edges; using eager "
                "execution for this objective",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        try:
            if dynamic and validation_batch is None:
                raise RuntimeError(
                    "dynamic training compilation requires a second validation batch"
                )
            if _TORCH_TOO_OLD_FOR_DYNAMIC_SHAPES:
                # Inductor in torch releases older than 2.6 cannot lower the
                # dynamic-shape views this path traces (floor-division shape
                # cascades); route through the shared fallback below instead
                # of failing mid-lowering at runtime.
                raise RuntimeError(
                    "torch >= 2.6 is required for dynamic-shape lowering of "
                    "compiled training"
                )
            fx_model = make_training_fx(
                model,
                example_batch,
                output_args,
                validation_batch=validation_batch,
            )
            self.input_keys = fx_model.meta["mace_input_keys"]
            self.parameter_names = fx_model.meta["mace_parameter_names"]
            self.buffer_names = fx_model.meta["mace_buffer_names"]
            self.dynamic_shape_signatures = fx_model.meta.get(
                "mace_dynamic_shape_signatures"
            )
            compiled = torch.compile(
                fx_model,
                mode=mode,
                dynamic=dynamic,
                fullgraph=False,
            )
            self._compiled_ref = (compiled,)
        except Exception as exc:  # pylint: disable=broad-except
            self.compilation_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "MACE compiler-friendly training setup failed; using eager "
                f"execution for this objective: {self.compilation_error}",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def is_compiled(self) -> bool:
        """Whether this objective has an active lowered compiled variant."""

        return self._compiled_ref[0] is not None

    def disable_compilation(self, reason: str | None = None) -> None:
        """Permanently use eager execution without changing model ownership."""

        self._compiled_ref = (None,)
        if reason is not None:
            self.compilation_error = reason

    def materialize_compiled_backward(self, batch: Mapping[str, torch.Tensor]) -> None:
        """Compile the backward graph before DDP installs reducer hooks."""

        if not self.is_compiled:
            return
        try:
            outputs = self(batch)
            if not self.is_compiled:
                return
            differentiable_outputs = tuple(
                output for output in outputs.values() if output.requires_grad
            )
            parameters = tuple(
                parameter
                for parameter in self.base_model.parameters()
                if parameter.requires_grad
            )
            torch.autograd.grad(
                differentiable_outputs,
                parameters,
                grad_outputs=tuple(
                    torch.ones_like(output) for output in differentiable_outputs
                ),
                allow_unused=False,
            )
        except Exception as exc:  # pylint: disable=broad-except
            reason = f"{type(exc).__name__}: {exc}"
            self.disable_compilation(reason)
            warnings.warn(
                "MACE compiler-friendly training backward setup failed before "
                f"DDP construction; using eager execution: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _eager_forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        prepared = _prepare_batch(batch, self.output_args)
        outputs = self.base_model(prepared, **_model_kwargs(self.output_args))
        return {key: outputs[key] for key in self.output_keys}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        compiled = self._compiled_ref[0]
        if compiled is None or _has_small_graph_dimension(batch):
            return self._eager_forward(batch)

        prepared = _prepare_batch(batch, self.output_args)
        parameters = dict(self.base_model.named_parameters())
        buffers = dict(self.base_model.named_buffers())
        flat_inputs = [prepared[key] for key in self.input_keys]
        flat_inputs.extend(parameters[name] for name in self.parameter_names)
        flat_inputs.extend(buffers[name] for name in self.buffer_names)
        try:
            flat_outputs = compiled(*flat_inputs)
        except Exception as exc:  # pylint: disable=broad-except
            self.compilation_error = f"{type(exc).__name__}: {exc}"
            self._compiled_ref = (None,)
            warnings.warn(
                "MACE compiler-friendly training execution failed; permanently "
                f"using eager execution for this objective: {self.compilation_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._eager_forward(batch)
        return dict(zip(self.output_keys, flat_outputs))

    def parameters(self, recurse: bool = True):
        if not recurse:
            return iter(())
        return self.base_model.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        if not recurse:
            return iter(())
        return self.base_model.named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def buffers(self, recurse: bool = True):
        if not recurse:
            return iter(())
        return self.base_model.buffers(recurse=recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        if not recurse:
            return iter(())
        return self.base_model.named_buffers(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.train(mode)
        return self


def compile_training_model(
    model: torch.nn.Module,
    example_batch: Mapping[str, torch.Tensor],
    output_args: Mapping[str, bool],
    mode: str = "default",
    dynamic: bool = True,
    validation_batch: Mapping[str, torch.Tensor] | None = None,
) -> CompiledTrainingModel:
    """Create a compiler-friendly force/stress training facade for ``model``."""

    return CompiledTrainingModel(
        model=model,
        example_batch=example_batch,
        output_args=output_args,
        mode=mode,
        dynamic=dynamic,
        validation_batch=validation_batch,
    )


__all__ = [
    "CompiledTrainingModel",
    "compile_training_model",
    "mace_decomp_table",
    "make_training_fx",
]
