from contextlib import contextmanager
from functools import wraps
from typing import Callable, Mapping, Tuple

import torch

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from e3nn import get_optimization_defaults, set_optimization_defaults
from torch import autograd, nn
from torch.fx import symbolic_trace

ModuleFactory = Callable[..., nn.Module]
TypeTuple = Tuple[type, ...]


def compile_training_model(
    model: nn.Module,
    example_batch: Mapping[str, torch.Tensor],
    output_args: Mapping[str, bool],
    mode: str = "default",
    dynamic: bool = True,
    validation_batch: Mapping[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Compile the experimental training path when its PyTorch APIs exist."""

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "--compile_training requires PyTorch 2.0 or newer; upgrade PyTorch "
            "or omit --compile_training to use standard training."
        )

    try:
        from mace.tools.compile_training import (
            compile_training_model as _compile_training_model,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--compile_training requires PyTorch 2.0 or newer with torch.func "
            "and make_fx support; install a current PyTorch release or omit "
            "--compile_training to use standard training."
        ) from exc

    return _compile_training_model(
        model,
        example_batch,
        output_args=output_args,
        mode=mode,
        dynamic=dynamic,
        validation_batch=validation_batch,
    )


def configure_autograd_for_compile(allow_autograd: bool = True) -> None:
    """Align Dynamo autograd settings with MACE's compile strategy.

    Recent PyTorch releases require `trace_autograd_ops=True` when
    `torch.autograd.grad` is traced into the compiled graph.
    """
    if dynamo is None:
        return

    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
        if hasattr(dynamo.config, "trace_autograd_ops"):
            dynamo.config.trace_autograd_ops = True
    else:
        dynamo.disallow_in_graph(autograd.grad)


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    try:
        yield
    finally:
        set_optimization_defaults(jit_script_fx=init_val)


def prepare(func: ModuleFactory, allow_autograd: bool = True) -> ModuleFactory:
    """Function transform that prepares a MACE module for torch.compile

    Args:
        func (ModuleFactory): A function that creates an nn.Module
        allow_autograd (bool, optional): Force inductor compiler to inline call to
            `torch.autograd.grad`. Defaults to True.

    Returns:
        ModuleFactory: Decorated function that creates a torch.compile compatible module
    """
    configure_autograd_for_compile(allow_autograd=allow_autograd)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_e3nn_codegen():
            model = func(*args, **kwargs)

        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: nn.Module) -> nn.Module:
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplifed using `torch.fx.symbolic_trace`.
    This constrains the module to not have any dynamic control flow, see:

    https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

    Args:
        module (nn.Module): the module to register

    Returns:
        nn.Module: registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module


def simplify(module: nn.Module) -> nn.Module:
    """Recursively searches for registered modules to simplify with
    `torch.fx.symbolic_trace` to support compiling with the PyTorch Dynamo compiler.

    Modules are registered with the `simplify_if_compile` decorator and

    Args:
        module (nn.Module): the module to simplify

    Returns:
        nn.Module: the simplified module
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)

    for name, child in module.named_children():
        if isinstance(child, simplify_types):
            traced = symbolic_trace(child)
            setattr(module, name, traced)
        else:
            simplify(child)

    return module
