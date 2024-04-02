from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

import torch
import torch._dynamo as dynamo
import torch.nn as nn
import torch.nn.functional as F
from e3nn import get_optimization_defaults, set_optimization_defaults
from torch import autograd
from torch.fx import symbolic_trace

ModuleFactory = Callable[..., nn.Module]


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
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
    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
    elif dynamo.allowed_functions.is_allowed(autograd.grad):
        dynamo.disallow_in_graph(autograd.grad)

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

    Modules are registered with the `simplify_if_compile` decorator.

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


def find_attr(module: nn.Module, name: str) -> Any:
    """Recursively search the given module for a named attribute.

    Args:
        module (nn.Module): root module to search
        name (str): the attribute to match

    Raises:
        ValueError: the attribute must exist in the module.

    Returns:
        Any: mapped value associated with the attribute
    """
    out = list(set(getattr(m, name) for m in module.modules() if hasattr(m, name)))

    if len(out) == 0:
        raise ValueError(f"Failed to find attribute {name}")

    return out[0] if len(out) == 1 else out


def module_like(input: nn.Module, *args, **kwargs) -> nn.Module:
    """Create new module with the same state as the input.

    Args:
        input (nn.Module): reference module used to create the new one

    Returns:
        nn.Module: the new module
    """
    like = type(input)(*args, **kwargs)
    like.load_state_dict(input.state_dict())
    return like


def trampoline(file: str) -> nn.Module:
    """trampoline to jump from file -> model -> torch.compile variant.

    Args:
        file (str): file to initiate the leaping sequence.

    Returns:
        nn.Module: torch.compile compatible model with weights from the input file.
    """
    model = torch.load(file)

    model_config = {
        "gate": F.silu,
        "max_ell": model.spherical_harmonics.irreps_out.lmax,
        "correlation": find_attr(model, "correlation"),
        "r_max": float(model.r_max),
        "num_bessel": len(find_attr(model, "bessel_weights")),
        "num_polynomial_cutoff": float(model.radial_embedding.cutoff_fn.p),
        "interaction_cls_first": model.interactions[0].__class__,
        "interaction_cls": model.interactions[1].__class__,
        "num_interactions": len(model.interactions),
        "num_elements": len(model.atomic_numbers),
        "hidden_irreps": model.interactions[0].hidden_irreps,
        "MLP_irreps": model.readouts[-1].hidden_irreps,
        "atomic_energies": model.atomic_energies_fn.atomic_energies.numpy(),
        "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
        "atomic_numbers": model.atomic_numbers.numpy(),
    }

    if hasattr(model, "scale_shift"):
        model_config.update(
            {
                "atomic_inter_scale": float(model.scale_shift.scale),
                "atomic_inter_shift": float(model.scale_shift.shift),
            }
        )

    return prepare(module_like)(model, **model_config)
