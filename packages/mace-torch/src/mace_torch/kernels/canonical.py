"""Everything a model saves, collected from its modules and put back.

Which tensors are saved is :class:`~mace_core.kernels.protocol.CanonicalState`'s
rule. These two functions walk the module tree and are the one route between a
model and what is saved of it. A tensor no module returns is rebuilt when the
model is built and is registered with ``persistent=False``, so the state dict
holds what is saved and nothing else; a parameter or persistent buffer outside
every :class:`CanonicalState` module is refused rather than dropped.

The result is keyed by module path and then by tensor name. The path is the
model's own tree, the same under every backend because a backend builds each
op at the position the model gives it, and it is already what the weight seeds
are derived from. Renaming a module therefore changes the saved format. Each
module's tensors are in the canonical layout, so the state loads under any
backend. Turning it into files is the checkpoint's business, not this module's.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mace_core.kernels.protocol import CanonicalState
from torch import Tensor, nn

__all__ = ["TrainableConstants", "canonical_state", "load_canonical_state"]


class TrainableConstants(nn.Module):
    """A module whose constants are learned when it is built trainable.

    Each constant is a parameter when trainable and otherwise a buffer the
    config rebuilds, so what the module saves is exactly its parameters: empty
    unless it is trainable. Radial bases, distance transforms and the ZBL term
    are built on it.
    """

    def register_constant(self, name: str, value: Tensor, trainable: bool) -> None:
        """Hold ``value`` as ``name``, learned when ``trainable``."""
        if trainable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value, persistent=False)

    def to_canonical(self) -> dict[str, Tensor]:
        """The trained constants; empty unless built trainable."""
        return {name: value.detach() for name, value in self.named_parameters()}

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            for name, value in self.named_parameters():
                value.copy_(state[name])


def _saving_modules(model: nn.Module) -> dict[str, CanonicalState]:
    """Every module that saves state, by path.

    Raises:
        ValueError: If a parameter or persistent buffer sits outside all of
            them, which a checkpoint would lose.
    """
    modules = {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, CanonicalState)
    }
    unsaved = [
        key
        for key in model.state_dict()
        if not any(path == "" or key.startswith(f"{path}.") for path in modules)
    ]
    if unsaved:
        raise ValueError(
            f"{unsaved} are held by modules that do not implement CanonicalState, "
            f"so a checkpoint would lose them. Give the module that holds each "
            f"to_canonical and load_canonical, or register it with "
            f"persistent=False if the config rebuilds it."
        )
    return modules


def canonical_state(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """Every saved tensor of ``model``, by module path and then by name.

    A module with nothing to save in this configuration, such as a radial basis
    that is not trainable, is left out.

    Raises:
        ValueError: If the model holds a tensor no module would save.
    """
    state: dict[str, dict[str, Tensor]] = {}
    for path, module in _saving_modules(model).items():
        tensors = dict(module.to_canonical())
        if tensors:
            state[path] = tensors
    return state


def load_canonical_state(
    model: nn.Module, state: Mapping[str, Mapping[str, Tensor]]
) -> None:
    """Put a saved state back into a model built from the same config.

    Raises:
        ValueError: If ``state`` lacks a tensor the model saves, holds one it
            does not or one of another shape, or names a module that saves
            nothing. Nothing is loaded then.
    """
    modules = _saving_modules(model)
    expected = canonical_state(model)
    problems = []
    for path in sorted(set(expected) | set(state)):
        label = path or "<root>"
        if path not in modules:
            problems.append(f"{label}: no module here saves state")
            continue
        wanted = expected.get(path, {})
        given = state.get(path, {})
        for name in sorted(set(wanted) - set(given)):
            problems.append(f"{label}: missing {name!r}")
        for name in sorted(set(given) - set(wanted)):
            problems.append(f"{label}: unexpected {name!r}")
        for name in sorted(set(wanted) & set(given)):
            if tuple(given[name].shape) != tuple(wanted[name].shape):
                problems.append(
                    f"{label}: {name!r} has shape {tuple(given[name].shape)}, "
                    f"the model's is {tuple(wanted[name].shape)}"
                )
    if problems:
        raise ValueError(
            "the saved state does not fit this model, which has to be built from "
            "the config it was saved with: " + "; ".join(problems) + "."
        )
    for path, tensors in state.items():
        if tensors:
            modules[path].load_canonical(dict(tensors))
