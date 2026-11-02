"""Logical model-component identities at the checkpoint boundary.

A component may declare ``canonical_components()``: logical names mapped to
its child modules. Those names are file-format identities, independent of
Python attribute names. An empty name makes an implementation wrapper
transparent. Backend operators retain their existing tensor contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import Tensor, nn

__all__ = ["canonical_modules", "require_member_state"]


def require_member_state(
    state: Mapping[str, Tensor], names: Iterable[str], config_key: str
) -> None:
    """A member accepts exactly its own canonical tensors, never a partial load."""
    expected = set(names)
    if set(state) != expected:
        raise ValueError(
            f"{config_key}: incompatible member state; expected {sorted(expected)}, "
            f"received {sorted(state)}. Load weights written for this member kind."
        )


def canonical_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Resolve logical component names and reject ambiguous or omitted state.

    Plain containers and standalone operators use their registered names.
    Model/block boundaries explicitly bind stable roles to their current
    implementation. No lookup or traversal takes place during ``forward``.
    """
    modules: dict[str, nn.Module] = {}
    ancestors: set[int] = set()

    def recorded(module: nn.Module) -> bool:
        return any(
            callable(getattr(module, name, None))
            for name in ("to_canonical", "canonical_metadata")
        )

    def visit(module: nn.Module, identity: str) -> None:
        if id(module) in ancestors:
            raise ValueError(f"cycle in canonical components at {identity!r}")
        if identity in modules and modules[identity] is not module:
            previous = modules[identity]
            if id(previous) not in ancestors or recorded(previous):
                raise ValueError(f"duplicate canonical state identity {identity!r}")
        modules[identity] = module
        ancestors.add(id(module))
        declared = getattr(module, "canonical_components", None)
        children = (
            declared().items() if declared is not None else module.named_children()
        )
        for role, child in children:
            if child is None:
                continue
            if "::" in role or (role and (role.startswith(".") or role.endswith("."))):
                raise ValueError(f"invalid canonical component identity {role!r}")
            visit(child, ".".join(part for part in (identity, role) if part))
        ancestors.remove(id(module))

    visit(model, "")
    # A component's tensors or layout metadata cannot disappear because its
    # owner forgot to declare it. Derived buffers without a writer are excluded.
    expected = {id(module) for module in model.modules() if recorded(module)}
    found = {id(module) for module in modules.values() if recorded(module)}
    if expected != found:
        raise ValueError(
            "canonical components omit or invent operators with saved state or metadata"
        )
    return modules
