"""A configuration section that cannot be written to after it is validated.

The whole point of a resolved config is that it is the answer, so the shape it
replaces is the one where the answer keeps changing: legacy's `run()` assigns
to its argparse namespace at 43 sites, and a reader of any of those lines
cannot tell which value the training loop eventually saw.

`frozen` on the root alone does not do it. Pydantic freezes a model's own
fields, and a section nested in it is a different model, so
`config.training.lr = 1.0` goes through while `config.training = ...` does not.
Every section therefore inherits this rather than the base, and a test walks
the tree so a section that picks the wrong base fails rather than being the one
mutable corner.

Whether the base itself should be frozen is the config machinery's call, not
this module's; until it is, this is where the invariant lives.
"""

from __future__ import annotations

from mace_core.config.base import ConfigSection

__all__ = ["FrozenSection"]


class FrozenSection(ConfigSection):
    """A section whose fields cannot be assigned to after validation."""

    model_config = ConfigSection.model_config | {"frozen": True}
