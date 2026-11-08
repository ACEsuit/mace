"""Compatibility name for configuration sections; ConfigSection owns immutability."""

from __future__ import annotations

from mace_core.config.base import ConfigSection

__all__ = ["FrozenSection"]


FrozenSection = ConfigSection
