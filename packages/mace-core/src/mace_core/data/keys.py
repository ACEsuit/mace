"""Which file key each convention name is read from.

Two dictionaries, because a structure file stores two kinds of thing and they
are looked up in different places: per-structure values (an energy, a stress, a
total charge) and per-atom arrays (forces, charges, magnetic moments). The
split is not cosmetic. Asking for ``forces`` among the per-structure values
finds nothing and reports it as a missing label, which is indistinguishable
from a file that genuinely has no forces.

This object is the only place a file key appears. Everything downstream of
parsing is keyed by convention name -- see
:mod:`mace_core.data.configuration`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from mace_core.elements.default_keys import DefaultKeys

__all__ = [
    "ARRAYS_CONVENTION_NAMES",
    "INFO_CONVENTION_NAMES",
    "EmbeddingFeatureSpec",
    "KeySpecification",
]

#: The default convention names stored once per structure.
INFO_CONVENTION_NAMES: frozenset[str] = frozenset(
    {
        "energy",
        "stress",
        "virials",
        "dipole",
        "polarizability",
        "head",
        "elec_temp",
        "total_charge",
        "total_spin",
    }
)

#: The default convention names stored once per atom. ``magmom`` and
#: ``magforces`` are here, on the default path, and not behind a magnetic
#: switch: they are part of the default key table, so every parse resolves
#: them and a magnetically labelled file reads without any flag being set.
ARRAYS_CONVENTION_NAMES: frozenset[str] = frozenset(
    {"forces", "charges", "magmom", "magforces"}
)


def _known_convention_names() -> frozenset[str]:
    return INFO_CONVENTION_NAMES | ARRAYS_CONVENTION_NAMES


@dataclass(frozen=True)
class EmbeddingFeatureSpec:
    """A user-declared input feature the model is given alongside the structure.

    Declaring one is the only way a quantity that is neither a position nor a
    label becomes nameable: the parser learns to read it, and the derivative
    grammar can then be asked for a derivative with respect to it.

    Args:
        per: ``"atom"`` for a per-atom array, ``"graph"`` for one value per
            structure. Nothing else is accepted, because the two are read from
            different places in the file and there is no third place.
        key: The file key to read it from. Defaults to the feature's own name.
    """

    per: Literal["atom", "graph"]
    key: str | None = None

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> EmbeddingFeatureSpec:
        """Build one from the plain mapping a config file or the CLI supplies."""
        try:
            per = spec["per"]
        except KeyError:
            raise ValueError(
                f"embedding feature {name!r} does not say where it is stored. "
                f"Add per: atom for a per-atom array or per: graph for one "
                f"value per structure."
            ) from None
        if per not in ("atom", "graph"):
            raise ValueError(
                f"embedding feature {name!r} declares per: {per!r}, which is "
                f"not a place a value can be read from. Use per: atom for a "
                f"per-atom array or per: graph for one value per structure."
            )
        return cls(per=per, key=spec.get("key"))

    def file_key(self, name: str) -> str:
        """The key to read this feature from, defaulting to the feature name."""
        return self.key if self.key is not None else name


@dataclass
class KeySpecification:
    """Convention name to file key, split by where the value is stored.

    Args:
        info_keys: Per-structure properties.
        arrays_keys: Per-atom properties.
    """

    info_keys: dict[str, str] = field(default_factory=dict)
    arrays_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_defaults(cls) -> KeySpecification:
        """All thirteen default keys, routed to the half each is stored in."""
        return cls().apply_overrides(DefaultKeys.keydict())

    def copy(self) -> KeySpecification:
        """An independent copy. Rewriting keys on it cannot reach the original."""
        return replace(
            self, info_keys=dict(self.info_keys), arrays_keys=dict(self.arrays_keys)
        )

    def update(
        self,
        info_keys: Mapping[str, str] | None = None,
        arrays_keys: Mapping[str, str] | None = None,
    ) -> KeySpecification:
        """Merge explicit keys in, by convention name. Returns ``self``."""
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    def apply_overrides(self, overrides: Mapping[str, Any]) -> KeySpecification:
        """Apply ``<convention name>_key`` overrides. Returns ``self``.

        Entries whose name does not end in ``_key`` are ignored, so a whole
        settings mapping can be handed over without filtering it first. An
        entry that *does* end in ``_key`` but names no known convention is an
        error: silently dropping it would leave the property unparsed, with
        nothing anywhere saying why the labels never arrived.
        """
        known = _known_convention_names()
        unknown: list[str] = []
        for setting, value in overrides.items():
            if not setting.endswith("_key"):
                continue
            name = setting[: -len("_key")]
            if name in INFO_CONVENTION_NAMES:
                self.info_keys[name] = value
            elif name in ARRAYS_CONVENTION_NAMES:
                self.arrays_keys[name] = value
            else:
                unknown.append(setting)
        if unknown:
            raise ValueError(
                f"no property is named by {sorted(unknown)}. The properties "
                f"that can be given a file key are {sorted(known)}; to read "
                f"anything else, declare it as an embedding feature."
            )
        return self

    def add_embedding_features(
        self, embedding_specs: Mapping[str, EmbeddingFeatureSpec | Mapping[str, Any]]
    ) -> KeySpecification:
        """Extend the specification with user-declared input features.

        Returns ``self``. A feature with ``per: atom`` becomes a per-atom key
        and one with ``per: graph`` a per-structure key, read from its declared
        ``key`` or, failing that, from its own name.
        """
        for name, spec in embedding_specs.items():
            feature = (
                spec
                if isinstance(spec, EmbeddingFeatureSpec)
                else EmbeddingFeatureSpec.from_mapping(name, spec)
            )
            target = self.arrays_keys if feature.per == "atom" else self.info_keys
            target[name] = feature.file_key(name)
        return self

    def property_names(self) -> tuple[str, ...]:
        """Every convention name this specification resolves, arrays first.

        The order is the one the legacy weight loop used, and it is the order
        ``property_weights`` is built in. It matters only for reproducibility
        of iteration, not for correctness.
        """
        return (*self.arrays_keys, *self.info_keys)
