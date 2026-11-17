"""Files ase can read, parsed whole when opened and held in memory.

Parsing is :func:`~mace_core.data.xyz.read_configurations`, so the key
resolution, the weights and the isolated-atom references are the ones every
other reader of ase structures gets. What this backend adds is the opening:
the rest is an in-memory dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ase.io.formats import UnknownFileTypeError

from mace_core.data.backend import DataBackendError
from mace_core.data.backends.memory import InMemoryBackend
from mace_core.data.configuration import DEFAULT_HEAD
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import read_configurations

__all__ = ["XYZBackend"]

#: The suffixes ``format="auto"`` claims. Any other format ase reads is opened
#: by naming this backend, ``format="xyz"``: guessing a format from an
#: arbitrary suffix is the fall-through the registry exists to remove.
_SUFFIXES = frozenset({".xyz", ".extxyz"})


class XYZBackend(InMemoryBackend):
    """Every structure of one file, parsed when the backend is opened."""

    @classmethod
    def sniff(cls, source: str) -> bool:
        """An existing file with an extended-XYZ suffix."""
        path = Path(source)
        return path.suffix.lower() in _SUFFIXES and path.is_file()

    @classmethod
    def open(
        cls,
        source: str,
        *,
        key_spec: KeySpecification,
        head: str = DEFAULT_HEAD,
        config_type_weights: Mapping[str, float] | None = None,
        keep_isolated_atoms: bool = False,
        require_labels: bool = True,
    ) -> XYZBackend:
        """Parse ``source``. See :meth:`~mace_core.data.backend.DataBackend.open`.

        Raises:
            DataBackendError: wrapping whatever the parse refused: an
                unreadable file, a label of the wrong shape, two isolated
                atoms of one element that disagree, or no labels at all
                while ``require_labels`` is set.
        """
        try:
            parsed = read_configurations(
                source,
                key_spec,
                head_name=head,
                config_type_weights=config_type_weights,
                extract_isolated_atom_energies=True,
                keep_isolated_atoms=keep_isolated_atoms,
                no_data_ok=not require_labels,
            )
        except (OSError, ValueError, KeyError, UnknownFileTypeError) as failure:
            # What a malformed file raises: ase's XYZError is an OSError, an
            # empty file an UnknownFileTypeError, an unknown element symbol a
            # KeyError; the parser reports bad labels as a ValueError.
            raise DataBackendError(
                str(source), reason=f"{type(failure).__name__}: {failure}"
            ) from failure
        if require_labels and not parsed.configurations:
            raise DataBackendError(
                str(source),
                reason=(
                    "it holds no structure to train on, only isolated-atom "
                    "references. Pass require_labels=False to open it anyway."
                ),
            )
        return cls(
            parsed.configurations,
            source=str(source),
            isolated_atom_energies=parsed.isolated_atom_energies,
        )
