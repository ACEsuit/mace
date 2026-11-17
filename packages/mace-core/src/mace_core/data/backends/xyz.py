"""Files ase can read, parsed whole when opened and held in memory.

Parsing is :func:`~mace_core.data.xyz.read_configurations`, so the key
resolution, the weights and the isolated-atom references are the ones every
other reader of ase structures gets. What this backend adds is the opening:
the rest is an in-memory dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ase.io.formats import UnknownFileTypeError

from mace_core.data.backend import DataBackendError
from mace_core.data.backends.memory import InMemoryBackend
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import read_configurations

__all__ = ["XYZBackend"]

#: The suffixes ``format="auto"`` claims. Any other format ase reads is opened
#: by naming this backend, ``format="xyz"``: guessing a format from an
#: arbitrary suffix is the fall-through the registry exists to remove.
_SUFFIXES = frozenset({".xyz", ".extxyz"})


def _require_labels(
    source: str,
    key_spec: KeySpecification,
    configurations: Sequence[Configuration],
    references_taken_out: bool,
) -> None:
    """Refuse a dataset none of whose structures carries a label.

    Checked on the structures the dataset holds, through ``is_labelled``,
    rather than by the parser's check over the whole file, which counts the
    isolated-atom references that are taken out, and a key that ase reserves,
    which holds ``None`` on every structure once rewritten.
    """
    keys = key_spec.graph_keys | key_spec.atom_keys
    names = [name for name in ("energy", "forces", "dipole") if name in keys]
    if any(
        configuration.is_labelled(name)
        for configuration in configurations
        for name in names
    ):
        return
    searched = [keys[name] for name in names]
    if not configurations:
        what = "it holds no structure besides the isolated-atom references"
    elif references_taken_out:
        what = (
            f"no structure it keeps carries any of {searched}; the isolated-atom "
            f"references taken out of the dataset do not count"
        )
    else:
        what = f"no structure carries any of {searched}"
    raise DataBackendError(
        source, reason=f"{what}. Pass require_labels=False to open it unlabelled."
    )


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
                atoms of one element that disagree, or, while
                ``require_labels`` is set, no structure it keeps that carries
                an energy, forces or a dipole.
        """
        try:
            parsed = read_configurations(
                source,
                key_spec,
                head_name=head,
                config_type_weights=config_type_weights,
                extract_isolated_atom_energies=True,
                keep_isolated_atoms=keep_isolated_atoms,
                # The labels are checked below, on the structures kept.
                no_data_ok=True,
            )
        except (OSError, ValueError, KeyError, UnknownFileTypeError) as failure:
            # What a malformed file raises: ase's XYZError is an OSError, an
            # empty file an UnknownFileTypeError, an unknown element symbol a
            # KeyError; the parser reports bad labels as a ValueError.
            raise DataBackendError(
                str(source), reason=f"{type(failure).__name__}: {failure}"
            ) from failure
        if require_labels:
            _require_labels(
                str(source),
                key_spec,
                parsed.configurations,
                references_taken_out=bool(parsed.isolated_atom_energies)
                and not keep_isolated_atoms,
            )
        return cls(
            parsed.configurations,
            source=str(source),
            isolated_atom_energies=parsed.isolated_atom_energies,
        )
