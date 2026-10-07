"""Reading labelled structure files into configurations.

Why ``ase`` is a dependency of a framework-agnostic package: the extended-XYZ
contract is *defined* in terms of ase. A label lives in ``atoms.info`` or in
``atoms.arrays`` depending on whether it is per-structure or per-atom, ase
decides which of the two a given key lands in when a file is read, and the
calculator back-fill below reads values ase moved into ``atoms.calc.results``.
Reimplementing the parser would mean reimplementing those decisions, and then
differing from them on some file nobody tested. ase itself pulls in numpy and
nothing framework-shaped, so it costs the purity rule nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress

from mace_core.data.configuration import (
    DEFAULT_CONFIG_TYPE,
    DEFAULT_HEAD,
    Configuration,
)
from mace_core.data.keys import KeySpecification
from mace_core.elements.default_keys import DefaultKeys

__all__ = [
    "ISOLATED_ATOM_CONFIG_TYPE",
    "ParsedConfigurations",
    "configuration_from_atoms",
    "read_configurations",
]

logger = logging.getLogger(__name__)

#: The config type that marks a single-atom structure as an E0 reference.
ISOLATED_ATOM_CONFIG_TYPE = "IsolatedAtom"


@dataclass(frozen=True)
class _ReservedKey:
    """One property ase reserves the obvious spelling of.

    Args:
        reserved: The spelling that stopped being safe. It coincides with the
            convention name for all three, which is exactly why the collision
            happens, but the two are different things: one is ase's calculator
            property, the other is what this stack calls the label.
        rewritten: What the key is rewritten to for the duration of the parse.
            Read off the default key table rather than spelled again: the
            rewrite has to land on a key the parser then reads, and a second
            copy of ``REF_energy`` here would go on working while the table
            moved underneath it.
        getter: The ``Atoms`` method that recovers the value ase moved into
            the calculator.
        stored_in: Which of ``info`` / ``arrays`` the value belongs in. ase's
            own two words, because this is the attribute the value is read from
            and written to; the format-neutral spelling of the same split is
            ``graph`` and ``atom``, in :mod:`mace_core.data.keys`.
    """

    reserved: str
    rewritten: str
    getter: str
    stored_in: str


#: The structure-level tensors a configuration always holds as a full
#: ``[3, 3]`` matrix. A file may also give the six Voigt components in ase's
#: order (xx, yy, zz, yz, xz, xy), which is what ``Atoms.get_stress`` returns
#: and so what the reserved ``stress`` key recovers; those are expanded here.
#: Any other shape is refused, a flat nine included: reshaping it would have to
#: guess whether it was written row by row.
_FULL_MATRIX_PROPERTIES = frozenset(
    {DefaultKeys.STRESS.convention_name, DefaultKeys.VIRIALS.convention_name}
)

#: Configuring one of these as the file key for a label stopped being safe in
#: ase 3.23: the value written under it is read back into
#: ``atoms.calc.results`` instead of into ``atoms.info``, so a parser looking
#: in ``info`` finds nothing at all.
_RESERVED_KEYS: dict[str, _ReservedKey] = {
    "energy": _ReservedKey(
        "energy", DefaultKeys.ENERGY.value, "get_potential_energy", "info"
    ),
    "forces": _ReservedKey("forces", DefaultKeys.FORCES.value, "get_forces", "arrays"),
    "stress": _ReservedKey("stress", DefaultKeys.STRESS.value, "get_stress", "info"),
}


@dataclass(frozen=True)
class ParsedConfigurations:
    """What one file yielded.

    Args:
        configurations: The structures to train or evaluate on, in file order.
        isolated_atom_energies: ``{atomic number: energy in eV}`` read from the
            single-atom reference structures. Empty unless extraction was
            asked for.
    """

    configurations: list[Configuration] = field(default_factory=list)
    isolated_atom_energies: dict[int, float] = field(default_factory=dict)


def configuration_from_atoms(
    atoms: Atoms,
    key_spec: KeySpecification,
    *,
    config_type_weights: Mapping[str, float] | None = None,
    head_name: str = DEFAULT_HEAD,
) -> Configuration:
    """Convert one structure, resolving every declared key against the file.

    A declared property the file does not carry is stored as ``None`` with
    weight ``0.0``, rather than left out: a zero weight is how a loss term is
    told this structure has no such label, and an absent entry would instead
    look like a property nobody declared.

    The head is not a property and is not read from the file. It is
    ``head_name``, stored once, on :attr:`Configuration.head`.

    A stress or virial is stored as a ``[3, 3]`` matrix; see
    :data:`_FULL_MATRIX_PROPERTIES` for the layouts a file may use.

    Args:
        atoms: The structure, already read.
        key_spec: Which file key each convention name is read from.
        config_type_weights: Per-config-type multiplier on the structure
            weight. An unlisted config type gets ``1.0``; it is not an error.
        head_name: The head this structure trains.

    Raises:
        ValueError: if a stress or virial has a shape that is neither
            ``[3, 3]`` nor six Voigt components.
    """
    config_type = atoms.info.get("config_type", DEFAULT_CONFIG_TYPE)
    type_weight = (config_type_weights or {}).get(config_type, 1.0)

    properties: dict[str, Any] = {}
    property_weights: dict[str, float] = {
        name: atoms.info.get(f"config_{name}_weight", 1.0)
        for name in key_spec.property_names()
    }

    for name, file_key in key_spec.graph_keys.items():
        value = atoms.info.get(file_key)
        if name in _FULL_MATRIX_PROPERTIES and value is not None:
            value = _as_full_matrix(name, value)
        properties[name] = value
        if file_key not in atoms.info:
            property_weights[name] = 0.0
    for name, file_key in key_spec.atom_keys.items():
        properties[name] = atoms.arrays.get(file_key)
        if file_key not in atoms.arrays:
            property_weights[name] = 0.0

    return Configuration(
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        properties=properties,
        property_weights=property_weights,
        cell=np.array(atoms.get_cell()),
        pbc=tuple(atoms.get_pbc().tolist()),
        weight=atoms.info.get("config_weight", 1.0) * type_weight,
        config_type=config_type,
        head=head_name,
    )


def _as_full_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3, 3):
        return array
    if array.shape == (6,):
        return voigt_6_to_full_3x3_stress(array)
    raise ValueError(
        f"{name} has shape {array.shape}. It is read as a 3x3 matrix, or as "
        f"the six Voigt components (xx, yy, zz, yz, xz, xy). A flat list of "
        f"nine is refused rather than reshaped; write the matrix itself."
    )


def read_configurations(
    path: str | Path,
    key_spec: KeySpecification,
    *,
    head_name: str = DEFAULT_HEAD,
    config_type_weights: Mapping[str, float] | None = None,
    extract_isolated_atom_energies: bool = False,
    keep_isolated_atoms: bool = False,
    no_data_ok: bool = False,
) -> ParsedConfigurations:
    """Read every structure in a file into a configuration.

    ``key_spec`` is **not** modified. Reading a file whose keys are ase's
    reserved spellings needs the specification rewritten for the duration of
    the parse, and doing that in place would leave a specification shared
    between two datasets holding whatever the first parse rewrote it to. The
    rewrite happens on a copy, so there is nothing to restore and no ordering
    for a caller to get wrong.

    Args:
        path: The structure file, in any format ase can read.
        key_spec: Which file key each convention name is read from.
        head_name: The head every structure in this file trains.
        config_type_weights: Per-config-type multiplier on the structure weight.
        extract_isolated_atom_energies: Read E0 references out of the
            single-atom structures marked as isolated atoms.
        keep_isolated_atoms: Keep those structures in the returned list as
            well. Without it they are consumed by the extraction.
        no_data_ok: Downgrade the "this file has no labels at all" error to a
            warning. For files that are genuinely unlabelled, such as
            structures to run inference on.

    Raises:
        ValueError: if no structure in the file carries an energy, forces or a
            dipole, and ``no_data_ok`` is not set; if a stress or virial has a
            shape that cannot be read as a ``[3, 3]`` matrix; or if two isolated
            atoms of one element give different reference energies.
    """
    # ase returns a single structure or a list depending on the index; ":"
    # always gives a list, but the signature does not say so.
    read = ase.io.read(str(path), index=":")
    atoms_list: list[Atoms] = read if isinstance(read, list) else [read]

    resolved = _rewrite_reserved_keys(key_spec.copy(), atoms_list)

    _check_something_is_labelled(resolved, atoms_list, str(path), no_data_ok)

    isolated_atom_energies: dict[int, float] = {}
    if extract_isolated_atom_energies:
        isolated_atom_energies = _extract_isolated_atom_energies(
            atoms_list, resolved.graph_keys["energy"]
        )
        if isolated_atom_energies:
            logger.info(
                "Read %d isolated-atom energies from %s",
                len(isolated_atom_energies),
                path,
            )

    # Indexed before the isolated atoms are dropped, so an error names the
    # structure's position in the file.
    indexed = list(enumerate(atoms_list))
    if extract_isolated_atom_energies and not keep_isolated_atoms:
        indexed = [(i, atoms) for i, atoms in indexed if not _is_isolated_atom(atoms)]

    configurations: list[Configuration] = []
    for index, atoms in indexed:
        try:
            configuration = configuration_from_atoms(
                atoms,
                resolved,
                config_type_weights=config_type_weights,
                head_name=head_name,
            )
        except ValueError as error:
            raise ValueError(f"structure {index} of {str(path)!r}: {error}") from None
        configurations.append(configuration)
    return ParsedConfigurations(
        configurations=configurations,
        isolated_atom_energies=isolated_atom_energies,
    )


def _rewrite_reserved_keys(
    key_spec: KeySpecification, atoms_list: Sequence[Atoms]
) -> KeySpecification:
    """Move any reserved key onto its ``REF_`` spelling and recover the values.

    The value is taken from the calculator, which is where ase put it. When a
    structure has no calculator, or the calculator cannot produce that
    quantity, the rewritten key is written anyway, holding ``None``. That is
    what legacy does and what the presence check below is measured against:
    the key counts as present, so a file whose values all failed to recover
    passes the check and yields ``None`` labels at full weight.
    """
    for name, reserved in _RESERVED_KEYS.items():
        store = (
            key_spec.atom_keys
            if reserved.stored_in == "arrays"
            else key_spec.graph_keys
        )
        if store.get(name) != reserved.reserved:
            continue
        logger.warning(
            "Reading %s from the key %r is not safe with ase 3.23 and newer: "
            "ase reads that key back into the calculator, not into the "
            "structure. Rewriting it to %r and recovering the values from the "
            "calculator. Label the file with %r to read it directly.",
            name,
            reserved.reserved,
            reserved.rewritten,
            reserved.rewritten,
        )
        store[name] = reserved.rewritten
        for atoms in atoms_list:
            try:
                value = getattr(atoms, reserved.getter)()
            except Exception as error:
                # Deliberately broad: a structure may carry no calculator at
                # all, or one that raises for this one quantity, and both mean
                # the same thing here -- there is nothing to recover.
                logger.warning(
                    "Failed to recover %s from the calculator: %s", name, error
                )
                value = None
            getattr(atoms, reserved.stored_in)[reserved.rewritten] = value
    return key_spec


def _check_something_is_labelled(
    key_spec: KeySpecification,
    atoms_list: Sequence[Atoms],
    path: str,
    no_data_ok: bool,
) -> None:
    if "energy" not in key_spec.graph_keys or "forces" not in key_spec.atom_keys:
        raise ValueError(
            "the key specification names no energy key, no forces key, or "
            "neither, so there is nothing to look for in the file. Build it "
            "with KeySpecification.from_defaults() and override from there."
        )
    energy_key = key_spec.graph_keys["energy"]
    forces_key = key_spec.atom_keys["forces"]
    # A specification with no dipole entry still has to name a key in the
    # message below, and this is the name legacy reports.
    dipole_key = key_spec.graph_keys.get("dipole", "REF_dipole")

    has_energy = any(energy_key in atoms.info for atoms in atoms_list)
    has_forces = any(forces_key in atoms.arrays for atoms in atoms_list)
    has_dipole = any(dipole_key in atoms.info for atoms in atoms_list)

    if not (has_energy or has_forces or has_dipole):
        message = (
            f"none of {energy_key!r}, {forces_key!r} and {dipole_key!r} is "
            f"present in any structure in {path!r}"
        )
        if not no_data_ok:
            raise ValueError(
                message
                + ". Point the energy, forces or dipole key at the names the "
                + "file actually uses, or pass no_data_ok to read it unlabelled."
            )
        logger.warning("%s. Continuing: no_data_ok was set.", message)
        return

    if not has_energy:
        logger.warning("No energies found under %r in %r.", energy_key, path)
    if not has_forces:
        logger.warning("No forces found under %r in %r.", forces_key, path)


def _is_isolated_atom(atoms: Atoms) -> bool:
    """Both conditions, and both are load-bearing.

    A two-atom structure labelled as an isolated atom is training data that was
    mislabelled, and a lone atom without the label is a legitimate one-atom
    structure to fit. Neither may be consumed as an E0 reference.
    """
    return (
        len(atoms) == 1 and atoms.info.get("config_type") == ISOLATED_ATOM_CONFIG_TYPE
    )


def _extract_isolated_atom_energies(
    atoms_list: Sequence[Atoms], energy_key: str
) -> dict[int, float]:
    """One reference energy per element, independent of the file's order.

    Several isolated atoms of one element are accepted when they agree. When
    they do not, nothing in the file says which is meant (two spin states of
    an oxygen atom are both legitimate isolated-atom energies), so taking any
    one of them would make the E0 depend on the order of the structures. That
    is an error, naming the structures that disagree.

    A marked atom without an energy contributes zero, with a warning, and only
    when no other isolated atom of its element carries an energy.
    """
    labelled: dict[int, dict[float, list[int]]] = {}
    unlabelled: dict[int, list[int]] = {}
    for index, atoms in enumerate(atoms_list):
        if not _is_isolated_atom(atoms):
            continue
        atomic_number = int(atoms.get_atomic_numbers()[0])
        energy = atoms.info.get(energy_key)
        if energy is None:
            unlabelled.setdefault(atomic_number, []).append(index)
        else:
            labelled.setdefault(atomic_number, {}).setdefault(float(energy), []).append(
                index
            )

    conflicts = {
        atomic_number: by_energy
        for atomic_number, by_energy in labelled.items()
        if len(by_energy) > 1
    }
    if conflicts:
        details = "; ".join(
            f"element {atomic_number}: "
            + ", ".join(
                f"{energy!r} (structures {indices})"
                for energy, indices in sorted(by_energy.items())
            )
            for atomic_number, by_energy in sorted(conflicts.items())
        )
        raise ValueError(
            f"isolated atoms of one element give different reference energies "
            f"under {energy_key!r}: {details}. Keep one isolated atom per "
            f"element, or mark the others with another config_type."
        )

    energies = {
        atomic_number: next(iter(by_energy))
        for atomic_number, by_energy in labelled.items()
    }
    for atomic_number, indices in unlabelled.items():
        if atomic_number in energies:
            continue
        logger.warning(
            "Structure(s) %s are marked as an isolated atom but carry no "
            "energy under %r. Recording zero for element %d.",
            indices,
            energy_key,
            atomic_number,
        )
        energies[atomic_number] = 0.0
    return energies
