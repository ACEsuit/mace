"""Checking a converted model against committed reference values.

A conversion is correct when the converted model computes what the source did,
and the way to know that without the source is a reference written from it.
The reference is the golden harness's JSON: one entry per named structure, each
with the outputs the source computed. The structures themselves come from a
directory holding the harness's ``manifest.json`` and the files it names, so a
reference is checked against exactly the geometries it was written on.

The comparison is on outputs, never on weights: the conversion projects the
symmetric contraction onto another basis, so no weight is equal to a source
weight and none is meant to be.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable

from mace_torch.data.batch import collate_training
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.physics import DerivativeEngine

__all__ = [
    "REFERENCE_ATOL",
    "REFERENCE_RTOL",
    "Deviation",
    "ReferenceError",
    "VerificationError",
    "VerificationReport",
    "read_reference",
    "verify_against_reference",
]

#: The tolerance a committed float64 CPU reference is held to. The golden
#: harness states the same numbers in its own table, and a test keeps the two
#: equal, so there is one bound however the check is reached.
REFERENCE_ATOL = 1e-6
REFERENCE_RTOL = 0.0

#: The outputs compared, as the reference names them.
QUANTITIES = ("energy", "forces", "stress")


class ReferenceError(ValueError):
    """The reference or its structures cannot be read as a reference."""


class VerificationError(RuntimeError):
    """The converted model does not reproduce the reference."""

    def __init__(self, report: VerificationReport):
        super().__init__(report.describe())
        self.report = report


@dataclass(frozen=True)
class Deviation:
    """How far one output of one structure is from its reference.

    Attributes:
        fixture: The structure's name.
        quantity: ``"energy"``, ``"forces"`` or ``"stress"``.
        largest: The largest absolute difference, in the output's units.
        allowed: The bound at the element where the excess over the bound is
            largest: ``atol + rtol * |reference|``.
    """

    fixture: str
    quantity: str
    largest: float
    allowed: float
    passed: bool


@dataclass(frozen=True)
class VerificationReport:
    """Every comparison made, and which structures could not be evaluated."""

    deviations: tuple[Deviation, ...]
    skipped: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(deviation.passed for deviation in self.deviations)

    def describe(self) -> str:
        lines = [
            f"{d.fixture} {d.quantity}: {d.largest:.3e} against {d.allowed:.3e}"
            f"{'' if d.passed else '  EXCEEDED'}"
            for d in self.deviations
        ]
        if self.skipped:
            lines.append(
                f"skipped, holding elements the model was not built for: "
                f"{', '.join(self.skipped)}"
            )
        verdict = "reproduces" if self.passed else "does not reproduce"
        return f"the converted model {verdict} the reference:\n  " + "\n  ".join(lines)


@dataclass(frozen=True)
class _Case:
    name: str
    atoms: Atoms
    expected: dict[str, np.ndarray]


def read_reference(reference: str | Path, fixtures: str | Path) -> list[_Case]:
    """The structures a reference was written on, with its values.

    Args:
        reference: The harness JSON.
        fixtures: The directory holding ``manifest.json`` and the structures.

    Raises:
        ReferenceError: If the reference is not schema version 1, names a
            structure the manifest lacks, or holds no structure at all.
    """
    document = json.loads(Path(reference).read_text())
    if document.get("schema_version") != 1:
        raise ReferenceError(
            f"{reference} is schema version {document.get('schema_version')!r}, "
            f"and this reads 1."
        )
    manifest_path = Path(fixtures) / "manifest.json"
    if not manifest_path.exists():
        raise ReferenceError(
            f"no manifest.json in {fixtures}. The structures a reference was "
            f"written on are named there."
        )
    manifest = json.loads(manifest_path.read_text())["fixtures"]
    cases = []
    for name, entry in document.get("fixtures", {}).items():
        if name not in manifest:
            raise ReferenceError(
                f"{reference} names the structure {name!r}, which {manifest_path} "
                f"does not."
            )
        atoms = read(Path(fixtures) / manifest[name]["file"], index=0, format="extxyz")
        assert isinstance(atoms, Atoms)
        outputs = entry.get("outputs", {})
        expected = {
            quantity: np.asarray(outputs[quantity]["value"], dtype=np.float64)
            for quantity in QUANTITIES
            if quantity in outputs
        }
        cases.append(_Case(name=name, atoms=atoms, expected=expected))
    if not cases:
        raise ReferenceError(
            f"{reference} holds no structure. A reference that checks nothing "
            f"passes whatever the model computes."
        )
    return cases


def _evaluate(
    engine: DerivativeEngine,
    atoms: Atoms,
    *,
    z_table: AtomicNumberTable,
    cutoff: float,
    head: int,
) -> dict[str, np.ndarray]:
    cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
    periodic = atoms.get_pbc()
    configuration = Configuration(
        atomic_numbers=np.asarray(atoms.get_atomic_numbers()),
        positions=np.asarray(atoms.get_positions(), dtype=np.float64),
        cell=cell,
        pbc=(bool(periodic[0]), bool(periodic[1]), bool(periodic[2])),
    )
    graph = graph_from_configuration(
        configuration, cutoff=cutoff, z_table=z_table, head=head
    )
    batch = collate_training([(graph, {}, {})], z_table=z_table)
    output = engine(dict(batch.graph), compute=("forces", "stress"))
    values = {
        "energy": output.total_energy.detach().reshape(()),
        "forces": output.forces.detach() if output.forces is not None else None,
        "stress": output.stress.detach().reshape(3, 3)
        if output.stress is not None
        else None,
    }
    return {
        name: value.to(torch.float64).cpu().numpy()
        for name, value in values.items()
        if value is not None
    }


def verify_against_reference(
    engine: DerivativeEngine,
    reference: str | Path,
    fixtures: str | Path,
    *,
    z_table: AtomicNumberTable,
    cutoff: float,
    head: int = 0,
    atol: float = REFERENCE_ATOL,
    rtol: float = REFERENCE_RTOL,
) -> VerificationReport:
    """Evaluate the model on every structure of a reference and compare.

    Structures holding an element outside ``z_table`` are skipped and listed,
    since the model cannot evaluate them at all; if that leaves none, the
    reference checks nothing and is refused.

    Raises:
        ReferenceError: If the reference cannot be read, or nothing in it can
            be evaluated.
    """
    deviations: list[Deviation] = []
    skipped: list[str] = []
    for case in read_reference(reference, fixtures):
        if not set(int(z) for z in case.atoms.get_atomic_numbers()) <= set(z_table.zs):
            skipped.append(case.name)
            continue
        got = _evaluate(engine, case.atoms, z_table=z_table, cutoff=cutoff, head=head)
        deviations.extend(_compare(case.name, got, case.expected, atol, rtol))
    if not deviations:
        raise ReferenceError(
            f"no structure in {reference} could be evaluated: {skipped} all hold "
            f"elements outside the model's table {list(z_table.zs)}."
        )
    return VerificationReport(deviations=tuple(deviations), skipped=tuple(skipped))


def _compare(
    fixture: str,
    got: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    atol: float,
    rtol: float,
) -> Iterable[Deviation]:
    for quantity, reference in expected.items():
        if quantity not in got:
            raise ReferenceError(
                f"the reference holds {quantity} for {fixture} and the model "
                f"computes none."
            )
        value = got[quantity].reshape(reference.shape)
        difference = np.abs(value - reference)
        bound = atol + rtol * np.abs(reference)
        excess = difference - bound
        worst = int(np.argmax(excess)) if excess.size else 0
        yield Deviation(
            fixture=fixture,
            quantity=quantity,
            largest=float(difference.max()) if difference.size else 0.0,
            allowed=float(np.ravel(bound)[worst]) if bound.size else atol,
            passed=bool(np.all(difference <= bound)),
        )
