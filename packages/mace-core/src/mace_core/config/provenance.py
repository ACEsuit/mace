"""Turning what a run asked for into what its model records.

The configuration says where a head's isolated-atom energies should come from;
the metadata says where they came from. They are different statements, and the
gap between them is the whole reason the record exists: `average` is a request
that can fail, and a table read out of a file is a fact.

So this module does the half that is a pure function of the request, and takes
the other half from whoever resolved it. Nothing here fits an E0 or reads a
file. The value of doing it here rather than in the stage that resolves them is
that the mapping from a kind to a recorded method is one line per kind, in one
place, next to the kinds.

The record is keyed by chemical symbol, because a JSON key is a string and an
integer one does not survive the round trip the record exists to make. The
caller converts: a periodic table is not something this module should carry to
rename five keys, and the side that resolved the energies already has one.
"""

from __future__ import annotations

from collections.abc import Mapping

from mace_core.config.e0s import E0Spec, E0sTable
from mace_core.metadata import E0Details

__all__ = ["E0_METHODS", "e0_details"]

#: What each requested kind is recorded as. A table is the only one that is
#: already an answer; the rest name a method that produced one.
E0_METHODS: dict[str, str | None] = {
    "table": None,
    "isolated_atoms": "isolated_atoms",
    "average": "least_squares",
    "foundation": "foundation_model",
    "estimated": "foundation_corrected_least_squares",
}


def e0_details(spec: E0Spec, values: Mapping[str, float]) -> E0Details:
    """Record how one head's E0s were obtained, given the resolved values.

    Args:
        spec: What the configuration asked for.
        values: Chemical symbol to energy, as resolved. Even for a table kind,
            whose spec already carries values: those are keyed by atomic
            number and are the request, and recording a request as a result is
            the confusion this whole object exists to remove.

    Returns:
        The metadata record. ``"explicit"`` for a table, which is already an
        answer, and ``"estimated"`` with a named method for the four that are
        requests. The kind's own settings are carried as ``parameters``, so
        two runs of the same method against different foundation heads read as
        two different things.

    Raises:
        KeyError: If the kind has no recorded method, which means a kind was
            added to the union and not to the table here.
    """
    if spec.kind not in E0_METHODS:
        raise KeyError(
            f"{spec.kind!r} has no recorded method. A kind added to the E0 "
            f"union needs a row here, or a model trained with it records how "
            f"its energies were obtained as a blank."
        )
    return E0Details(
        source="explicit" if isinstance(spec, E0sTable) else "estimated",
        method=E0_METHODS[spec.kind],
        parameters={
            name: value
            for name, value in spec.model_dump(mode="json").items()
            if name not in {"kind", "values"}
        },
        values=dict(sorted(values.items())),
    )
