"""Record resolved isolated-atom energies and the method that produced them.

Callers supply resolved values keyed by chemical symbol for JSON round trips.
This module maps the requested E0 kind to provenance without fitting or I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def e0_details(
    spec: E0Spec,
    values: Mapping[str, float],
    *,
    from_foundation: Sequence[str] = (),
) -> E0Details:
    """Record one head's resolved E0s and provenance.

    Args:
        spec: Requested E0 source and method settings.
        values: Resolved energies in eV, keyed by chemical symbol.
        from_foundation: Symbols whose references came from the foundation model.

    Returns:
        ``"explicit"`` provenance for a table; ``"estimated"`` with a method name
        otherwise. Method settings are retained as parameters.

    Raises:
        KeyError: The E0 kind has no entry in ``E0_METHODS``.
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
            **{
                name: value
                for name, value in spec.model_dump(mode="json").items()
                if name not in {"kind", "values"}
            },
            **({"from_foundation": sorted(from_foundation)} if from_foundation else {}),
        },
        values=dict(sorted(values.items())),
    )
