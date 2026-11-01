"""Where a head's isolated-atom energies come from.

Legacy spells this as one string that means five different things: a JSON
table, the literal words ``"isolated_atoms"``, ``"average"`` or
``"foundation"``, or a python dict literal parsed with :func:`ast.literal_eval`.
The five carry different fields, and a string cannot hold them, so the extra
settings ended up as separate flags and as behaviour keyed on which word was
written.

Here they are five kinds of one field, written as pydantic's tagged form: the
kind beside its settings::

    [data.heads.default.e0s]
    kind = "foundation"
    head = "mp"
    missing = "average"

**Two legacy fallbacks become errors.** Reading an isolated atom whose energy
key is absent logs a warning and contributes ``0.0``
(``mace/data/utils.py:333-337``), and a singular least-squares system logs an
error and zeroes *every* element (``mace/data/utils.py:381-387``). Both leave a
run training against energies that are silently wrong by a constant per
species, which is a shift the model absorbs into its readout and nothing later
recovers. The first is a field with an explicit default of ``"error"``; the
second has no option at all, because a policy for it would only be chosen by
someone who had not read this paragraph.

Values are energies in eV, IEEE float64 end to end. The resolver that turns one
of these into a table, and the provenance it records, belong to the data stage;
what lives here is the request and the rules that make it coherent.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mace_core.config.section import FrozenSection

__all__ = [
    "E0Spec",
    "E0sAverage",
    "E0sEstimated",
    "E0sFromFoundation",
    "E0sIsolatedAtoms",
    "E0sTable",
    "MissingElementPolicy",
]

#: What to do about an element the source does not cover. ``"error"`` is the
#: default everywhere: a missing E0 is a per-species constant offset in every
#: energy the head trains on, and the two quiet options exist only because a
#: dataset can legitimately outrun its reference table. ``"average"`` fits the
#: uncovered elements to the head's training energies with the covered ones
#: held at the source's values. ``"zero"`` pads with 0.0, and only elements no
#: training structure holds, since a padding anything trains against is a
#: reference energy of zero.
MissingElementPolicy = Literal["error", "average", "zero"]


class E0sTable(FrozenSection):
    """Energies given outright, which is also the parsed form of a JSON file.

    Args:
        values: Atomic number to isolated-atom energy, in eV. Required: a table
            kind with nothing in it resolves to no energies at all, which is
            the all-zero state the other kinds go out of their way to refuse.
    """

    kind: Literal["table"] = "table"
    values: dict[int, float]


class E0sIsolatedAtoms(FrozenSection):
    """Read from the ``IsolatedAtom`` structures in the head's training file.

    The default, because it is what legacy reaches for first.

    Args:
        on_missing_energy: What an isolated atom carrying no energy means.
            ``"error"`` here rather than legacy's warn-and-use-zero: a zero is
            indistinguishable from a real reference energy of zero, and it
            shifts every structure containing that element.
    """

    kind: Literal["isolated_atoms"] = "isolated_atoms"
    on_missing_energy: Literal["error", "zero"] = "error"


class E0sAverage(FrozenSection):
    """Least squares over the head's training energies.

    There is deliberately no fallback field. A singular system means the
    element composition does not determine the E0s, and legacy answers that by
    zeroing every element and carrying on, which trains the model against
    energies wrong by an unknown constant per species. Here it raises.
    """

    kind: Literal["average"] = "average"


class E0sFromFoundation(FrozenSection):
    """Copied from a foundation model's recorded metadata.

    From the artifact's metadata, never read back out of a loaded module's
    buffers: a buffer says what some model was built with, and says nothing
    about which head it belonged to or what it was fitted against.

    Args:
        head: Which of the foundation model's heads to copy from. Required
            when it has more than one, and that is a validation error rather
            than a default: legacy takes head 0 and only logs which it took
            (``mace/cli/run_train.py:510-514``).
        missing: Elements the foundation model does not cover.
    """

    kind: Literal["foundation"] = "foundation"
    head: str | None = None
    missing: MissingElementPolicy = "error"


class E0sEstimated(FrozenSection):
    """Least squares corrected by a foundation model's own energies.

    Args:
        head: Which of the foundation model's heads to correct against.
        missing: Elements the foundation model does not cover.
    """

    kind: Literal["estimated"] = "estimated"
    head: str | None = None
    missing: MissingElementPolicy = "error"


#: The five ways a head can get its isolated-atom energies.
E0Spec = Annotated[
    E0sTable | E0sIsolatedAtoms | E0sAverage | E0sFromFoundation | E0sEstimated,
    Field(discriminator="kind"),
]

#: The kinds that cannot be resolved without a foundation model to read from.
#: Named once here so the cross-section validator and its error message cannot
#: disagree about which they are.
FOUNDATION_E0_KINDS: frozenset[str] = frozenset({"foundation", "estimated"})
