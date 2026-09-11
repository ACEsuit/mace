"""The typed object a model returns instead of a dictionary of tensors.

Every legacy model ``forward`` returns ``Dict[str, Optional[torch.Tensor]]``,
and between the eleven of them they emit 43 distinct string keys. Nothing
checks a key against anything, so each consumer keeps its own hand-written
list of the names it knows: the ase calculator classifies 22 of the 43 and
returns the other 21 with their padding rows still in them. This module
replaces the dictionary with a typed object whose core fields are named once.

Six fields are core and everything else goes through :attr:`MACEOutput.extras`.
That split is deliberate rather than a compromise: the electrostatic, magnetic
and dielectric families are large, model-specific and still moving, so pinning
them as attributes would mean a core type that changes shape every time a new
model lands. What classifies an entry of ``extras`` as per-atom or per-graph is
its :class:`~mace_core.observables.ObservableSpec`, not a second key list kept
somewhere else.

The class is generic over the tensor type. ``mace_core`` imports no framework,
so ``TensorT`` is bound to ``torch.Tensor`` in ``mace_torch``, to ``jax.Array``
in ``mace_jax``, and to ``numpy.ndarray`` in this package's own tests. The same
pattern carries the kernel Protocol, so it has to work with no framework
installed at all.

Units follow the project convention: eV, Å, and eV/Å for a force.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Generic, TypeVar

__all__ = [
    "CORE_FIELD_NAMES",
    "FIELD_BY_OBSERVABLE",
    "MACEOutput",
    "TensorT",
]

#: The array type a framework binds. Deliberately unbound: a bound of, say,
#: "something with a .shape" would be a structural claim about torch and jax
#: that this package cannot check and does not need.
TensorT = TypeVar("TensorT")


@dataclass
class MACEOutput(Generic[TensorT]):
    """What a model computed, in one typed object.

    A field left at ``None`` was not computed. That is different from an entry
    of ``extras`` that is missing: a core field is part of the type whether or
    not this model produces it, while ``extras`` carries only what was asked
    for.

    Attributes:
        total_energy: Total energy per graph, in eV. Shape ``(n_graphs,)``.
            The observable is named ``energy``; see :data:`FIELD_BY_OBSERVABLE`.
        node_energies: Per-atom energy, in eV, shape ``(n_atoms,)``. Whether
            the isolated-atom reference is included is the model's business
            and is stated by the model, not here.
        forces: ``-d(energy)/d(positions)``, in eV/Å, shape ``(n_atoms, 3)``.
        stress: ``+d(energy)/d(strain) / volume``, in eV/Å³, shape
            ``(n_graphs, 3, 3)``.
        virials: The same derivative before the volume division, in eV, shape
            ``(n_graphs, 3, 3)``.
        dipole: Total dipole per graph, shape ``(n_graphs, 3)``.
        extras: Every other declared observable, keyed by its
            :class:`~mace_core.observables.ObservableSpec` name.

    The object is mutable on purpose. Forces and stress are computed by a
    derivative engine *around* the model call rather than inside a module's
    forward, so something has to fill those fields in after the model returned,
    and a frozen object would mean copying the whole thing to do it.
    """

    total_energy: TensorT | None = None
    node_energies: TensorT | None = None
    forces: TensorT | None = None
    stress: TensorT | None = None
    virials: TensorT | None = None
    dipole: TensorT | None = None
    extras: dict[str, TensorT] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an ``extras`` key that a core field already owns.

        Writing ``extras["forces"]`` is otherwise silent: the value is stored,
        ``output.forces`` stays ``None``, and every consumer that reads the
        field sees nothing. That is the shape of bug this class exists to
        remove, so it is an error at construction instead.
        """
        shadowed = sorted(
            name
            for name in self.extras
            if FIELD_BY_OBSERVABLE.get(name, name) in CORE_FIELD_NAMES
        )
        if shadowed:
            raise ValueError(
                f"{shadowed} are core fields of MACEOutput and cannot also be "
                f"keys of `extras`: a consumer reading the field would see "
                f"nothing. Assign them as fields instead."
            )

    def get(self, name: str) -> TensorT | None:
        """The value stored under ``name``, or ``None`` if there is none.

        ``name`` is an observable name or a core field name. Without this, a
        consumer that iterates over declared observables has to branch on
        which of them happen to be core fields, and that branch is the
        hand-kept key list this class exists to remove.
        """
        field_name = FIELD_BY_OBSERVABLE.get(name, name)
        if field_name in CORE_FIELD_NAMES:
            return getattr(self, field_name)
        return self.extras.get(name)

    def names(self) -> tuple[str, ...]:
        """Every name that carries a value, core fields first, then ``extras``.

        A core field holding ``None`` was not computed and is left out, so this
        is what the model actually produced rather than what it could produce.
        """
        present = [name for name in CORE_FIELD_NAMES if getattr(self, name) is not None]
        present.extend(self.extras)
        return tuple(present)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None


#: The six fields that are part of the type. Derived from the dataclass rather
#: than written out again, so the two cannot disagree.
CORE_FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in fields(MACEOutput) if f.name != "extras"
)

#: The one place an observable's name and its storage field differ. The field
#: says "total" because the type also carries per-atom energies, while the
#: observable is named ``energy`` because that is the name the derivative
#: grammar's special cases are keyed on (``energy`` + positions -> ``forces``).
#: Written down as one entry rather than left to each consumer to remember.
FIELD_BY_OBSERVABLE: dict[str, str] = {"energy": "total_energy"}
