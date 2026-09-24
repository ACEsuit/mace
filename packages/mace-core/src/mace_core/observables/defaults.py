"""The default catalogue: energy and its derivatives, the dipole and the
polarizability, and their derivatives against the positions.

It is built from the same objects as any other catalogue, which is what makes
it the example every other declaration copies. A new property is one more
:class:`~mace_core.observables.ObservableSpec` in the catalogue a model is
given, and needs no edit to this package: the spec drives the head, the loss
term, the derivative names and the per-atom or per-graph padding.

A derivative with a name of its own declares it here, together with the sign
it is reported with, rather than being special-cased in code. Leave both out
and it is called ``d_<quantity>_d_<input>`` and carries the gradient's own
sign; give a name and the sign becomes required, because a renamed quantity
that silently inherited +1 is a model trained on inverted forces that runs.

Units follow the project convention: eV and Angstrom. The strain is
dimensionless, so its input carries ``"1"``.
"""

from __future__ import annotations

from mace_core.observables.spec import (
    DerivativeRequest,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
)

__all__ = ["DEFAULT_CATALOGUE"]

#: Atomic positions. A polar vector: it changes sign under inversion, which is
#: what makes the energy gradient taken against it a ``1o`` quantity as well.
_POSITIONS = InputSpec(name="pos", irreps="1o", per_atom=True, units="Å")

#: The symmetric strain the stress is the derivative against, not the nine cell
#: entries: a symmetric rank-2 tensor is a scalar plus an l=2 part, six
#: components. Unlike ``pos`` it is not read from the data. The derivative
#: engine materialises it as zeros around the model call and applies it to the
#: positions and the cell, which is how the frozen tree does it too.
_STRAIN = InputSpec(name="strain", irreps="0e+2e", per_atom=False, units="1")

_ENERGY = ObservableSpec(
    name="energy",
    irreps="0e",
    per_atom=False,
    units="eV",
    derivatives=(
        DerivativeRequest(wrt="pos", name="forces", sign=-1, units="eV/Å"),
        # The volume division that turns the strain derivative into a stress is
        # not a sign, and does not belong here. It is applied by whatever
        # computes the stress.
        DerivativeRequest(wrt="strain", name="stress", sign=+1, units="eV/Å^3"),
    ),
)

#: The total dipole of a structure, in e Angstrom. Its derivative against the
#: positions is what an infrared intensity is computed from, and it keeps the
#: frozen tree's name for it and the gradient's own sign.
_DIPOLE = ObservableSpec(
    name="dipole",
    irreps="1o",
    per_atom=False,
    units="e*Å",
    derivatives=(DerivativeRequest(wrt="pos", name="dmu_dr", sign=+1),),
)

#: The polarizability of a structure, a symmetric matrix, in Angstrom cubed.
#: Its position derivative is what a Raman intensity is computed from.
_POLARIZABILITY = ObservableSpec(
    name="polarizability",
    irreps="0e+2e",
    per_atom=False,
    units="Å^3",
    derivatives=(DerivativeRequest(wrt="pos", name="dalpha_dr", sign=+1),),
)

#: Every spec in it is frozen, so it is shared rather than rebuilt.
DEFAULT_CATALOGUE = ObservableCatalogue(
    inputs=(_POSITIONS, _STRAIN), observables=(_ENERGY, _DIPOLE, _POLARIZABILITY)
)
