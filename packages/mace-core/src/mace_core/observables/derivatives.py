"""How a derivative of a declared quantity is named by default.

The rule is one line: the derivative of a declared quantity ``q`` with respect
to a declared input ``x`` is called ``d_<q>_d_<x>`` and is reported with the
gradient's own sign.

A quantity whose derivative has a name of its own, and a sign of its own, says
so **in its declaration**: the ``name`` and ``sign`` of a
:class:`~mace_core.observables.DerivativeRequest` on the observable. Nothing in
this module knows any such name. ``forces`` and ``stress`` are declared that way
in :data:`~mace_core.observables.DEFAULT_CATALOGUE`, and a further one, torques
as ``-dE/d(orientation)`` or a polarizability as ``d(dipole)/d(field)``, is one
more request in a catalogue rather than an edit to this package.
"""

from __future__ import annotations

import re

__all__ = [
    "DEFAULT_SIGN",
    "default_derivative_name",
    "is_default_shaped_name",
]

#: A derivative is reported with the gradient's own sign unless its declaration
#: says otherwise. ``reported = sign * d(quantity)/d(input)``.
DEFAULT_SIGN: int = 1

_DEFAULT_SHAPE = re.compile(r"^d_.+_d_.+$")


def default_derivative_name(quantity: str, wrt: str) -> str:
    """The name of ``d(quantity)/d(wrt)`` when the declaration gives none.

    Args:
        quantity: The name of the differentiated observable.
        wrt: The name of the declared input it is differentiated against.

    Returns:
        ``f"d_{quantity}_d_{wrt}"``.
    """
    return f"d_{quantity}_d_{wrt}"


def is_default_shaped_name(name: str) -> bool:
    """Whether ``name`` is spelled like one this rule would generate.

    A declared name that is shaped like the grammar's own but belongs to a
    different pair reads as a fact about which quantity was differentiated, and
    is not one. The catalogue refuses it rather than letting the two spellings
    mean different things.
    """
    return bool(_DEFAULT_SHAPE.match(name))
