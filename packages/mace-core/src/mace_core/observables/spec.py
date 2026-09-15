"""Declaring an observable, an input, and the derivatives between them.

Which quantities a legacy model computes is decided by a ladder of string
comparisons on the model class name that sets six boolean flags, plus ten more
``compute_*`` arguments on the forward signatures. Adding one property means
editing several files. Here a property is a row: name it, say what shape it
has, say whether there is one per atom or one per structure, and it is declared.

Three objects, and they do different jobs:

``InputSpec``
    something the model is given -- positions, the cell, a magnetic moment, an
    electronic temperature. Declared so that a derivative can be taken with
    respect to it without new code.

``ObservableSpec``
    something the model produces and a loss can be written against.

``DerivativeSpec``
    derived, never declared: the result of asking an observable for its
    derivative with respect to an input. Its name and sign come from
    :mod:`mace_core.observables.derivatives`.

``ObservableCatalogue`` holds a set of them and is what the defaults file loads
into. Validation lives there rather than in the individual specs because the
interesting errors are between rows: a derivative asked against an input nobody
declared, or two rows whose derived names collide.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mace_core.observables.derivatives import derivative_name, derivative_sign
from mace_core.observables.grammar import (
    IrrepTerm,
    irreps_dimension,
    parse_irreps,
)

__all__ = [
    "NORMALIZATIONS",
    "DerivativeRequest",
    "DerivativeSpec",
    "InputSpec",
    "Normalization",
    "ObservableCatalogue",
    "ObservableSpec",
]

#: How a target is scaled before it reaches a head and a loss term. The
#: per-observable successor of the legacy scaling registry, whose three entries
#: were a global choice for the whole model. This package stores and validates
#: the value; the head applies it in the output layer and the loss applies the
#: matching term, both reading this one field, so there is never a second
#: scaling mechanism to keep in step with it.
Normalization = Literal["none", "std", "rms"]

#: The accepted values, for error messages and for callers that enumerate them.
NORMALIZATIONS: tuple[str, ...] = ("none", "std", "rms")

_SCALAR = (IrrepTerm(multiplicity=1, degree=0, parity="e"),)


def _check_name(value: str, kind: str) -> str:
    if not value.isidentifier():
        raise ValueError(
            f"{kind} name {value!r} is not usable: a name must be a valid "
            f"Python identifier, because it is also the key the value is "
            f"stored under and part of any derivative name derived from it."
        )
    return value


class InputSpec(BaseModel):
    """Something the model is given, and can be differentiated against.

    ``pos`` and ``cell`` are the two every model has. Anything else is declared
    the same way, which is what makes ``d_energy_d_<feature>`` reachable for a
    new feature without touching code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    irreps: str
    #: ``True`` for one value per atom (positions, magnetic moments), ``False``
    #: for one per structure (the cell, a total charge). This is what decides
    #: whether a derivative taken against the input is padded per node or per
    #: graph.
    per_atom: bool
    units: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> InputSpec:
        _check_name(self.name, "input")
        parse_irreps(self.irreps, observable=self.name)
        return self


class DerivativeRequest(BaseModel):
    """A derivative an observable asks for, and the loss settings it carries.

    The name and the sign are not here: they are derived, and letting a
    declaration override them would reintroduce the per-consumer naming this
    abstraction removes. What a declaration does own is what a loss needs --
    its weight, its normalization, and the unit string to report.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The name of the declared input to differentiate against.
    wrt: str
    default_loss_weight: float = Field(default=1.0, ge=0.0)
    normalization: Normalization = "none"
    #: Left to the declaration. Deriving it would mean unit algebra over the
    #: quantity and the input, which this ticket does not own.
    units: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: object) -> object:
        """``derivatives: [pos, cell]`` is the same as spelling out ``wrt``."""
        if isinstance(value, str):
            return {"wrt": value}
        return value


class DerivativeSpec(BaseModel):
    """A derivative, as resolved by the catalogue. Never declared directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    #: The observable being differentiated.
    of: str
    #: The input it is differentiated against.
    wrt: str
    #: ``reported = sign * d(of)/d(wrt)``.
    sign: int
    #: Inherited from the input: a derivative against a per-atom input has one
    #: value per atom, whatever the differentiated quantity is.
    per_atom: bool
    #: Known when the differentiated quantity is a single scalar, in which case
    #: the gradient carries the input's irreps. ``None`` otherwise, because the
    #: general case is a tensor product and the algebra is not this module's.
    irreps: str | None
    units: str | None
    normalization: Normalization
    default_loss_weight: float


class ObservableSpec(BaseModel):
    """One declared property: what it is, what shape it has, how it is scaled.

    Any atomic or total spherical-tensor property declared here becomes
    trainable with no new code: the spec drives the head, the loss term and the
    padding classification.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    #: The spherical-tensor shape, in the grammar of
    #: :mod:`mace_core.observables.grammar`.
    irreps: str
    #: ``True`` for one value per atom, ``False`` for one per structure. Every
    #: output carries this, which is what lets padding be added and removed
    #: without a consumer keeping its own list of which names are which.
    per_atom: bool
    #: Project convention: eV, Å.
    units: str = Field(min_length=1)
    normalization: Normalization
    default_loss_weight: float = Field(default=1.0, ge=0.0)
    #: The derivatives this observable asks for. Naming works for any declared
    #: input whether or not it is listed here; listing it is what says the
    #: model should compute it.
    derivatives: tuple[DerivativeRequest, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> ObservableSpec:
        _check_name(self.name, "observable")
        parse_irreps(self.irreps, observable=self.name)
        seen: set[str] = set()
        for request in self.derivatives:
            if request.wrt in seen:
                raise ValueError(
                    f"observable {self.name!r} asks for the derivative with "
                    f"respect to {request.wrt!r} twice. Declare it once."
                )
            seen.add(request.wrt)
        return self

    @property
    def dimension(self) -> int:
        """The number of components one value of this observable spans."""
        return irreps_dimension(self.irreps, observable=self.name)

    @property
    def is_scalar(self) -> bool:
        """Whether the declaration is a single ``0e``."""
        return parse_irreps(self.irreps, observable=self.name) == _SCALAR

    def derivative_name(self, wrt: str) -> str:
        """The canonical name of this observable's derivative against ``wrt``."""
        return derivative_name(self.name, wrt)

    def derivative_sign(self, wrt: str) -> int:
        """The sign that derivative is reported with."""
        return derivative_sign(self.name, wrt)

    def requested_derivatives(self) -> tuple[str, ...]:
        """The inputs this observable asked to be differentiated against."""
        return tuple(request.wrt for request in self.derivatives)


class ObservableCatalogue(BaseModel):
    """A set of declared inputs and observables, validated against each other."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: tuple[InputSpec, ...] = ()
    observables: tuple[ObservableSpec, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> ObservableCatalogue:
        self._reject_duplicates([spec.name for spec in self.inputs], "input")
        self._reject_duplicates([spec.name for spec in self.observables], "observable")
        declared = {spec.name for spec in self.inputs}
        taken = {spec.name for spec in self.observables}
        for observable in self.observables:
            for request in observable.derivatives:
                if request.wrt not in declared:
                    raise ValueError(
                        f"observable {observable.name!r} asks for its "
                        f"derivative with respect to {request.wrt!r}, which is "
                        f"not a declared input. Declare it under `inputs`, or "
                        f"use one of {sorted(declared)}."
                    )
                name = observable.derivative_name(request.wrt)
                if name in taken:
                    raise ValueError(
                        f"the derivative of {observable.name!r} with respect "
                        f"to {request.wrt!r} is named {name!r}, which is "
                        f"already taken. Rename the observable that holds it, "
                        f"or drop the derivative."
                    )
                taken.add(name)
        return self

    @staticmethod
    def _reject_duplicates(names: list[str], kind: str) -> None:
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(
                    f"{kind} {name!r} is declared twice. Every {kind} name is "
                    f"a key, so it has to be unique."
                )
            seen.add(name)

    def input(self, name: str) -> InputSpec:
        """The declared input called ``name``."""
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(
            f"{name!r} is not a declared input. The declared inputs are "
            f"{sorted(spec.name for spec in self.inputs)}."
        )

    def observable(self, name: str) -> ObservableSpec:
        """The declared observable called ``name``."""
        for spec in self.observables:
            if spec.name == name:
                return spec
        raise KeyError(
            f"{name!r} is not a declared observable. The declared observables "
            f"are {sorted(spec.name for spec in self.observables)}."
        )

    def derivative(self, observable: str, wrt: str) -> DerivativeSpec:
        """Resolve one derivative, whether or not the observable asked for it.

        Naming and signing are properties of the pair, not of the request, so a
        consumer can ask what a derivative *would* be called without the
        declaration having listed it.
        """
        spec = self.observable(observable)
        input_spec = self.input(wrt)
        request = next(
            (r for r in spec.derivatives if r.wrt == wrt), DerivativeRequest(wrt=wrt)
        )
        return DerivativeSpec(
            name=spec.derivative_name(wrt),
            of=spec.name,
            wrt=wrt,
            sign=spec.derivative_sign(wrt),
            per_atom=input_spec.per_atom,
            irreps=input_spec.irreps if spec.is_scalar else None,
            units=request.units,
            normalization=request.normalization,
            default_loss_weight=request.default_loss_weight,
        )

    def requested_derivatives(self) -> tuple[DerivativeSpec, ...]:
        """Every derivative the declarations actually asked for."""
        return tuple(
            self.derivative(spec.name, request.wrt)
            for spec in self.observables
            for request in spec.derivatives
        )

    def names(self) -> tuple[str, ...]:
        """Every name this catalogue puts on a model output."""
        return tuple(spec.name for spec in self.observables) + tuple(
            spec.name for spec in self.requested_derivatives()
        )
