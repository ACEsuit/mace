"""Declaring an observable, an input, and the derivatives between them.

Which quantities a legacy model computes is decided by a ladder of string
comparisons on the model class name that sets six boolean flags, plus ten more
``compute_*`` arguments on the forward signatures. Adding one property means
editing several files. Here a property is a row: name it, say what shape it
has, say whether there is one per atom or one per structure, and it is declared.

Three objects, and they do different jobs:

``InputSpec``
    a leaf a derivative can be taken against: positions, the strain, a magnetic
    moment, an electronic temperature. Declaring one is what makes its
    derivative reachable without new code.

``ObservableSpec``
    something the model produces and a loss can be written against.

``DerivativeSpec``
    derived, never declared: the result of asking an observable for its
    derivative with respect to an input. Its name and sign are the ones the
    observable's :class:`DerivativeRequest` declares, and otherwise the
    ``d_<q>_d_<x>`` rule of :mod:`mace_core.observables.derivatives` with the
    gradient's own sign.

``ObservableCatalogue`` holds a set of inputs and observables;
:data:`~mace_core.observables.DEFAULT_CATALOGUE` is one, declared in
:mod:`mace_core.observables.defaults`. Validation of a single row lives on the
row, and validation between rows lives on the catalogue: a derivative asked
against an input nobody declared, a per-atom quantity differentiated against a
per-atom input, or two names that collide.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from mace_core.observables.derivatives import (
    DEFAULT_SIGN,
    default_derivative_name,
    is_default_shaped_name,
)
from mace_core.observables.grammar import (
    IrrepTerm,
    irreps_dimension,
    parse_irreps,
)
from mace_core.outputs import OBSERVABLE_BY_FIELD, RETIRED_NAMES

__all__ = [
    "DerivativeRequest",
    "DerivativeSpec",
    "InputSpec",
    "ObservableCatalogue",
    "ObservableSpec",
]

_SCALAR = (IrrepTerm(multiplicity=1, degree=0, parity="e"),)


#: Names :class:`~mace_core.outputs.MACEOutput` uses for storage but that are
#: not observable names, mapped to the name to declare instead. Declaring one
#: would pass here and fail only when a model filled ``extras`` under it.
_STORAGE_ONLY_NAMES: dict[str, str] = {**OBSERVABLE_BY_FIELD, **RETIRED_NAMES}


def _check_name(value: str, kind: str) -> str:
    if not value.isidentifier():
        raise ValueError(
            f"{kind} name {value!r} is not usable: a name must be a valid "
            f"Python identifier, because it is also the key the value is "
            f"stored under and part of any derivative name derived from it."
        )
    return value


def _check_output_name(value: str, kind: str) -> str:
    """Check a name that a model output is stored under."""
    _check_name(value, kind)
    if value in _STORAGE_ONLY_NAMES:
        raise ValueError(
            f"{kind} name {value!r} is not usable: MACEOutput reserves it, "
            f"and a value declared under it could not be stored. Use "
            f"{_STORAGE_ONLY_NAMES[value]!r} instead."
        )
    return value


class InputSpec(BaseModel):
    """Something a derivative can be taken against.

    ``pos`` and ``strain`` are the two every model has. Anything else is
    declared the same way, which is what makes ``d_energy_d_<feature>``
    reachable for a new feature without touching code.

    An input is a leaf of the derivative graph, not necessarily a field read
    from the data. ``pos`` is both. ``strain`` is only the first: the
    derivative engine materialises it as zeros around the model call and
    applies it to the positions and the cell, so nothing reads a strain from a
    dataset and none is stored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    irreps: str
    #: ``True`` for one value per atom (positions, magnetic moments), ``False``
    #: for one per structure (the strain, a total charge). This is what decides
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
    """A derivative an observable asks for.

    Both the name and the sign belong to the declaration, so a derivative with
    a name of its own, ``forces`` or ``magforces``, is added without editing
    this package. The name is decided **once**, here, and every consumer reads
    it off the resolved spec, so nothing about this lets a consumer invent one.

    Giving a ``name`` obliges the declaration to give a ``sign`` too. A renamed
    derivative that silently inherited ``+1`` is a model trained on inverted
    forces that runs perfectly well, which is a failure nothing downstream can
    see.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The name of the declared input to differentiate against.
    wrt: str
    #: What the derivative is called. ``None`` takes the grammar's own
    #: ``d_<quantity>_d_<input>``.
    name: str | None = None
    #: ``reported = sign * d(quantity)/d(input)``, either ``+1`` or ``-1``.
    #: ``None`` takes the gradient's own sign. Only the integers ``1`` and
    #: ``-1`` are accepted: ``True``, ``1.0`` and ``"-1"`` are refused rather
    #: than coerced into a sign.
    sign: int | None = None
    #: Left to the declaration. Deriving it would mean unit algebra over the
    #: quantity and the input, which this ticket does not own.
    units: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: object) -> object:
        """``derivatives: [pos, strain]`` is the same as spelling out ``wrt``."""
        if isinstance(value, str):
            return {"wrt": value}
        return value

    @field_validator("sign", mode="before")
    @classmethod
    def _check_sign(cls, value: object, info: ValidationInfo) -> object:
        # `type(...) is int` rather than `isinstance`: `bool` is an `int`.
        if value is not None and (type(value) is not int or value not in (1, -1)):
            raise ValueError(
                f"the derivative with respect to {info.data.get('wrt')!r} "
                f"declares sign {value!r}. A sign is the integer +1 or -1; a "
                f"scale factor is not a sign and belongs to whatever computes "
                f"the quantity."
            )
        return value

    @model_validator(mode="after")
    def _validate(self) -> DerivativeRequest:
        if self.name is not None:
            _check_output_name(self.name, "derivative")
            if self.sign is None:
                raise ValueError(
                    f"the derivative with respect to {self.wrt!r} is named "
                    f"{self.name!r} but declares no sign. A name of its own "
                    f"means a convention of its own, so state it: `sign: -1` "
                    f"for a quantity reported as the negative gradient, "
                    f"`sign: +1` otherwise."
                )
            if is_default_shaped_name(self.name):
                raise ValueError(
                    f"the derivative with respect to {self.wrt!r} is named "
                    f"{self.name!r}, which is spelled like the grammar's own "
                    f"`d_<quantity>_d_<input>`. That spelling states which "
                    f"quantity was differentiated, so a custom name must not "
                    f"use it. Drop `name` to get the generated one."
                )
        return self


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
    #: ``True`` when either side is per-atom. A per-graph quantity against a
    #: per-atom input (the energy against the positions) has one value per
    #: atom, and so does a per-atom quantity against a per-graph input (a
    #: per-atom energy against the strain). A per-atom quantity against a
    #: per-atom input is an ``(n_atoms, n_atoms, ...)`` block, which neither
    #: classification describes, and the catalogue refuses it.
    per_atom: bool
    #: Known when the differentiated quantity is a single scalar, in which case
    #: the gradient carries the input's irreps. ``None`` otherwise, because the
    #: general case is a tensor product and the algebra is not this module's.
    irreps: str | None
    units: str | None


class ObservableSpec(BaseModel):
    """One declared property: what it is and what shape it has.

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
    #: The derivatives this observable asks for. Naming works for any declared
    #: input whether or not it is listed here; listing it is what says the
    #: model should compute it.
    derivatives: tuple[DerivativeRequest, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> ObservableSpec:
        _check_output_name(self.name, "observable")
        if is_default_shaped_name(self.name):
            raise ValueError(
                f"observable name {self.name!r} is spelled like a generated "
                f"derivative name, `d_<quantity>_d_<input>`, which states that "
                f"it is the derivative of a declared quantity. If it is one, "
                f"request it in that quantity's `derivatives`; otherwise "
                f"rename it."
            )
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

    def _request(self, wrt: str) -> DerivativeRequest | None:
        """This observable's declared request against ``wrt``, if it made one.

        Naming works for any declared input whether or not it was requested, so
        this is allowed to find nothing.
        """
        for request in self.derivatives:
            if request.wrt == wrt:
                return request
        return None

    def derivative_name(self, wrt: str) -> str:
        """The canonical name of this observable's derivative against ``wrt``."""
        request = self._request(wrt)
        if request is not None and request.name is not None:
            return request.name
        return default_derivative_name(self.name, wrt)

    def derivative_sign(self, wrt: str) -> int:
        """The sign that derivative is reported with."""
        request = self._request(wrt)
        if request is not None and request.sign is not None:
            return request.sign
        return DEFAULT_SIGN

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
        for observable in self.observables:
            if observable.name in declared:
                raise ValueError(
                    f"observable {observable.name!r} has the name of a declared "
                    f"input. An input and an output cannot share a key; rename "
                    f"one of them."
                )
        taken = declared | {spec.name for spec in self.observables}
        for observable in self.observables:
            for request in observable.derivatives:
                if request.wrt not in declared:
                    raise ValueError(
                        f"observable {observable.name!r} asks for its "
                        f"derivative with respect to {request.wrt!r}, which is "
                        f"not a declared input. Declare it under `inputs`, or "
                        f"use one of {sorted(declared)}."
                    )
                _derivative_per_atom(observable, self.input(request.wrt))
                name = observable.derivative_name(request.wrt)
                if name in taken:
                    raise ValueError(
                        f"the derivative of {observable.name!r} with respect "
                        f"to {request.wrt!r} is named {name!r}, which is "
                        f"already the name of a declared input, observable or "
                        f"derivative. Rename one of them, or drop the "
                        f"derivative."
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

        Raises:
            KeyError: If the observable or the input is not declared.
            ValueError: If both are per-atom, which is not representable.
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
            per_atom=_derivative_per_atom(spec, input_spec),
            irreps=input_spec.irreps if spec.is_scalar else None,
            units=request.units,
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


def _derivative_per_atom(observable: ObservableSpec, input_spec: InputSpec) -> bool:
    """Whether ``d(observable)/d(input)`` has one value per atom.

    Raises:
        ValueError: If both are per-atom. Each atom's value then depends on
            every atom's input, an ``(n_atoms, n_atoms, ...)`` block that is
            neither per-atom nor per-graph.
    """
    if observable.per_atom and input_spec.per_atom:
        raise ValueError(
            f"the derivative of the per-atom observable {observable.name!r} "
            f"with respect to the per-atom input {input_spec.name!r} has one "
            f"block per pair of atoms, shape (n_atoms, n_atoms, ...), which "
            f"this specification cannot represent. Differentiate a per-graph "
            f"quantity against {input_spec.name!r}, or differentiate "
            f"{observable.name!r} against a per-graph input such as the strain."
        )
    return observable.per_atom or input_spec.per_atom
