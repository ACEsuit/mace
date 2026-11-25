"""Config sections of the model schema.

The radial embedding, one section each for the radial basis, the distance
transform and the cutoff envelope, and the model section that holds them with
the backbone's architecture. A field of several kinds is a union tagged
by `kind` (`radial_basis: {kind: gaussian, num_basis: 16}`); "no transform" is
the kind `none`, not an absent key. The envelope has one kind, so its section
is a plain section without a tag.

The model section also holds the observables the model reads out and how
(`observables`, `readout`), and, per head, how that head's interaction energy
is scaled and shifted (`scale_shift`). It holds the methods only: the scale and
the shift are computed from the data before the model is built and are saved
with the weights.

The cutoff radius `r_max` is in none of them. The basis, the envelope and the
neighbour list must all use the same radius, so it is one model-level field,
handed to each section's module when the model is built.

Defaults are the legacy command-line defaults (`--num_radial_basis 8`,
`--num_cutoff_basis 5`), which differ from some of the legacy class defaults.

A section with `trainable` set gives the starting value of what it makes
trainable. The trained value is saved with the weights, and is what a loaded
model computes with.
"""

import logging
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mace_core.clebsch_gordan.irreps import Irrep, Irreps, IrrepsError
from mace_core.config.base import ConfigSection
from mace_core.heads import DEFAULT_HEAD, HeadName
from mace_core.kernels.canonical import gate_irreps_problem
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    ObservableCatalogue,
    check_observable_name,
    declared_as,
)
from mace_core.outputs import CORE_FIELD_NAMES, FIELD_BY_OBSERVABLE

logger = logging.getLogger(__name__)

__all__ = [
    "AgnesiTransformConfig",
    "BesselBasisConfig",
    "ChebyshevBasisConfig",
    "DistanceTransformConfig",
    "GaussianBasisConfig",
    "InteractionKind",
    "ModelConfig",
    "NoDistanceTransformConfig",
    "PolynomialCutoffConfig",
    "RadialBasisConfig",
    "ReadoutConfig",
    "ScaleShiftSpec",
    "ScalingMethod",
    "ShiftMethod",
    "SoftTransformConfig",
]

# ---------------------------------------------------------------------------
# Radial basis
# ---------------------------------------------------------------------------


class BesselBasisConfig(ConfigSection):
    """Spherical Bessel functions of order zero, the default basis."""

    kind: Literal["bessel"] = "bessel"
    num_basis: int = Field(default=8, ge=1)
    #: Learn the frequencies instead of fixing them at `n * pi / r_max`.
    trainable: bool = False


class GaussianBasisConfig(ConfigSection):
    """Gaussians on evenly spaced centres in `[0, r_max]`."""

    kind: Literal["gaussian"] = "gaussian"
    #: At least two: the width is the spacing `r_max / (num_basis - 1)`.
    num_basis: int = Field(default=8, ge=2)
    #: Learn the centres instead of fixing them.
    trainable: bool = False


class ChebyshevBasisConfig(ConfigSection):
    """Chebyshev polynomials of the raw length; the input is not rescaled by
    `r_max`, so this basis ignores it."""

    kind: Literal["chebyshev"] = "chebyshev"
    num_basis: int = Field(default=8, ge=1)
    include_constant: bool = False  #: Start at `T_0 = 1` instead of `T_1`.


RadialBasisConfig = Annotated[
    BesselBasisConfig | GaussianBasisConfig | ChebyshevBasisConfig,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------


class NoDistanceTransformConfig(ConfigSection):
    """The basis sees the raw edge lengths."""

    kind: Literal["none"] = "none"


class AgnesiTransformConfig(ConfigSection):
    """The Agnesi transform of ACEpotentials.jl, `1 / (1 + a y^q / (1 + y^(q-p)))`
    with `y` the length over half the pair's covalent radii sum. Defaults are
    the paper's fitted values."""

    kind: Literal["agnesi"] = "agnesi"
    exponent_q: float = 0.9183
    exponent_p: float = 4.5791
    amplitude: float = 1.0805
    trainable: bool = False


class SoftTransformConfig(ConfigSection):
    """A tanh clamp that flattens lengths below three quarters of the pair's
    covalent radii sum."""

    kind: Literal["soft"] = "soft"
    #: Dimensionless; divided by the width of the switching window.
    steepness: float = 4.0
    trainable: bool = False


DistanceTransformConfig = Annotated[
    NoDistanceTransformConfig | AgnesiTransformConfig | SoftTransformConfig,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Cutoff envelope
# ---------------------------------------------------------------------------


class PolynomialCutoffConfig(ConfigSection):
    """The polynomial envelope: 1 at `r = 0`, zero with two vanishing
    derivatives at `r_max`, exactly zero beyond. The only envelope there is."""

    polynomial_order: int = Field(default=5, ge=1)


# ---------------------------------------------------------------------------
# Scale and shift
# ---------------------------------------------------------------------------

#: What a head's scale is taken from.
#:
#: * `std`: the standard deviation of the per-atom interaction energy.
#: * `rms_forces`: the root mean square of the force components.
#: * `rms_dipoles`: the root mean square of the dipole components.
#: * `none`: a scale of one.
ScalingMethod = Literal["std", "rms_forces", "rms_dipoles", "none"]

#: What its shift is: the mean per-atom interaction energy, or zero.
ShiftMethod = Literal["mean", "none"]

#: The statistic each method is taken from, by the name the dataset statistics
#: give it; a method without one is a scale of one or a shift of zero.
_STATISTIC_OF: dict[str, str] = {
    "std": "energy_std",
    "rms_forces": "rms_forces",
    "rms_dipoles": "rms_dipoles",
    "mean": "energy_mean",
}


class ScaleShiftSpec(ConfigSection):
    """How one head's interaction energy is scaled and shifted.

    The methods only. The two numbers they give are computed from the head's
    training data by :meth:`scale_and_shift` before the model is built, and
    the model saves them with its weights.

    The default is the legacy command line's: `--model MACE` scales by the
    force RMS and does not shift. `--model ScaleShiftMACE` is
    `shift: mean`.
    """

    scale: ScalingMethod = "rms_forces"
    shift: ShiftMethod = "none"

    @model_validator(mode="after")
    def _a_dipole_scale_takes_no_energy_shift(self) -> "ScaleShiftSpec":
        if self.scale == "rms_dipoles" and self.shift != "none":
            raise ValueError(
                "scale 'rms_dipoles' scales a dipole, which a per-atom energy "
                "cannot shift; set shift to 'none'."
            )
        return self

    def scale_and_shift(self, **statistics: float | None) -> tuple[float, float]:
        """``(scale, shift)`` from one head's measured statistics.

        Args:
            statistics: By the dataset statistics' names: ``energy_mean`` and
                ``energy_std`` in eV/atom, ``rms_forces`` in eV/A,
                ``rms_dipoles``. Absent or ``None`` when the data does not
                give one.

        A zero scale becomes one, with a warning: a set whose spread is zero
        gives nothing to scale by, and dividing the targets by it would not be
        a model.

        Raises:
            ValueError: If a method needs a statistic that was not given,
                which is what a head without the matching labels measures.
        """
        needed = [
            _STATISTIC_OF[method]
            for method in (self.scale, self.shift)
            if method in _STATISTIC_OF
        ]
        missing = [name for name in needed if statistics.get(name) is None]
        if missing:
            raise ValueError(
                f"scale_shift (scale {self.scale!r}, shift {self.shift!r}) needs "
                f"{missing}, which the head's data did not provide. Label the "
                f"data, or choose methods it can provide."
            )
        scale = statistics.get(_STATISTIC_OF.get(self.scale, ""))
        shift = statistics.get(_STATISTIC_OF.get(self.shift, ""))
        scale = 1.0 if scale is None else float(scale)
        if scale == 0.0:
            logger.warning(
                "scale_shift scale %r measured zero; using a scale of one.",
                self.scale,
            )
            scale = 1.0
        return scale, 0.0 if shift is None else float(shift)


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

_PER = {True: "per atom", False: "per structure"}


def _parse_irreps(value: str, setting: str) -> Irreps:
    """``value`` parsed, an error naming the setting it was given for."""
    try:
        return Irreps.parse(value)
    except IrrepsError as error:
        raise ValueError(f"{setting}={value!r}: {error}") from None


class ReadoutConfig(ConfigSection):
    """How each declared observable is read out of the node features, and
    reported.

    Every layer is read out linearly and the readouts are summed, except the
    last layer of a model with more than one, which is read out through a
    gated middle: a linear map to the terms of ``mlp_irreps`` whose irrep the
    readout produces, a SiLU on its scalars and each higher copy scaled by the
    SiLU of a scalar of its own, and a linear map to the readout's output. A
    middle left with none of the output's irreps gives that layer no readout.
    """

    #: The gated middle's irreps per head, the legacy `--MLP_irreps`, sorted
    #: by degree, odd before even at one degree. Scalars are even.
    mlp_irreps: str = "16x0e"
    #: Read out only the last layer, the legacy `--use_last_readout_only`.
    last_only: bool = False
    #: Observables read out together, by one readout producing all of them.
    #: Within each irrep their copies follow the order of the group, so
    #: `[charges, polarizability, dipole]` is the frozen dielectric readout's
    #: `2x0e+1o+2e`: the charge, the polarizability's trace, the dipole and
    #: the polarizability's traceless part. An observable is in one group at
    #: most.
    groups: tuple[tuple[str, ...], ...] = ()
    #: Where the dipole's charge part, ``sum_i q_i r_i``, comes from, as the
    #: frozen tree's dipole models add it:
    #:
    #: * `none`: no charge part.
    #: * `input`: the `charges` input, the structure's own, converted from
    #:   e*Angstrom to Debye by :data:`~mace_core.units.DEBYE_FROM_SI`.
    #: * `predicted`: the `charges` observable, every atom of a structure
    #:   shifted by one constant so they sum to the `total_charge` input, in
    #:   e*Angstrom.
    dipole_charges: Literal["none", "input", "predicted"] = "none"
    #: `0e+2e` observables of a structure reported as the symmetric matrix
    #: they are, under their own name, and as their six components under
    #: `<name>_sh`.
    symmetric_matrices: tuple[str, ...] = ()
    #: Structure observables whose per-atom values are reported as well, by
    #: observable, under the given name.
    atomic_contributions: dict[str, str] = Field(default_factory=dict)

    @field_validator("atomic_contributions")
    @classmethod
    def _contributions_are_reported_under_usable_names(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        for name in value.values():
            check_observable_name(name, "readout.atomic_contributions")
        return value

    @model_validator(mode="after")
    def _the_middle_is_sorted_with_even_scalars(self) -> "ReadoutConfig":
        problem = gate_irreps_problem(_parse_irreps(self.mlp_irreps, "mlp_irreps"))
        if problem:
            raise ValueError(f"mlp_irreps={self.mlp_irreps!r} {problem}.")
        return self

    @model_validator(mode="after")
    def _groups_are_disjoint(self) -> "ReadoutConfig":
        members = [name for group in self.groups for name in group]
        repeated = sorted({name for name in members if members.count(name) > 1})
        if repeated:
            raise ValueError(f"{repeated} are in more than one readout group.")
        short = [list(group) for group in self.groups if len(group) < 2]
        if short:
            raise ValueError(f"a readout group has two members at least, got {short}.")
        return self


#: What the energy head writes besides the total energy.
_ENERGY_WRITES = ("node_energies", "interaction_energy")


def _check_destinations(catalogue: ObservableCatalogue, readout: ReadoutConfig) -> None:
    """Refuse a value that would land where another one does.

    The energy head fills ``node_energies`` and ``extras["interaction_energy"]``,
    every observable and every derivative asked for fills its own name, a
    symmetric matrix also ``<name>_sh``, and a per-atom contribution the name
    it is given. A core
    field of the output has one shape. Two values with one destination would
    leave the one written last, which depends on the order of the
    declarations.
    """
    destinations: dict[str, str] = {}

    def claim(name: str, by: str) -> None:
        if name in destinations:
            raise ValueError(
                f"{by} and {destinations[name]} would both be reported as "
                f"{name!r}; declare one of them under another name."
            )
        destinations[name] = by

    specs = {spec.name: spec for spec in catalogue.observables}
    if "energy" in specs:
        for name in _ENERGY_WRITES:
            claim(name, "the energy")
    for derivative in catalogue.requested_derivatives():
        claim(derivative.name, f"the derivative {derivative.name!r}")
    for name, spec in specs.items():
        claim(name, f"the observable {name!r}")
        if name in readout.symmetric_matrices:
            claim(f"{name}_sh", f"the components of {name!r}")
        if name in readout.atomic_contributions:
            claim(readout.atomic_contributions[name], f"the atoms' {name!r}")
        if name == "energy":
            continue
        field = FIELD_BY_OBSERVABLE.get(name, name)
        if field not in CORE_FIELD_NAMES:
            continue
        if field == "dipole" and declared_as(spec, "1o", per_atom=False):
            continue
        raise ValueError(
            f"the observable {name!r} would fill the output's {field!r} field, "
            f"which holds "
            + (
                "one 1o vector per structure"
                if field == "dipole"
                else "what the energy or its derivatives give"
            )
            + f"; it is declared {spec.irreps!r}, "
            f"{_PER[spec.per_atom]}. Declare it to "
            f"match, or under another name."
        )
    for name in list(readout.atomic_contributions.values()) + [
        f"{name}_sh" for name in readout.symmetric_matrices
    ]:
        if FIELD_BY_OBSERVABLE.get(name, name) in CORE_FIELD_NAMES:
            raise ValueError(
                f"{name!r} is a field of the output, and a reported value "
                f"cannot take it; choose another name."
            )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

#: The interaction blocks. Each is a different function and a trained model
#: depends on which one each layer has. The first layer reads only the element
#: embedding, and published models choose its block separately, so it can be
#: set apart from the rest.
#:
#: * `plain` applies the skip connection to the message, which replaces it.
#: * `residual` takes the skip from the layer's input and adds it after the
#:   product basis.
#: * `density` and `density_residual` are those two, dividing the messages by
#:   a density learned per atom instead of the average neighbour count.
#: * `nonlinear` conditions its radial network on both elements, normalizes
#:   by a learned density with two trained scalars, and ends in a gate. It
#:   carries its skip to the product basis, in the first layer too.
InteractionKind = Literal[
    "plain", "residual", "density", "density_residual", "nonlinear"
]


class ModelConfig(ConfigSection):
    """The model: the backbone's architecture, what it reads out and how,
    and each head's scale and shift.

    Defaults are the legacy command-line defaults. The node features of every
    layer but the last carry `num_channels` copies of each irrep up to `max_L`
    with natural parity, `0e+1o+2e+...`; the last carries `num_channels`
    copies of the irreps `last_layer_irreps` names.
    """

    #: The cutoff radius in Angstrom, shared by the radial basis, the envelope
    #: and the neighbour list.
    r_max: float = Field(default=5.0, gt=0.0)
    num_interactions: int = Field(default=2, ge=1)
    num_channels: int = Field(default=128, ge=1)
    #: The highest degree of the node features.
    max_L: int = Field(default=1, ge=0)
    #: The highest degree of the edge spherical harmonics.
    max_ell: int = Field(default=3, ge=0)
    #: The body order of the product basis.
    correlation: int = Field(default=3, ge=1)
    radial_basis: RadialBasisConfig = BesselBasisConfig()
    distance_transform: DistanceTransformConfig = NoDistanceTransformConfig()
    cutoff: PolynomialCutoffConfig = PolynomialCutoffConfig()
    #: Multiply the envelope into the radial basis. Off, every interaction
    #: multiplies it into what its radial networks produce instead.
    apply_cutoff: bool = True
    #: The hidden widths of every interaction's radial network.
    radial_mlp: tuple[int, ...] = (64, 64, 64)
    #: The interaction block of every layer, the first included unless
    #: `interaction_first` names another.
    interaction: InteractionKind = "residual"
    #: The interaction block of the first layer, when it differs from
    #: `interaction`.
    interaction_first: InteractionKind | None = None
    #: What a `nonlinear` layer convolves, a whole declaration with one
    #: multiplicity for every term, such as `128x0e+128x1o`. Unset, the node
    #: features at their own width.
    edge_irreps: str | None = None
    #: Convolve only the scalars of `edge_irreps` in a `nonlinear` first layer.
    use_edge_irreps_first: bool = False
    #: Share the product basis's weights across elements.
    use_agnostic_product: bool = False
    #: The Clebsch-Gordan basis the symmetric contraction's weights are written
    #: against. Recorded with the model and never inferred from what is
    #: installed; `full` builds only on a backend that declares it.
    clebsch_gordan_basis: Literal["reduced", "full"] = "reduced"
    #: One channel of the last layer's node features: its scalars by
    #: default, which is what an energy needs. A dipole model keeps its
    #: vectors, and a dielectric one every degree, the legacy
    #: `keep_last_layer_irreps`.
    last_layer_irreps: str = "0e"
    #: What the model computes: the observables it reads out and the inputs
    #: they are differentiated against.
    observables: ObservableCatalogue = DEFAULT_CATALOGUE
    #: How the observables are read out.
    readout: ReadoutConfig = ReadoutConfig()
    #: How each head's interaction energy is scaled and shifted, by head name.
    scale_shift: dict[HeadName, ScaleShiftSpec] = Field(
        default_factory=lambda: {DEFAULT_HEAD: ScaleShiftSpec()}, min_length=1
    )

    @model_validator(mode="after")
    def _the_last_layer_keeps_node_irreps(self) -> "ModelConfig":
        hidden = Irreps.spherical_harmonics(self.max_L)
        kept = _parse_irreps(self.last_layer_irreps, "last_layer_irreps")
        if kept.terms != tuple((1, ir) for _, ir in hidden.terms if ir in kept):
            raise ValueError(
                f"last_layer_irreps={self.last_layer_irreps!r} is not a sorted "
                f"choice of one copy each of the node irreps {str(hidden)!r}."
            )
        return self

    @model_validator(mode="after")
    def _the_readout_reads_what_is_declared(self) -> "ModelConfig":
        readout = self.readout

        def need(
            setting: str,
            name: str,
            *,
            per_atom: bool | None = None,
            irreps: str | None = None,
            kind: Literal["observable", "input"] = "observable",
        ) -> None:
            try:
                spec = getattr(self.observables, kind)(name)
            except KeyError as error:
                raise ValueError(f"readout.{setting}: {error.args[0]}") from None
            if per_atom is None:
                return
            if irreps is None:
                if spec.per_atom == per_atom:
                    return
            elif declared_as(spec, irreps, per_atom=per_atom):
                return
            shape = f"{irreps!r} {_PER[per_atom]}" if irreps else _PER[per_atom]
            raise ValueError(
                f"readout.{setting} reads the {kind} {name!r}, which is not "
                f"declared {shape}; declare it so in observables."
            )

        for group in readout.groups:
            for name in group:
                need("groups", name)
        if readout.dipole_charges != "none":
            need("dipole_charges", "dipole", irreps="1o", per_atom=False)
        if readout.dipole_charges == "input":
            need("dipole_charges", "charges", irreps="0e", per_atom=True, kind="input")
        if readout.dipole_charges == "predicted":
            need("dipole_charges", "charges", irreps="0e", per_atom=True)
            need(
                "dipole_charges",
                "total_charge",
                irreps="0e",
                per_atom=False,
                kind="input",
            )
        if readout.dipole_charges != "none":
            in_debye = readout.dipole_charges == "input"
            units = self.observables.observable("dipole").units
            if (units == "Debye") != in_debye:
                raise ValueError(
                    f"readout.dipole_charges {readout.dipole_charges!r} adds a "
                    f"charge part in {'Debye' if in_debye else 'e*Å'}, and "
                    f"'dipole' is declared in {units!r}; declare it in "
                    f"{'Debye' if in_debye else 'e*Å, not Debye'}."
                )
        for name in readout.symmetric_matrices:
            need("symmetric_matrices", name, irreps="0e+2e", per_atom=False)
        for name in readout.atomic_contributions:
            if name == "energy":
                raise ValueError(
                    "readout.atomic_contributions names 'energy', whose "
                    "per-atom values are node_energies already."
                )
            need("atomic_contributions", name, per_atom=False)
        return self

    @model_validator(mode="after")
    def _every_value_is_reported_under_its_own_name(self) -> "ModelConfig":
        _check_destinations(self.observables, self.readout)
        return self

    @model_validator(mode="after")
    def _edge_irreps_need_a_nonlinear_layer(self) -> "ModelConfig":
        """`edge_irreps` and `use_edge_irreps_first` are read by `nonlinear`
        layers only, and a setting nothing reads is refused rather than kept."""
        later_nonlinear = self.num_interactions > 1 and self.interaction == "nonlinear"
        first_nonlinear = self.interaction_of(0) == "nonlinear"
        first_reads = first_nonlinear and self.use_edge_irreps_first
        if self.edge_irreps is not None and not (later_nonlinear or first_reads):
            raise ValueError(
                "edge_irreps is set but no layer reads it: a 'nonlinear' layer "
                "after the first does, and a 'nonlinear' first layer does with "
                "use_edge_irreps_first."
            )
        if self.use_edge_irreps_first and (
            self.edge_irreps is None or not first_nonlinear
        ):
            raise ValueError(
                "use_edge_irreps_first needs edge_irreps and a 'nonlinear' first layer."
            )
        if self.edge_irreps is not None:
            edge_irreps = _parse_irreps(self.edge_irreps, "edge_irreps")
            multiplicities = sorted({mul for mul, _ in edge_irreps.terms})
            if len(multiplicities) != 1:
                raise ValueError(
                    f"edge_irreps={self.edge_irreps!r} has the multiplicities "
                    f"{multiplicities}; every term needs the same one, such as "
                    f"'128x0e+128x1o', since the convolution has one radial "
                    f"weight per path and channel."
                )
            if self.use_edge_irreps_first and not edge_irreps.multiplicity(Irrep(0, 1)):
                raise ValueError(
                    f"use_edge_irreps_first convolves the scalars of edge_irreps, "
                    f"and edge_irreps={self.edge_irreps!r} has no '0e' term."
                )
        return self

    def interaction_of(self, layer: int) -> InteractionKind:
        """The interaction block layer `layer` builds, counting from 0."""
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer is an integer, not {layer!r}.")
        if not 0 <= layer < self.num_interactions:
            raise ValueError(
                f"layer {layer} is not one of the {self.num_interactions} layers "
                f"of this model, 0 to {self.num_interactions - 1}."
            )
        if layer == 0 and self.interaction_first is not None:
            return self.interaction_first
        return self.interaction
