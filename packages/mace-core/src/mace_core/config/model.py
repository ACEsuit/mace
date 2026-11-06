"""Config sections of the model schema.

The radial embedding, one section each for the radial basis, the distance
transform and the cutoff envelope, and the model section that holds them with
the backbone's architecture. A field of several kinds is a union tagged
by `kind` (`radial_basis: {kind: gaussian, num_basis: 16}`); "no transform" is
the kind `none`, not an absent key. The envelope has one kind, so its section
is a plain section without a tag.

The cutoff radius `r_max` is in none of them. The basis, the envelope and the
neighbour list must all use the same radius, so it is one model-level field,
handed to each section's module when the model is built.

Defaults are the legacy command-line defaults (`--num_radial_basis 8`,
`--num_cutoff_basis 5`), which differ from some of the legacy class defaults.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from mace_core.config.base import ConfigSection
from mace_core.elements.default_keys import Storage

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
# The model
# ---------------------------------------------------------------------------

#: The interaction blocks. Each is a different function and a trained model
#: depends on which one each layer has, so the first layer and the rest are
#: named separately.
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


ScalingMethod = Literal["none", "std", "rms_forces", "rms_dipoles"]


class NoScaling(ConfigSection):
    kind: Literal["none"] = "none"


class StandardDeviationScaling(ConfigSection):
    kind: Literal["std"] = "std"


class ForceRMSScaling(ConfigSection):
    kind: Literal["rms_forces"] = "rms_forces"


class DipoleRMSScaling(ConfigSection):
    kind: Literal["rms_dipoles"] = "rms_dipoles"


# Recipes only: measured values belong to the output layer's buffers.
ScaleShiftSpec = Annotated[
    NoScaling | StandardDeviationScaling | ForceRMSScaling | DipoleRMSScaling,
    Field(discriminator="kind"),
]
ClebschGordanBasis = Literal["reduced", "full"]


class ReadoutConfig(ConfigSection):
    """The shape of the readouts, which is a head property and not a top knob.

    ``MLP_irreps`` is the hidden width of the last readout's own MLP, so it
    belongs to the readout rather than beside the interaction widths where
    legacy keeps it.

    Args:
        mlp_irreps: Hidden irreps of the final non-linear readout.
        gate: The readout activation; ``"None"`` leaves scalars and gates linear.
        last_only: Read out only from the last interaction layer, instead of
            summing a readout per layer.
        from_embedding: Add a readout on the node embedding, before any
            interaction.
        bias: Give the last readout a biased middle map, gated again, and a
            biased output map.
    """

    mlp_irreps: str = "16x0e"
    gate: Literal["silu", "tanh", "abs", "None"] = "silu"
    last_only: bool = False
    from_embedding: bool = False
    bias: bool = False


class GraphFeatureConfig(ConfigSection):
    """One input embedded beside the elements, such as the total charge.

    Args:
        name: The graph key it is read from.
        kind: ``categorical`` for a small integer label, ``continuous`` for a
            real value.
        embedding_dim: The width it is embedded to.
        num_classes: How many labels, for a categorical input.
        input_dim: How many components, for a continuous input.
        per: ``graph`` for one value per structure, ``atom`` for one per atom.
        offset: Added to a categorical value before it is used as a label.
        use_bias: Whether a continuous input's maps carry biases.
    """

    name: str
    kind: Literal["categorical", "continuous"]
    embedding_dim: int = Field(ge=1)
    num_classes: int = Field(default=0, ge=0)
    input_dim: int = Field(default=1, ge=1)
    per: Literal["graph", "atom"] = "graph"
    offset: int = 0
    use_bias: bool = True

    @model_validator(mode="after")
    def _valid_input(self) -> GraphFeatureConfig:
        from math import prod

        from mace_core.graph import GRAPH_SCHEMA, INPUT_KEYS
        from mace_core.graph.collate import feature_levels
        from mace_core.graph.schema import NODE_LEVEL

        if not self.name or "." in self.name:
            raise ValueError("graph feature name must be nonempty and contain no '.'")
        if self.kind == "categorical" and not self.num_classes:
            raise ValueError(f"categorical feature {self.name!r} needs num_classes")
        if self.name in INPUT_KEYS:
            shape = GRAPH_SCHEMA[self.name].shape
            storage = "atom" if shape[0] == NODE_LEVEL else "graph"
            if self.per != storage:
                raise ValueError(f"feature {self.name!r} must be per {storage}")
            if all(isinstance(size, int) for size in shape[1:]):
                width = prod(int(size) for size in shape[1:])
                expected = self.input_dim if self.kind == "continuous" else 1
                if expected != width:
                    raise ValueError(f"feature {self.name!r} needs {width} components")
        else:
            feature_levels({self.name: self.per})
        return self


class PolarConfig(ConfigSection):
    """The charge-aware part of the ``polar`` model.

    Read only when ``model`` is ``polar``. The solver the long-range ops come
    from, the systems they are set up for and the reciprocal-space cutoff are
    the ``electrostatics`` section's, since they are what a solver is asked
    for; what is here is the architecture around them. The defaults are the
    frozen tree's command line's.

    Args:
        multipole_max_l: The highest multipole order of each atom's density.
        multipole_width: The Gaussian width of that density, in Angstrom.
        feature_max_l: The highest order the potential is projected onto.
        feature_widths: The Gaussian widths it is projected onto, in Angstrom.
        feature_norms: One divisor per order and width, order major, for the
            projected potential. Unset divides by one.
        num_recursion_steps: How many refinement steps the density takes. The
            recursion is unrolled and differentiated through every step.
        feature_self_interaction: Whether an atom's projection includes the
            potential of its own density.
        energy_self_interaction: Whether the electrostatic energy includes each
            atom's interaction with its own density.
        add_local_electron_energy: Whether the local energy of the density is
            added to the total. The reported contribution is zero when disabled.
        quadrupole_feature_corrections: Whether an open system's projection
            carries the quadrupole correction as well.
        field_update: The block each step predicts the increment with. One
            exists.
        field_readout: The block the local energy is read with. One exists.
    """

    multipole_max_l: int = Field(default=0, ge=0)
    multipole_width: float = Field(default=1.0, gt=0)
    feature_max_l: int = Field(default=0, ge=0)
    feature_widths: tuple[float, ...] = (1.0,)
    feature_norms: tuple[float, ...] | None = None
    num_recursion_steps: int = Field(default=1, ge=0)
    feature_self_interaction: bool = False
    energy_self_interaction: bool = False
    add_local_electron_energy: bool = False
    quadrupole_feature_corrections: bool = False
    field_update: Literal["embedded_one_body"] = "embedded_one_body"
    field_readout: Literal["one_body_mlp"] = "one_body_mlp"


class MagneticConfig(ConfigSection):
    """The moment-reading architecture, read by the ``magnetic`` model.

    Args:
        saturation: The moment length at which each element saturates, in
            muB. By atomic number, where an element of the table that is not
            listed saturates at one and a listed element outside the table is
            ignored, so one mapping serves every dataset. Or one value per
            element of the table in its order, which is the frozen tree's
            older form, or one value for every element.
        num_basis: How many Chebyshev polynomials the moment length is
            expanded in for the radial networks.
        lmax: The highest degree of the moment harmonics.
        one_body: Whether each atom carries an energy of its moment length
            alone, beside the readouts.
        one_body_basis: How many polynomials that term has, the constant
            included.
        train_one_body: Whether that term's coefficients are trained. Off,
            they keep the values they were built or loaded with.
        time_reversal: Average the energy at opposite moments before taking
            derivatives. ``batched`` evaluates both orientations together;
            ``sequential`` uses less peak memory. ``none`` leaves it unchanged.
    """

    saturation: dict[int, float] | tuple[float, ...] | float = Field(
        default_factory=dict
    )
    num_basis: int = Field(default=8, ge=1)
    lmax: int = Field(default=3, ge=0)
    one_body: bool = False
    one_body_basis: int = Field(default=10, ge=1)
    train_one_body: bool = True
    time_reversal: Literal["none", "batched", "sequential"] = "none"

    @field_validator("saturation")
    @classmethod
    def _positive(
        cls, value: dict[int, float] | tuple[float, ...] | float
    ) -> dict[int, float] | tuple[float, ...] | float:
        if isinstance(value, dict):
            given = list(value.values())
        elif isinstance(value, tuple):
            given = list(value)
        else:
            given = [value]
        bad = [m for m in given if not m > 0]
        if bad:
            raise ValueError(
                f"model.magnetic.saturation has {bad}: a saturation is a moment "
                f"length, and a length at or below zero saturates every moment."
            )
        return value

    def saturation_for(self, atomic_numbers: Sequence[int]) -> list[float]:
        """One saturation per element of ``atomic_numbers``, in its order.

        Raises:
            ValueError: If the saturations are a list of another length than
                the element table.
        """
        numbers = [int(z) for z in atomic_numbers]
        if isinstance(self.saturation, dict):
            return [float(self.saturation.get(z, 1.0)) for z in numbers]
        if isinstance(self.saturation, tuple):
            if len(self.saturation) != len(numbers):
                raise ValueError(
                    f"model.magnetic.saturation has {len(self.saturation)} "
                    f"values and expected {len(numbers)}, one per element of "
                    f"the table {numbers}. Give them by atomic number instead "
                    f"to be independent of the table."
                )
            return [float(value) for value in self.saturation]
        return [float(self.saturation)] * len(numbers)


class LESConfig(ConfigSection):
    """Latent multipole readouts and the pinned LES energy/Born-charge solver."""

    use_atomwise: Literal[False] = False
    compute_bec: bool = False
    bec_output_index: Literal[0, 1, 2] | None = None
    use_dipole: bool = False
    use_quad: bool = False
    use_induced_charge: bool = False
    use_induced_dipole: bool = False
    use_anisotropic_polarizability: bool = False
    alpha_irreps: str = "0e+1o+2e"
    alpha_1o_nonlinear_readout: bool = False
    alpha_1o_linear_w_pos: bool = True
    make_alpha_positive: bool = False
    make_kappa_positive: bool = False
    output_scale: float = Field(default=0.1, allow_inf_nan=False)
    kappa_scale: float = Field(default=0.01, allow_inf_nan=False)
    alpha_scale: float = Field(default=0.01, allow_inf_nan=False)
    # Solver settings are model state too. sigma and dl are lengths in Å.
    sigma: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    dl: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    remove_self_interaction: bool = True
    use_epsilon_r_scaling: bool = False
    remove_mean: bool = True
    epsilon_factor: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    use_atomic_alpha: bool = False
    use_fixed_atomic_charges: bool = False
    fixed_atomic_charges_scaling_factor: float = Field(default=0.5, allow_inf_nan=False)
    n_layers: int = Field(default=3, ge=1)
    n_hidden: int | tuple[int, ...] | None = (32, 16)
    add_linear_nn: bool = True
    output_scaling_factor: float = Field(default=0.1, allow_inf_nan=False)

    @field_validator("use_atomwise", mode="before")
    @classmethod
    def _charge_source(cls, value):
        if value is True or (
            isinstance(value, str) and value.lower() in {"true", "1", "yes"}
        ):
            raise ValueError(
                "The model readouts supply the charges. Set use_atomwise to false."
            )
        return value

    @field_validator("alpha_irreps")
    @classmethod
    def _alpha_channels(cls, value: str) -> str:
        from mace_core.clebsch_gordan.irreps import Irreps

        terms = Irreps.parse(value).terms
        if not terms or any(str(ir) not in {"0e", "1o", "2e"} for _, ir in terms):
            raise ValueError("alpha_irreps must select from 0e, 1o and 2e")
        return value


class ModelConfig(ConfigSection):
    """The architecture of the backbone.

    Defaults are the legacy command-line defaults. The node features of every
    layer but the last carry `num_channels` copies of each irrep up to `max_L`
    with natural parity, `0e+1o+2e+...`; the last carries only the scalars.
    """

    #: The cutoff radius in Angstrom, shared by the radial basis, the envelope
    #: and the neighbour list.
    model: str = "scale_shift"
    observables: tuple[str, ...] = ("energy", "forces")
    hidden_irreps: str | None = None
    pair_repulsion: bool = False
    scaling: ScalingMethod = "rms_forces"
    #: Observable -> dataset head -> recipe. Unspecified energy heads retain
    #: `scaling`; other unspecified heads use the identity transformation.
    normalization: dict[str, dict[str, ScaleShiftSpec]] = Field(default_factory=dict)
    readout: ReadoutConfig = ReadoutConfig()
    graph_features: tuple[GraphFeatureConfig, ...] = ()
    backend: str = "reference"
    les: LESConfig | None = None
    polar: PolarConfig = PolarConfig()
    magnetic: MagneticConfig = MagneticConfig()

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
    interaction_first: InteractionKind = "residual"
    interaction: InteractionKind = "residual"
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

    @model_validator(mode="after")
    def _les_settings_belong_to_les(self) -> ModelConfig:
        if self.les is not None and self.model != "les":
            raise ValueError("model.les requires model.model='les'")
        return self

    @property
    def feature_storages(self) -> dict[str, Storage]:
        """Custom inputs' row ownership for graph building, batching and padding.

        Schema inputs already carry their row declaration in GRAPH_SCHEMA.
        """
        from mace_core.graph import INPUT_KEYS

        return {
            spec.name: spec.per
            for spec in self.graph_features
            if spec.name not in INPUT_KEYS
        }

    @field_validator("graph_features")
    @classmethod
    def _unique_features(
        cls, specs: tuple[GraphFeatureConfig, ...]
    ) -> tuple[GraphFeatureConfig, ...]:
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("model.graph_features repeats an input name")
        return specs

    def normalization_for_heads(
        self, sources: Mapping[str, str]
    ) -> dict[str, dict[str, ScaleShiftSpec]]:
        """Keep recipes aligned when dataset heads are selected or renamed."""
        result = {}
        for name, recipes in self.normalization.items():
            mapped = {
                head: recipes.get(source, NoScaling())
                for head, source in sources.items()
                if name != "energy" or source in recipes
            }
            # Energy always has buffers, even under the legacy aggregate
            # `scaling` default. Other readouts retain their optional operator.
            if mapped:
                result[name] = mapped
        return result

    @property
    def node_irreps(self) -> str:
        """One channel's node irreps, expanding ``max_L`` when unspecified."""
        return self.hidden_irreps or "+".join(
            f"{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(self.max_L + 1)
        )

    @model_validator(mode="after")
    def _edge_irreps_need_a_nonlinear_layer(self) -> ModelConfig:
        """`edge_irreps` and `use_edge_irreps_first` are read by `nonlinear`
        layers only, and a setting nothing reads is refused rather than kept."""
        later_nonlinear = self.num_interactions > 1 and self.interaction == "nonlinear"
        first_nonlinear = self.interaction_first == "nonlinear"
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
        return self


__all__ += [
    "ClebschGordanBasis",
    "GraphFeatureConfig",
    "MagneticConfig",
    "PolarConfig",
    "ReadoutConfig",
    "ScaleShiftSpec",
    "ScalingMethod",
]
