"""Optional inference regions, separate from the live-weight kernel contract.

These shapes describe existing algebra. They neither select an implementation
nor change the canonical checkpoint format. Member precision and normalization
remain those of the unfused operations, including raw canonical weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.precision import PRECISIONS, Precision
from mace_core.kernels.spans import EdgeConvolutionSpan
from mace_core.kernels.specs import (
    FullyConnectedTPSpec,
    KernelSpec,
    LinearSpec,
    SymmetricContractionSpec,
)

__all__ = ["InferenceSpec", "InteractionFusionSpec", "ProductReadoutSpec"]


def _expanded(irreps: str, channels: int) -> str:
    return "+".join(f"{channels * mul}x{ir}" for mul, ir in Irreps.parse(irreps))


def _same_irreps(left: str, right: str, boundary: str) -> None:
    if Irreps.parse(left).terms != Irreps.parse(right).terms:
        raise ValueError(f"{boundary}: incompatible irreps {left!r} and {right!r}")


def _one_layout(members: tuple[KernelSpec, ...]) -> None:
    layouts = {member.activation_layout or "mul_ir" for member in members}
    if len(layouts) != 1:
        raise ValueError("an inference region requires one activation layout")


@dataclass(frozen=True)
class ProductReadoutSpec:
    """Contraction, channel mixing, optional residual, optional linear readout.

    ``features = linear(contraction(message, element)) + residual``; the add
    exists only when ``residual`` is True. A readout consumes these final
    features and returns its full declared components, before head selection,
    scaling, isolated-atom energies or graph reduction. Return features whenever
    a following layer or another observable needs them. A terminal readout-only
    request may omit them to avoid materializing an unused intermediate.

    Runtime inputs are ``(message, element[, residual])``. Message and residual
    are [N, declared width]; element is int64 [N], indexing the contraction's
    element table. Outputs follow output_names, normally ``(features[, readout])``.
    Nonlinear readouts remain ordinary operations outside this shape.
    ``interface_dtype`` is the floating input/output dtype, independently of
    each member's compute and accumulation policy. Preserve the unfused
    boundary casts when these differ.
    """

    contraction: SymmetricContractionSpec
    linear: LinearSpec
    residual: bool = False
    readout: LinearSpec | None = None
    interface_dtype: Precision = field(default="float64", kw_only=True)
    return_features: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.contraction, SymmetricContractionSpec) or not isinstance(
            self.linear, LinearSpec
        ):
            raise TypeError("product members must be contraction and linear specs")
        if type(self.residual) is not bool:
            raise TypeError("residual must be a bool")
        if type(self.return_features) is not bool:
            raise TypeError("return_features must be a bool")
        if not self.return_features and self.readout is None:
            raise ValueError("return_features=False requires a readout")
        if self.interface_dtype not in PRECISIONS:
            raise ValueError(f"unknown interface_dtype {self.interface_dtype!r}")
        if self.readout is not None and not isinstance(self.readout, LinearSpec):
            raise TypeError("readout must be a LinearSpec or None")
        if self.contraction.basis != "reduced":
            raise ValueError("prepare inference from canonical reduced weights")
        output = _expanded(self.contraction.irreps_out, self.contraction.num_features)
        _same_irreps(output, self.linear.irreps_in, "contraction to linear")
        if self.readout is not None:
            _same_irreps(self.linear.irreps_out, self.readout.irreps_in, "readout")
        _one_layout(self.members)

    @property
    def members(self) -> tuple[KernelSpec, ...]:
        return (self.contraction, self.linear) + (
            () if self.readout is None else (self.readout,)
        )

    @property
    def weight_roles(self) -> tuple[str, ...]:
        return ("contraction", "linear") + (
            () if self.readout is None else ("readout",)
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("message", "element") + (() if not self.residual else ("residual",))

    @property
    def output_names(self) -> tuple[str, ...]:
        return (("features",) if self.return_features else ()) + (
            () if self.readout is None else ("readout",)
        )


@dataclass(frozen=True)
class InteractionFusionSpec:
    """One standard interaction and product, optionally including a readout.

    Map node features with ``linear_up``, run ``edge``, then ``linear_down``.
    Divide by the canonical ``normalization.neighbours`` scalar, or by
    ``density + 1`` when the edge span declares density. If product.residual
    is False, ``skip(message, attributes)`` replaces the normalized message.
    Otherwise ``skip(input_features, attributes)`` is added after the product
    linear. The product then follows :class:`ProductReadoutSpec` exactly.

    Inputs: node_features [N,C], vectors [E,3] in Angstrom (receiver minus
    sender including periodic shift), attributes [N,Z], atomic_numbers [N],
    element [N], sender [E], receiver [E]. Indices are int64. Element indexes
    the product table; atomic_numbers are physical species, not table rows.
    All N nodes produce features; ownership masking and MPI exchange stay
    outside the region. Graph topology and vectors are live on every call.

    Edge cutoff placement is ``edge.radial_basis.apply_cutoff``: when False,
    multiply radial-network and density outputs by the envelope instead.
    This shape describes standard plain/residual and density interactions.
    Element-conditioned nonlinear interactions require a separate shape.
    """

    edge: EdgeConvolutionSpan
    linear_up: LinearSpec
    linear_down: LinearSpec
    skip: FullyConnectedTPSpec
    product: ProductReadoutSpec

    def __post_init__(self) -> None:
        if not isinstance(self.edge, EdgeConvolutionSpan) or not isinstance(
            self.product, ProductReadoutSpec
        ):
            raise TypeError("interaction requires edge and product specifications")
        if not all(
            isinstance(op, LinearSpec) for op in (self.linear_up, self.linear_down)
        ):
            raise TypeError("interaction maps must be LinearSpec")
        if not isinstance(self.skip, FullyConnectedTPSpec):
            raise TypeError("skip must be FullyConnectedTPSpec")
        if self.edge.radial_network.num_elements:
            raise ValueError("conditioned interactions need their own inference shape")
        convolution = self.edge.convolution
        channels = convolution.num_features
        _same_irreps(
            self.linear_up.irreps_out,
            _expanded(convolution.irreps_node, channels),
            "linear_up to convolution",
        )
        _same_irreps(
            _expanded(convolution.irreps_paths, channels),
            self.linear_down.irreps_in,
            "convolution to linear_down",
        )
        _same_irreps(
            self.linear_down.irreps_out,
            _expanded(
                self.product.contraction.irreps_in,
                self.product.contraction.num_features,
            ),
            "interaction to product",
        )
        _same_irreps(
            self.linear_up.irreps_in
            if self.product.residual
            else self.linear_down.irreps_out,
            self.skip.irreps_in1,
            "skip input",
        )
        _same_irreps(
            self.product.linear.irreps_out
            if self.product.residual
            else self.linear_down.irreps_out,
            self.skip.irreps_out,
            "skip output",
        )
        attributes = Irreps.parse(self.skip.irreps_in2)
        if any(ir.degree != 0 or ir.parity != 1 for _, ir in attributes):
            raise ValueError("skip attributes must be scalar species indicators")
        if self.product.contraction.num_elements not in (1, attributes.dimension):
            raise ValueError("product element table must be shared or match species")
        if self.edge.radial_network.channels[0] != self.edge.radial_basis.num_basis:
            raise ValueError("radial network input width must match the radial basis")
        if self.edge.radial_network.channels[-1] != len(convolution.paths) * channels:
            raise ValueError(
                "radial network output width must match convolution weights"
            )
        _one_layout(self.members)

    @property
    def members(self) -> tuple[KernelSpec, ...]:
        return (
            self.linear_up,
            *self.edge.kernel_specs,
            self.linear_down,
            self.skip,
            *self.product.members,
        )

    @property
    def weight_roles(self) -> tuple[str, ...]:
        return (
            ("linear_up", "radial_basis", "radial_network", "linear_down", "skip")
            + (() if self.edge.distance_transform is None else ("distance_transform",))
            + (("normalization",) if self.edge.density is None else ("density",))
            + tuple(f"product.{role}" for role in self.product.weight_roles)
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (
            "node_features",
            "vectors",
            "attributes",
            "atomic_numbers",
            "element",
            "sender",
            "receiver",
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.product.output_names

    @property
    def interface_dtype(self) -> Precision:
        return self.product.interface_dtype


InferenceSpec: TypeAlias = ProductReadoutSpec | InteractionFusionSpec
