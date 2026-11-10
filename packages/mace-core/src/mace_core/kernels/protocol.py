"""Framework-independent kernel factories, resolved once at model build time.

Each factory takes a kernel specification and returns an op bound into the model tree;
forward calls perform no backend lookup. The op set is closed. Spherical
harmonics, radial bases and radial MLPs may use reference implementations;
fused convolution is negotiated through the span factory.

Weight-owning ops implement ``InternalWeights``: canonical checkpoint I/O and
a fresh draw for training from scratch. Zero-only initialization can prevent
learning and does not satisfy that contract.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SegmentReduceSpec,
    SphericalHarmonicsSpec,
    SymmetricContractionSpec,
)

__all__ = [
    "DISPATCHED_OPS",
    "INTERNAL_WEIGHT_OPS",
    "REFERENCE_ONLY_OPS",
    "InternalWeights",
    "KernelBackend",
    "TensorT",
]

#: The array type a framework binds. Unbound for the same reason `MACEOutput`
#: leaves it unbound: a structural bound would be a claim about torch and jax
#: that this package cannot check.
TensorT = TypeVar("TensorT")

#: The primary dispatched op classes. A backend declares the subset it builds;
#: the framework's composite supplies reference fallback for the rest.
DISPATCHED_OPS: frozenset[str] = frozenset(
    {
        "linear",
        "channelwise_tp_conv",
        "symmetric_contraction",
        "fully_connected_tp",
        "segment_reduce",
    }
)

#: The ops a backend may decline, leaving them to the reference.
REFERENCE_ONLY_OPS: frozenset[str] = frozenset(
    {"spherical_harmonics", "radial_basis", "radial_mlp"}
)

#: The ops that hold weights, and therefore implement :class:`InternalWeights`.
#: The convolution takes its weights from the radial MLP, and the reduction
#: has none.
INTERNAL_WEIGHT_OPS: frozenset[str] = frozenset(
    {"linear", "symmetric_contraction", "fully_connected_tp", "radial_mlp"}
)


@runtime_checkable
class InternalWeights(Protocol):
    """What an op that owns its weights provides beyond computing.

    The canonical pair is called only at the checkpoint boundary, never in
    ``forward``. The draw is called once, when a model is built to be trained
    rather than loaded.
    """

    def to_canonical(self) -> Any:
        """This op's weights in the canonical layout, as a mapping of names."""
        ...

    def load_canonical(self, state: Any) -> None:
        """Put canonical weights back. The inverse of :meth:`to_canonical`."""
        ...

    def initialize_weights(self, seed: int) -> None:
        """Draw a fresh set, in the canonical layout and at the canonical scale.

        The scales are :mod:`mace_core.kernels.canonical`'s: a standard normal
        for the linear and skip-connection weights, whose ``1 / sqrt(fan_in)``
        is applied in the forward and never folded into the draw, a zero bias,
        a standard normal over the block's path count for the symmetric
        contraction, and for the radial MLP whichever draw its kind states.
        Two backends initialised from the same seed then differ in
        their draw and not in the size of what they drew, and a weight trains
        at the effective learning rate the frozen tree's does.

        Args:
            seed: Chosen by the caller per op, so that two ops of the same shape
                in one model do not receive the same weights.
        """
        ...


@runtime_checkable
class KernelBackend(Protocol):
    """What every kernel backend provides.

    Attributes:
        name: The registry name, which is also what a resolved config records
            so a run can be reproduced on the backend it actually used.
    """

    name: str

    def capabilities(self) -> BackendCapabilities:
        """What this backend declares it can do. Called before anything is built."""
        ...

    def make_linear(self, kernel_spec: LinearSpec) -> Any:
        """An equivariant linear map, with the specified bias."""
        ...

    def make_channelwise_tp_conv(self, kernel_spec: ChannelwiseTPConvSpec) -> Any:
        """The message-passing tensor product. Returns node-level values.

        Always ``[n_nodes, ...]``. Whether the reduction over edges is fused
        into the kernel is this backend's business; the model never learns of
        it and never branches on it.
        """
        ...

    def make_symmetric_contraction(self, kernel_spec: SymmetricContractionSpec) -> Any:
        """The many-body contraction over the specified basis."""
        ...

    def make_fully_connected_tp(self, kernel_spec: FullyConnectedTPSpec) -> Any:
        """The skip connection's tensor product."""
        ...

    def make_segment_reduce(self, kernel_spec: SegmentReduceSpec) -> Any:
        """A reduction into segments, with no irreps semantics."""
        ...

    def make_spherical_harmonics(
        self, kernel_spec: SphericalHarmonicsSpec
    ) -> Any | None:
        """Real spherical harmonics, or ``None`` to leave them to the reference.

        A backend that supplies its own must produce the convention
        :class:`~mace_core.kernels.specs.SphericalHarmonicsSpec`
        states, which is e3nn's with ``normalization="component"``, the one
        legacy models were trained in. Producing a different one is not an
        optimization, it is a different model.
        """
        ...

    def make_radial_basis(self, kernel_spec: RadialBasisSpec) -> Any | None:
        """The radial embedding, or ``None`` to leave it to the reference.

        The op evaluates lengths with an optional precomputed ``envelope``;
        this keeps a distance transform from changing the cutoff's argument.
        Torch ops expose ``kernel_spec``, ``cutoff``, ``canonical_components``
        and ``span_weights`` for composition and canonical checkpoint roles.
        """
        ...

    def make_radial_mlp(self, kernel_spec: RadialMLPSpec) -> Any | None:
        """The radial MLP, or ``None`` to leave it to the reference.

        Its weights are canonical whoever builds it, so declining changes where
        it runs and not what a checkpoint holds.
        """
        ...

    def make_interaction_layer(self, kernel_specs: tuple[Any, ...]) -> Any | None:
        """A stateless fused span for the supplied specifications, or ``None``.

        ``EdgeConvolutionSpan`` describes the standard radial/harmonic/convolution
        tuple. Unsupported tuples, including the empty tuple, return ``None``
        so the caller retains the member operations. The framework's span
        contract defines the call signature and live member weights.
        """
        ...
