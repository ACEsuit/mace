"""The contract a kernel backend implements.

Generic over the tensor type, so the same contract serves torch and jax and
this module imports neither. The pattern is the one
:class:`~mace_core.outputs.MACEOutput` uses, and it has to work with no
framework installed at all.

**Factories, not calls.** Every method takes a descriptor and returns an op.
The resolution happens once, at model build time, and the op is frozen into the
module tree. Nothing in this file is reachable from ``forward``: no dtype
check, no device check, no ``isinstance``, no lookup. That is the difference
between this and the frozen tree's dispatch, which branches inside the
interaction blocks on whether an attribute happens to be set.

**The op set is closed.** It is exactly the operations that differ between
backends, and it does not grow to accommodate one. Two of them are
reference-only: spherical harmonics and the radial basis are cheap closed forms
that a backend *may* override and is not required to, and declining is a normal
answer rather than a failure.

**Three of the ops hold their own weights**, and those three have a contract of
their own: they read and write the canonical layout, and they can draw a fresh
set. The last of the three is not an optimization. An op that allocates zeros
and waits to be loaded is correct for every use that loads a checkpoint, and it
builds a model that cannot be trained from scratch: the product of zero-valued
tensors is zero, so is its gradient, and the run finishes and reports a model
that never moved.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
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

#: The ops a backend must implement to be usable at all.
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
REFERENCE_ONLY_OPS: frozenset[str] = frozenset({"spherical_harmonics", "radial_basis"})

#: The ops that hold weights, and therefore implement :class:`InternalWeights`.
#: The other two dispatched ops take theirs from outside, or have none.
INTERNAL_WEIGHT_OPS: frozenset[str] = frozenset(
    {"linear", "symmetric_contraction", "fully_connected_tp"}
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

        The scales are :mod:`mace_core.kernels.canonical`'s, so two backends
        initialised from the same seed differ in their draw and not in the size
        of what they drew. A backend that used its own scale would train at a
        different effective learning rate than the checkpoint format implies.

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

    def make_linear(self, descriptor: LinearDescriptor) -> Any:
        """An equivariant linear map, bias included when the descriptor says so."""
        ...

    def make_channelwise_tp_conv(self, descriptor: ChannelwiseTPConvDescriptor) -> Any:
        """The message-passing tensor product. Returns node-level values.

        Always ``[n_nodes, ...]``. Whether the reduction over edges is fused
        into the kernel is this backend's business; the model never learns of
        it and never branches on it.
        """
        ...

    def make_symmetric_contraction(
        self, descriptor: SymmetricContractionDescriptor
    ) -> Any:
        """The many-body contraction, over the basis the descriptor records."""
        ...

    def make_fully_connected_tp(self, descriptor: FullyConnectedTPDescriptor) -> Any:
        """The skip connection's tensor product."""
        ...

    def make_segment_reduce(self, descriptor: SegmentReduceDescriptor) -> Any:
        """A reduction into segments, with no irreps semantics."""
        ...

    def make_spherical_harmonics(
        self, descriptor: SphericalHarmonicsDescriptor
    ) -> Any | None:
        """Real spherical harmonics, or ``None`` to leave them to the reference.

        A backend that supplies its own must produce the convention
        :mod:`mace_core.clebsch_gordan.real_basis` states. Producing a
        different one is not an optimization, it is a different model.
        """
        ...

    def make_radial_basis(self, descriptor: RadialBasisDescriptor) -> Any | None:
        """The radial embedding, or ``None`` to leave it to the reference."""
        ...

    def make_interaction_layer(self, descriptors: tuple[Any, ...]) -> Any | None:
        """A whole interaction layer in one kernel, or ``None``.

        The optional span factory. A backend that can fuse further than one op
        at a time says so here; ``None`` means the layer is built op by op, and
        that is the normal answer.
        """
        ...
