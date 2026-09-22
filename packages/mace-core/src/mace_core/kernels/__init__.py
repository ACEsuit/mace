"""The kernel backend contract: descriptors, capabilities, protocol, registry.

Imports no framework. Everything here is the statement of what an operation is
and what a backend can do, expressed over strings and numbers, so that the same
contract serves the torch and the jax implementations and can be written into a
checkpoint.

Three rules this package exists to enforce, each replacing something the frozen
tree does the other way round:

* **A backend declares, it never chooses.** On the frozen tree a host without
  ``cuequivariance`` silently switches the Clebsch-Gordan basis and trains a
  different network. Here the basis is a recorded field and a backend that
  cannot serve it fails the build loudly.
* **Dispatch resolves once, at build time.** The frozen tree branches inside
  the interaction blocks on whether an attribute is set, and monkeypatches
  ``forward`` in three places. Here a descriptor goes in, an op comes out, and
  the op is frozen into the module tree.
* **One canonical weight layout.** The frozen tree has five conversion command
  line tools because each backend stores weights its own way. The layout
  carries its normalization folded into the weight, so the scales a fresh
  draw uses are part of the format and are stated here too.
"""

from mace_core.kernels.canonical import (
    CANONICAL_LAYOUT,
    KERNEL_SPEC_VERSION,
    canonical_weight_shape,
    fully_connected_tp_weight_scale,
    linear_weight_scale,
    symmetric_contraction_weight_scale,
)
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import PRECISIONS, Precision
from mace_core.kernels.protocol import (
    DISPATCHED_OPS,
    INTERNAL_WEIGHT_OPS,
    REFERENCE_ONLY_OPS,
    InternalWeights,
    KernelBackend,
)
from mace_core.kernels.registry import (
    ENTRY_POINT_GROUPS,
    BackendNotAvailableError,
    DiscoveredBackend,
    available_backends,
    get_backend,
)

__all__ = [
    "CANONICAL_LAYOUT",
    "DISPATCHED_OPS",
    "ENTRY_POINT_GROUPS",
    "INTERNAL_WEIGHT_OPS",
    "KERNEL_SPEC_VERSION",
    "PRECISIONS",
    "REFERENCE_ONLY_OPS",
    "BackendCapabilities",
    "BackendNotAvailableError",
    "ChannelwiseTPConvDescriptor",
    "Descriptor",
    "DiscoveredBackend",
    "FullyConnectedTPDescriptor",
    "InternalWeights",
    "KernelBackend",
    "LinearDescriptor",
    "Precision",
    "RadialBasisDescriptor",
    "SegmentReduceDescriptor",
    "SphericalHarmonicsDescriptor",
    "SymmetricContractionDescriptor",
    "UnsupportedDescriptorError",
    "available_backends",
    "canonical_weight_shape",
    "fully_connected_tp_weight_scale",
    "get_backend",
    "linear_weight_scale",
    "symmetric_contraction_weight_scale",
]
