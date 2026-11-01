"""Framework-independent kernel descriptors, capabilities, protocol and registry.

Backends must support the recorded basis or reject the build. Dispatch resolves
once at construction. Canonical weights are raw; ``layout`` specifies the
per-path normalization applied in forward and the scale used for fresh draws.
"""

from mace_core.kernels.canonical import (
    CANONICAL_LAYOUT,
    KERNEL_SPEC_VERSION,
    LAYER_NORM_EPSILON,
    SILU_SECOND_MOMENT_SCALE,
    canonical_weight_shape,
    contraction_path_labels,
    contraction_path_order,
    exact_silu_second_moment_scale,
    fully_connected_tp_path_normalization,
    linear_path_normalization,
    radial_mlp_initial_bound,
    radial_mlp_layer_normalization,
    symmetric_contraction_initial_scale,
)
from mace_core.kernels.capabilities import (
    BackendCapabilities,
    UnsupportedDescriptorError,
)
from mace_core.kernels.descriptors import (
    CLEBSCH_GORDAN_BASES,
    ChannelwiseTPConvDescriptor,
    Descriptor,
    DistanceTransformDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import ConvolutionPath, channelwise_paths, path_irreps
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
    DuplicateBackendError,
    available_backends,
    get_backend,
)
from mace_core.kernels.reorder import (
    ReorderBlock,
    ReorderError,
    WeightReorder,
    derive_reorder,
)

__all__ = [
    "CANONICAL_LAYOUT",
    "CLEBSCH_GORDAN_BASES",
    "DISPATCHED_OPS",
    "ENTRY_POINT_GROUPS",
    "INTERNAL_WEIGHT_OPS",
    "KERNEL_SPEC_VERSION",
    "LAYER_NORM_EPSILON",
    "PRECISIONS",
    "REFERENCE_ONLY_OPS",
    "SILU_SECOND_MOMENT_SCALE",
    "BackendCapabilities",
    "BackendNotAvailableError",
    "ChannelwiseTPConvDescriptor",
    "ConvolutionPath",
    "Descriptor",
    "DiscoveredBackend",
    "DistanceTransformDescriptor",
    "DuplicateBackendError",
    "FullyConnectedTPDescriptor",
    "InternalWeights",
    "KernelBackend",
    "LinearDescriptor",
    "Precision",
    "RadialBasisDescriptor",
    "RadialMLPDescriptor",
    "ReorderBlock",
    "ReorderError",
    "SegmentReduceDescriptor",
    "SphericalHarmonicsDescriptor",
    "SymmetricContractionDescriptor",
    "UnsupportedDescriptorError",
    "WeightReorder",
    "available_backends",
    "canonical_weight_shape",
    "channelwise_paths",
    "contraction_path_labels",
    "contraction_path_order",
    "derive_reorder",
    "exact_silu_second_moment_scale",
    "fully_connected_tp_path_normalization",
    "get_backend",
    "linear_path_normalization",
    "path_irreps",
    "radial_mlp_initial_bound",
    "radial_mlp_layer_normalization",
    "symmetric_contraction_initial_scale",
]
