"""The native JAX backend, discovered through mace.kernel_backends.jax."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.kernels.canonical import KERNEL_SPEC_VERSION
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    FullyConnectedTPDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SegmentReduceDescriptor,
)
from mace_core.kernels.protocol import DISPATCHED_OPS, REFERENCE_ONLY_OPS

from mace_jax.kernels.ops import (
    CanonicalOp,
    ChannelwiseTPConv,
    FullyConnectedTP,
    Linear,
    RadialMLP,
    SymmetricContraction,
)
from mace_jax.nn.radial import bessel, polynomial_envelope


class SphericalHarmonics(nnx.Module):
    """Component-normalized e3nn convention (MIT), with y the polar axis.

    https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_spherical_harmonics.py
    Evaluation couples degree l-1 with a vector using mace_core's real Wigner
    tensors; normalization is fixed by Y_l(y) = sqrt(2l+1).
    """

    def __init__(self, descriptor):
        self.lmax = descriptor.lmax
        self.normalize = descriptor.normalize
        self.couplings = nnx.List()
        for degree in range(2, self.lmax + 1):
            coefficient = wigner_3j_real(degree, degree - 1, 1)
            factor = (
                math.sqrt((2 * degree + 1) / (3 * (2 * degree - 1)))
                / coefficient[degree, degree - 1, 1]
            )
            self.couplings.append(
                nnx.Variable(
                    jnp.asarray(factor * coefficient, dtype=descriptor.precision)
                )
            )

    def __call__(self, vectors):
        unit = (
            vectors
            / jnp.sqrt(jnp.maximum(jnp.sum(vectors**2, axis=-1, keepdims=True), 1e-24))
            if self.normalize
            else vectors
        )
        values = [jnp.ones((*vectors.shape[:-1], 1), dtype=vectors.dtype)]
        if self.lmax:
            values.append(math.sqrt(3) * unit)
        for coefficient in self.couplings:
            values.append(
                jnp.einsum(
                    "...i,...j,oij->...o",
                    values[-1],
                    values[1],
                    coefficient.get_value(),
                )
            )
        return jnp.concatenate(values, axis=-1)


class RadialBasis(CanonicalOp):
    def __init__(self, descriptor):
        self.cutoff, self.order = descriptor.cutoff, descriptor.cutoff_order
        super().__init__(
            {
                "frequencies": jnp.arange(
                    1, descriptor.num_basis + 1, dtype=descriptor.precision
                )
                * (math.pi / descriptor.cutoff),
                "prefactor": jnp.asarray(
                    math.sqrt(2 / descriptor.cutoff), dtype=descriptor.precision
                ),
            }
        )

    def __call__(self, lengths):
        return bessel(
            lengths, self.tensor("frequencies"), self.tensor("prefactor")
        ) * polynomial_envelope(lengths, self.cutoff, self.order)


class SegmentReduce(nnx.Module):
    def __init__(self, descriptor):
        self.reduction = descriptor.reduction

    def __call__(self, values, indices, num_segments):
        return jax.ops.segment_sum(values, indices, num_segments=num_segments)


@dataclass(frozen=True)
class ReferenceCapabilities(BackendCapabilities):
    def unsupported_reason(self, descriptor):
        reason = super().unsupported_reason(descriptor)
        if reason:
            return reason
        if descriptor.accumulate_floor != descriptor.precision:
            return (
                "JAX reference requires a uniform op dtype; request the wider precision"
            )
        if (
            isinstance(descriptor, SegmentReduceDescriptor)
            and descriptor.reduction != "sum"
        ):
            return "JAX reference segment_reduce supports sum only"
        if (
            isinstance(descriptor, RadialBasisDescriptor)
            and descriptor.kind != "bessel"
        ):
            return f"radial basis {descriptor.kind!r} is not supported; use bessel"
        if (
            isinstance(descriptor, RadialMLPDescriptor)
            and descriptor.kind != "normalized_silu"
        ):
            return (
                f"radial MLP {descriptor.kind!r} is not supported; use normalized_silu"
            )
        if isinstance(descriptor, FullyConnectedTPDescriptor) and any(
            ir.degree != 0 or ir.parity != 1
            for _, ir in Irreps.parse(descriptor.irreps_in2)
        ):
            return "fully connected TP requires even scalar second inputs"
        return None


class ReferenceBackend:
    name = "reference"

    def capabilities(self):
        return ReferenceCapabilities(
            ops=DISPATCHED_OPS | REFERENCE_ONLY_OPS,
            devices=frozenset({"cpu", "cuda"}),
            dtypes=frozenset({"float32", "float64"}),
            bases=frozenset({"reduced"}),
            supports_double_backward=False,
            spec_version=KERNEL_SPEC_VERSION,
        )

    def _build(self, constructor, descriptor):
        self.capabilities().require(descriptor, self.name)
        return constructor(descriptor)

    def make_linear(self, descriptor):
        return self._build(Linear, descriptor)

    def make_channelwise_tp_conv(self, descriptor):
        return self._build(ChannelwiseTPConv, descriptor)

    def make_fully_connected_tp(self, descriptor):
        return self._build(FullyConnectedTP, descriptor)

    def make_symmetric_contraction(self, descriptor):
        return self._build(SymmetricContraction, descriptor)

    def make_segment_reduce(self, descriptor):
        return self._build(SegmentReduce, descriptor)

    def make_spherical_harmonics(self, descriptor):
        return self._build(SphericalHarmonics, descriptor)

    def make_radial_basis(self, descriptor):
        return self._build(RadialBasis, descriptor)

    def make_radial_mlp(self, descriptor):
        return self._build(RadialMLP, descriptor)

    def make_interaction_layer(self, descriptors):
        return None
