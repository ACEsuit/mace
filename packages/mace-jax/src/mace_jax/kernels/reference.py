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
from mace_core.kernels.protocol import DISPATCHED_OPS, REFERENCE_ONLY_OPS
from mace_core.kernels.specs import (
    FullyConnectedTPSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SegmentReduceSpec,
)

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

    def __init__(self, kernel_spec):
        self.lmax = kernel_spec.lmax
        self.normalize = kernel_spec.normalize
        self.couplings = nnx.List()
        for degree in range(2, self.lmax + 1):
            coefficient = wigner_3j_real(degree, degree - 1, 1)
            factor = (
                math.sqrt((2 * degree + 1) / (3 * (2 * degree - 1)))
                / coefficient[degree, degree - 1, 1]
            )
            self.couplings.append(
                nnx.Variable(
                    jnp.asarray(factor * coefficient, dtype=kernel_spec.precision)
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
    def __init__(self, kernel_spec):
        self.cutoff, self.order = kernel_spec.cutoff, kernel_spec.cutoff_order
        self.apply_cutoff = kernel_spec.apply_cutoff
        super().__init__(
            {
                "frequencies": jnp.arange(
                    1, kernel_spec.num_basis + 1, dtype=kernel_spec.precision
                )
                * (math.pi / kernel_spec.cutoff),
                "prefactor": jnp.asarray(
                    math.sqrt(2 / kernel_spec.cutoff), dtype=kernel_spec.precision
                ),
            }
        )

    def __call__(self, lengths, envelope=None):
        features = bessel(lengths, self.tensor("frequencies"), self.tensor("prefactor"))
        if self.apply_cutoff:
            features = features * (
                polynomial_envelope(lengths, self.cutoff, self.order)
                if envelope is None
                else envelope
            )
        return features


class SegmentReduce(nnx.Module):
    def __init__(self, kernel_spec):
        self.reduction = kernel_spec.reduction

    def __call__(self, values, indices, num_segments):
        return jax.ops.segment_sum(values, indices, num_segments=num_segments)


@dataclass(frozen=True)
class ReferenceCapabilities(BackendCapabilities):
    def unsupported_reason(self, kernel_spec):
        reason = super().unsupported_reason(kernel_spec)
        if reason:
            return reason
        if kernel_spec.accumulate_floor != kernel_spec.precision:
            return (
                "JAX reference requires a uniform op dtype; request the wider precision"
            )
        if (
            isinstance(kernel_spec, SegmentReduceSpec)
            and kernel_spec.reduction != "sum"
        ):
            return "JAX reference segment_reduce supports sum only"
        if isinstance(kernel_spec, RadialBasisSpec) and kernel_spec.kind != "bessel":
            return f"radial basis {kernel_spec.kind!r} is not supported; use bessel"
        if (
            isinstance(kernel_spec, RadialMLPSpec)
            and kernel_spec.kind != "normalized_silu"
        ):
            return (
                f"radial MLP {kernel_spec.kind!r} is not supported; use normalized_silu"
            )
        if isinstance(kernel_spec, FullyConnectedTPSpec) and any(
            ir.degree != 0 or ir.parity != 1
            for _, ir in Irreps.parse(kernel_spec.irreps_in2)
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

    def _build(self, constructor, kernel_spec):
        self.capabilities().require(kernel_spec, self.name)
        return constructor(kernel_spec)

    def make_linear(self, kernel_spec):
        return self._build(Linear, kernel_spec)

    def make_channelwise_tp_conv(self, kernel_spec):
        return self._build(ChannelwiseTPConv, kernel_spec)

    def make_fully_connected_tp(self, kernel_spec):
        return self._build(FullyConnectedTP, kernel_spec)

    def make_symmetric_contraction(self, kernel_spec):
        return self._build(SymmetricContraction, kernel_spec)

    def make_segment_reduce(self, kernel_spec):
        return self._build(SegmentReduce, kernel_spec)

    def make_spherical_harmonics(self, kernel_spec):
        return self._build(SphericalHarmonics, kernel_spec)

    def make_radial_basis(self, kernel_spec):
        return self._build(RadialBasis, kernel_spec)

    def make_radial_mlp(self, kernel_spec):
        return self._build(RadialMLP, kernel_spec)

    def make_interaction_layer(self, kernel_specs):
        return None
