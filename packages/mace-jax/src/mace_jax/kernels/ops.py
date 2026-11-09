"""Native JAX operators in the shared canonical layout.

Normalization conventions follow e3nn (MIT), through mace_core's canonical
contract: https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/o3/_linear.py.
The reduced coupling tensors and path names come exclusively from mace_core.
No custom derivatives: JAX differentiates the ordinary array operations.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.canonical import (
    SILU_SECOND_MOMENT_SCALE,
    contraction_path_labels,
    contraction_path_order,
    fully_connected_tp_path_normalization,
    linear_path_normalization,
    linear_weight_table,
)
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    RadialMLPSpec,
    SymmetricContractionSpec,
)


def expanded(irreps: str, channels: int) -> str:
    return "+".join(f"{mul * channels}x{irrep}" for mul, irrep in Irreps.parse(irreps))


def per_channel(features: jax.Array, irreps: str, channels: int) -> jax.Array:
    parts, start = [], 0
    for multiplicity, irrep in Irreps.parse(irreps):
        width = multiplicity * irrep.dimension
        parts.append(
            features[:, start : start + channels * width].reshape(
                features.shape[0], channels, width
            )
        )
        start += channels * width
    return jnp.concatenate(parts, axis=-1)


def grouped(features: jax.Array, irreps: str) -> jax.Array:
    parts, start = [], 0
    for multiplicity, irrep in Irreps.parse(irreps):
        width = multiplicity * irrep.dimension
        parts.append(
            features[:, :, start : start + width].reshape(
                features.shape[0], features.shape[1] * width
            )
        )
        start += width
    return jnp.concatenate(parts, axis=-1)


class CanonicalOp(nnx.Module):
    """Weights in the on-disk layout; loading validates before changing any value."""

    def __init__(self, arrays: Mapping[str, jax.Array]) -> None:
        self.state = nnx.Dict(
            {
                name: nnx.Param(value)
                if jnp.issubdtype(value.dtype, jnp.floating)
                else nnx.Variable(value)
                for name, value in arrays.items()
            }
        )

    def tensor(self, name: str) -> jax.Array:
        return self.state[name].get_value()

    def to_canonical(self) -> dict[str, jax.Array]:
        return {name: value.get_value() for name, value in self.state.items()}

    def load_canonical(self, state: Mapping[str, Any]) -> None:
        expected = self.to_canonical()
        if state.keys() != expected.keys():
            raise ValueError(
                f"canonical keys {sorted(state)}; expected {sorted(expected)}"
            )
        for name, value in state.items():
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"{name}: shape {value.shape}; expected {expected[name].shape}"
                )
        for name, value in state.items():
            self.state[name].set_value(jnp.asarray(value, dtype=expected[name].dtype))

    def initialize_weights(self, seed: int) -> None:
        for index, (name, variable) in enumerate(self.state.items()):
            value = variable.get_value()
            if jnp.issubdtype(value.dtype, jnp.floating):
                value = (
                    jnp.zeros_like(value)
                    if name == "bias"
                    else jax.random.normal(
                        jax.random.key(seed + index), value.shape, value.dtype
                    )
                )
                variable.set_value(value)


class Linear(CanonicalOp):
    """Equivariant linear: [nodes, dim_in] -> [nodes, dim_out]."""

    def __init__(self, kernel_spec: LinearSpec, *, scalars: str | None = None):
        source = Irreps.parse(kernel_spec.irreps_in)
        target = Irreps.parse(kernel_spec.irreps_out)
        table = linear_weight_table(kernel_spec.irreps_in, kernel_spec.irreps_out)
        count = len(table)
        self.dim_out = target.dimension
        self.scalars = scalars
        self.plans = []
        source_copies = list(source.slices())
        out_copy = 0
        bias_offset = 0
        for multiplicity, irrep in target:
            matching = [
                (index, span)
                for index, (span, other) in enumerate(source_copies)
                if other == irrep
            ]
            components = tuple(
                tuple(range(span.start, span.stop)) for _, span in matching
            )
            indices = tuple(
                tuple(table[(out_copy + row, index)] for index, _ in matching)
                for row in range(multiplicity)
            )
            factor = (
                linear_path_normalization(kernel_spec.irreps_in, irrep)
                if scalars is None
                else fully_connected_tp_path_normalization(
                    kernel_spec.irreps_in, scalars, irrep
                )
            )
            bias = (
                bias_offset
                if kernel_spec.has_bias and irrep.degree == 0 and irrep.parity == 1
                else None
            )
            if bias is not None:
                bias_offset += multiplicity
            self.plans.append(
                (multiplicity, irrep.dimension, components, indices, factor, bias)
            )
            out_copy += multiplicity
        shape = (
            (count,) if scalars is None else (Irreps.parse(scalars).dimension, count)
        )
        arrays = {"weight": jnp.zeros(shape, dtype=kernel_spec.precision)}
        if scalars is None:
            arrays["bias"] = jnp.zeros((bias_offset,), dtype=kernel_spec.precision)
        super().__init__(arrays)

    def __call__(
        self, features: jax.Array, attributes: jax.Array | None = None
    ) -> jax.Array:
        weights = self.tensor("weight")
        if attributes is not None:
            weights = attributes @ weights
        pieces = []
        for multiplicity, dimension, components, indices, factor, bias in self.plans:
            result = jnp.zeros(
                (features.shape[0], multiplicity, dimension), dtype=features.dtype
            )
            if components:
                values = features[:, jnp.asarray(components)]
                selected = weights[..., jnp.asarray(indices)] * factor
                result = jnp.einsum(
                    "ncd,oc->nod" if attributes is None else "ncd,noc->nod",
                    values,
                    selected,
                )
            if bias is not None:
                result = (
                    result
                    + self.tensor("bias")[bias : bias + multiplicity][None, :, None]
                )
            pieces.append(result.reshape(features.shape[0], multiplicity * dimension))
        return jnp.concatenate(pieces, axis=-1)


class FullyConnectedTP(Linear):
    def __init__(self, kernel_spec: FullyConnectedTPSpec):
        if any(
            ir.degree != 0 or ir.parity != 1
            for _, ir in Irreps.parse(kernel_spec.irreps_in2)
        ):
            raise ValueError("fully_connected_tp requires even scalar second inputs")
        super().__init__(
            LinearSpec(
                irreps_in=kernel_spec.irreps_in1,
                irreps_out=kernel_spec.irreps_out,
                precision=kernel_spec.precision,
            ),
            scalars=kernel_spec.irreps_in2,
        )


class ChannelwiseTPConv(nnx.Module):
    """Weighted edge coupling followed by a sum into exactly num_nodes slots."""

    def __init__(self, kernel_spec: ChannelwiseTPConvSpec):
        self.kernel_spec = kernel_spec
        self.coefficients = nnx.List(
            [
                nnx.Variable(
                    jnp.asarray(
                        math.sqrt(path.irrep.dimension)
                        * wigner_3j_real(
                            path.irrep.degree,
                            Irreps.parse(kernel_spec.irreps_node)
                            .terms[path.node_term][1]
                            .degree,
                            Irreps.parse(kernel_spec.irreps_edge)
                            .terms[path.edge_term][1]
                            .degree,
                        ),
                        dtype=kernel_spec.precision,
                    )
                )
                for path in kernel_spec.paths
            ]
        )

    def __call__(self, features, attributes, weights, sender, receiver, num_nodes):
        kernel_spec = self.kernel_spec
        channels = per_channel(
            features, kernel_spec.irreps_node, kernel_spec.num_features
        )
        node_slices = tuple(
            span for span, _ in Irreps.parse(kernel_spec.irreps_node).slices()
        )
        edge_slices = tuple(
            span for span, _ in Irreps.parse(kernel_spec.irreps_edge).slices()
        )
        paths = []
        for index, path in enumerate(kernel_spec.paths):
            coupled = jnp.einsum(
                "eci,ej,oij->eco",
                channels[sender, :, node_slices[path.node_term]],
                attributes[:, edge_slices[path.edge_term]],
                self.coefficients[index].get_value(),
            )
            paths.append(coupled * weights[:, index, :, None])
        messages = jnp.concatenate(paths, axis=-1)
        return grouped(
            jax.ops.segment_sum(messages, receiver, num_segments=num_nodes),
            kernel_spec.irreps_paths,
        )


class SymmetricContraction(CanonicalOp):
    """Reduced, named coupling paths evaluated as shared symmetric monomials."""

    def __init__(self, kernel_spec: SymmetricContractionSpec):
        if kernel_spec.basis != "reduced":
            raise ValueError("JAX symmetric_contraction requires basis='reduced'")
        self.kernel_spec = kernel_spec
        self.path_labels = contraction_path_labels(
            kernel_spec.irreps_in, kernel_spec.irreps_out, kernel_spec.correlation
        )
        self.groups = []
        self.coefficients = nnx.List()
        counts = []
        width = Irreps.parse(kernel_spec.irreps_in).dimension
        offset = 0
        targets = tuple(str(ir) for _, ir in Irreps.parse(kernel_spec.irreps_out))
        for target, order in contraction_path_order(
            kernel_spec.irreps_out, kernel_spec.correlation
        ):
            basis = reduced_symmetric_tensor_product_basis(
                kernel_spec.irreps_in, order, target
            )[target]
            monomials = tuple(
                itertools.combinations_with_replacement(range(width), order)
            )
            lookup = {key: index for index, key in enumerate(monomials)}
            destinations = [
                lookup[tuple(sorted(key))]
                for key in itertools.product(range(width), repeat=order)
            ]
            compressed = np.zeros((basis.shape[0], basis.shape[1], len(monomials)))
            for index, destination in enumerate(destinations):
                compressed[:, :, destination] += basis.reshape(
                    basis.shape[0], basis.shape[1], width**order
                )[:, :, index]
            self.groups.append(
                (targets.index(target), offset, basis.shape[0], monomials)
            )
            self.coefficients.append(
                nnx.Variable(jnp.asarray(compressed, dtype=kernel_spec.precision))
            )
            counts.append(basis.shape[0])
            offset += basis.shape[0]
        super().__init__(
            {
                "weight": jnp.zeros(
                    (kernel_spec.num_elements, offset, kernel_spec.num_features),
                    dtype=kernel_spec.precision,
                ),
                "path_counts": jnp.asarray(counts, dtype=jnp.int64),
            }
        )

    def load_canonical(self, state: Mapping[str, Any]) -> None:
        if not np.array_equal(
            np.asarray(state["path_counts"]), np.asarray(self.tensor("path_counts"))
        ):
            raise ValueError(
                "symmetric_contraction path_counts disagree with the named basis"
            )
        super().load_canonical(state)

    def initialize_weights(self, seed: int) -> None:
        super().initialize_weights(seed)
        weight = self.tensor("weight")
        for _, offset, count, _ in self.groups:
            if count:
                weight = weight.at[:, offset : offset + count, :].divide(count)
        self.state["weight"].set_value(weight)

    def __call__(self, features, elements):
        channels = per_channel(
            features, self.kernel_spec.irreps_in, self.kernel_spec.num_features
        )
        results = [
            jnp.zeros(
                (features.shape[0], self.kernel_spec.num_features, ir.dimension),
                dtype=features.dtype,
            )
            for _, ir in Irreps.parse(self.kernel_spec.irreps_out)
        ]
        cached = {}
        for index, (target, offset, count, monomials) in enumerate(self.groups):
            if not count:
                continue
            order = len(monomials[0])
            if order not in cached:
                cached[order] = jnp.prod(
                    channels[:, :, jnp.asarray(monomials)], axis=-1
                )
            projection = jnp.einsum(
                "ncm,pom->ncpo", cached[order], self.coefficients[index].get_value()
            )
            weights = self.tensor("weight")[elements, offset : offset + count, :]
            results[target] = results[target] + jnp.einsum(
                "ncpo,npc->nco", projection, weights
            )
        return grouped(jnp.concatenate(results, axis=-1), self.kernel_spec.irreps_out)


class RadialMLP(CanonicalOp):
    def __init__(self, kernel_spec: RadialMLPSpec):
        if kernel_spec.kind != "normalized_silu":
            raise ValueError(f"JAX radial_mlp does not support {kernel_spec.kind!r}")
        self.widths = tuple(
            zip(kernel_spec.channels[:-1], kernel_spec.channels[1:], strict=True)
        )
        super().__init__(
            {
                f"layer_{index}": jnp.zeros((left, right), dtype=kernel_spec.precision)
                for index, (left, right) in enumerate(self.widths)
            }
        )

    def __call__(self, features):
        for index, (left, _) in enumerate(self.widths):
            features = features @ (self.tensor(f"layer_{index}") * left**-0.5)
            if index + 1 != len(self.widths):
                features = SILU_SECOND_MOMENT_SCALE * jax.nn.silu(features)
        return features
