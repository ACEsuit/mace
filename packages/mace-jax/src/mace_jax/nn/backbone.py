"""Descriptor-built backbone: node features out, without readouts or gradients."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from mace_core.kernels.canonical import KERNEL_SPEC_VERSION, contraction_path_labels
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialMLPDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)

from mace_jax.kernels.ops import CanonicalOp, expanded
from mace_jax.nn.radial import agnesi, bessel, polynomial_envelope


class Interaction(nnx.Module):
    def __init__(
        self,
        backend,
        config,
        node_irreps,
        message_irreps,
        output_irreps,
        elements,
        kind,
        dtype,
    ):
        channels = config.num_channels
        self.residual = kind in {"residual", "density_residual"}
        self.density = kind in {"density", "density_residual"}
        self.envelope_outside = not config.apply_cutoff
        self.channels = channels
        node = expanded(node_irreps, channels)
        message = expanded(message_irreps, channels)
        descriptor = ChannelwiseTPConvDescriptor(
            irreps_node=node_irreps,
            irreps_edge=message_irreps,
            irreps_out=message_irreps,
            num_features=channels,
            precision=dtype,
        )
        self.num_paths = len(descriptor.paths)
        self.convolution = backend.make_channelwise_tp_conv(descriptor)
        self.linear_up = backend.make_linear(
            LinearDescriptor(irreps_in=node, irreps_out=node, precision=dtype)
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=expanded(descriptor.irreps_paths, channels),
                irreps_out=message,
                precision=dtype,
            )
        )
        self.radial_mlp = backend.make_radial_mlp(
            RadialMLPDescriptor(
                channels=(
                    config.radial_basis.num_basis,
                    *config.radial_mlp,
                    self.num_paths * channels,
                ),
                kind="normalized_silu",
                precision=dtype,
            )
        )
        self.density_mlp = (
            backend.make_radial_mlp(
                RadialMLPDescriptor(
                    channels=(config.radial_basis.num_basis, 1),
                    kind="normalized_silu",
                    precision=dtype,
                )
            )
            if self.density
            else None
        )
        self.reduce = backend.make_segment_reduce(
            SegmentReduceDescriptor(reduction="sum", num_features=1, precision=dtype)
        )
        self.normalization = CanonicalOp({"neighbours": jnp.asarray(1.0, dtype=dtype)})
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=node if self.residual else message,
                irreps_in2=f"{elements}x0e",
                irreps_out=expanded(output_irreps, channels)
                if self.residual
                else message,
                precision=dtype,
            )
        )

    def __call__(
        self, features, attributes, radial, envelope, edge_attributes, sender, receiver
    ):
        weights = self.radial_mlp(radial)
        if self.envelope_outside:
            weights = weights * envelope
        message = self.convolution(
            self.linear_up(features),
            edge_attributes,
            weights.reshape(weights.shape[0], self.num_paths, self.channels),
            sender,
            receiver,
            features.shape[0],
        )
        if self.density_mlp is None:
            denominator = self.normalization.tensor("neighbours")
        else:
            density = jnp.tanh(self.density_mlp(radial) ** 2)
            if self.envelope_outside:
                density = density * envelope
            denominator = 1 + self.reduce(density, receiver, features.shape[0])
        message = self.linear(message) / denominator
        return (
            (message, self.skip(features, attributes))
            if self.residual
            else (self.skip(message, attributes), None)
        )

    def canonical_components(self, prefix):
        values = {prefix + ".body": self.normalization, prefix + ".skip": self.skip}
        for name in ("linear_up", "linear", "radial_mlp", "density_mlp"):
            op = getattr(self, name)
            if op is not None:
                values[prefix + ".body." + name] = op
        return values


class ProductBasis(nnx.Module):
    def __init__(self, backend, config, message, output, elements, dtype):
        self.layout = {
            "spec_version": KERNEL_SPEC_VERSION,
            "basis": "reduced",
            "paths": list(contraction_path_labels(message, output, config.correlation)),
        }
        self.contraction = backend.make_symmetric_contraction(
            SymmetricContractionDescriptor(
                irreps_in=message,
                irreps_out=output,
                correlation=config.correlation,
                num_elements=elements,
                num_features=config.num_channels,
                basis="reduced",
                precision=dtype,
            )
        )
        flat = expanded(output, config.num_channels)
        self.linear = backend.make_linear(
            LinearDescriptor(irreps_in=flat, irreps_out=flat, precision=dtype)
        )

    def __call__(self, message, elements, skip):
        result = self.linear(self.contraction(message, elements))
        return result if skip is None else result + skip


class MACEBackbone(nnx.Module):
    def __init__(self, backend, config, elements, dtype):
        import math

        import ase.data

        self.config = config
        self.embedding = backend.make_linear(
            LinearDescriptor(
                irreps_in=f"{elements}x0e",
                irreps_out=f"{config.num_channels}x0e",
                precision=dtype,
            )
        )
        self.harmonics = backend.make_spherical_harmonics(
            SphericalHarmonicsDescriptor(lmax=config.max_ell, precision=dtype)
        )
        self.basis = CanonicalOp(
            {
                "frequencies": jnp.arange(
                    1, config.radial_basis.num_basis + 1, dtype=dtype
                )
                * (math.pi / config.r_max),
                "prefactor": jnp.asarray(math.sqrt(2 / config.r_max), dtype=dtype),
            }
        )
        self.transform = (
            CanonicalOp(
                {
                    "amplitude": jnp.asarray(1.0, dtype=dtype),
                    "exponent_p": jnp.asarray(4.0, dtype=dtype),
                    "exponent_q": jnp.asarray(4.0, dtype=dtype),
                    "covalent_radii": jnp.asarray(ase.data.covalent_radii, dtype=dtype),
                }
            )
            if config.distance_transform.kind == "agnesi"
            else None
        )
        message = "+".join(
            f"{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(config.max_ell + 1)
        )
        self.interactions, self.products = nnx.List(), nnx.List()
        self.layer_irreps = []
        for index in range(config.num_interactions):
            source = "0e" if index == 0 else config.node_irreps
            output = (
                "0e" if index == config.num_interactions - 1 else config.node_irreps
            )
            kind = config.interaction_first if index == 0 else config.interaction
            self.interactions.append(
                Interaction(
                    backend, config, source, message, output, elements, kind, dtype
                )
            )
            self.products.append(
                ProductBasis(
                    backend,
                    config,
                    message,
                    output,
                    1 if config.use_agnostic_product else elements,
                    dtype,
                )
            )
            self.layer_irreps.append(expanded(output, config.num_channels))

    def __call__(self, graph):
        sender, receiver = graph["edge_index"]
        vectors = (
            graph["positions"][receiver] - graph["positions"][sender] + graph["shifts"]
        )
        lengths = jnp.sqrt(jnp.sum(vectors**2, axis=-1, keepdims=True))
        envelope = polynomial_envelope(
            lengths, self.config.r_max, self.config.cutoff.polynomial_order
        )
        transformed = (
            lengths
            if self.transform is None
            else agnesi(
                lengths,
                graph["node_atomic_numbers"],
                graph["edge_index"],
                self.transform.tensor,
            )
        )
        radial = bessel(
            transformed,
            self.basis.tensor("frequencies"),
            self.basis.tensor("prefactor"),
        )
        if self.config.apply_cutoff:
            radial = radial * envelope
        attributes = graph["node_attrs"]
        features = self.embedding(attributes)
        elements = jnp.argmax(attributes, axis=-1)
        if self.config.use_agnostic_product:
            elements = jnp.zeros_like(elements)
        angular = self.harmonics(vectors)
        layers = []
        for interaction, product in zip(self.interactions, self.products, strict=True):
            message, skip = interaction(
                features, attributes, radial, envelope, angular, sender, receiver
            )
            features = product(message, elements, skip)
            layers.append(features)
        return tuple(layers), lengths

    def canonical_components(self):
        prefix = "backbone.backbone"
        result = {
            prefix + ".node_embedding": self.embedding,
            prefix + ".radial_embedding.basis": self.basis,
        }
        if self.transform is not None:
            result[prefix + ".radial_embedding.distance_transform"] = self.transform
        for index, (interaction, product) in enumerate(
            zip(self.interactions, self.products, strict=True)
        ):
            result.update(
                interaction.canonical_components(f"{prefix}.interactions.{index}")
            )
            result[f"{prefix}.products.{index}.contraction"] = product.contraction
            result[f"{prefix}.products.{index}.linear"] = product.linear
        return result

    def canonical_layout(self):
        """Canonical semantics are model state, independent of the selected backend."""
        prefix = "backbone.backbone"
        result = {
            prefix + ".radial_embedding.basis": {"model.radial_basis.kind": "bessel"}
        }
        if self.transform is not None:
            result[prefix + ".radial_embedding.distance_transform"] = {
                "model.distance_transform.kind": "agnesi"
            }
        for index, (interaction, product) in enumerate(
            zip(self.interactions, self.products, strict=True)
        ):
            result[f"{prefix}.products.{index}.contraction"] = product.layout
            for name in ("radial_mlp", "density_mlp"):
                if getattr(interaction, name) is not None:
                    result[f"{prefix}.interactions.{index}.body.{name}"] = {
                        "model.radial_mlp": "normalized_silu"
                    }
        return result
