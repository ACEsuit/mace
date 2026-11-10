"""Contract doubles composed exclusively from the existing reference modules."""

from dataclasses import replace

import torch
from mace_core.config import AgnesiTransformConfig, SoftTransformConfig
from mace_core.kernels.inference import InteractionFusionSpec, ProductReadoutSpec
from mace_core.kernels.spans import EdgeConvolutionSpan
from mace_core.kernels.specs import (
    ChannelwiseTPConvSpec,
    DistanceTransformSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    RadialBasisSpec,
    RadialMLPSpec,
    SphericalHarmonicsSpec,
    SymmetricContractionSpec,
)
from mace_torch.backends.composite import CompositeBackend
from mace_torch.backends.inference import InferencePreparation
from mace_torch.backends.layout import expanded_irreps
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import SpanCapabilities
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.embedding import RadialEmbeddingBlock, build_distance_transform
from mace_torch.nn.interaction import InteractionBlock, ResidualInteractionBlock
from mace_torch.serialization import canonical_state, load_canonical_state
from torch import nn


def product_spec(layout="mul_ir", residual=True, readout=True):
    policy = dict(
        precision=str(torch.get_default_dtype()).removeprefix("torch."),
        activation_layout=layout,
    )
    return ProductReadoutSpec(
        SymmetricContractionSpec(
            irreps_in="0e+1o",
            irreps_out="0e+1o",
            num_features=2,
            num_elements=2,
            correlation=2,
            **policy,
        ),
        LinearSpec(irreps_in="2x0e+2x1o", irreps_out="2x0e+2x1o", **policy),
        residual=residual,
        readout=LinearSpec(irreps_in="2x0e+2x1o", irreps_out="2x0e", **policy)
        if readout
        else None,
        interface_dtype=policy["precision"],
    )


def interaction_spec(
    layout="mul_ir", residual=True, density=False, cutoff=True, transform=None
):
    product = product_spec(layout, residual)
    policy = dict(precision=product.linear.precision, activation_layout=layout)
    convolution = ChannelwiseTPConvSpec(
        irreps_node="0e+1o",
        irreps_edge="0e+1o",
        irreps_out="0e+1o",
        num_features=2,
        **policy,
    )
    return InteractionFusionSpec(
        EdgeConvolutionSpan(
            RadialBasisSpec(num_basis=3, apply_cutoff=cutoff, **policy),
            SphericalHarmonicsSpec(lmax=1, **policy),
            RadialMLPSpec(channels=(3, 4, 2 * len(convolution.paths)), **policy),
            convolution,
            distance_transform=DistanceTransformSpec(kind=transform, **policy)
            if transform
            else None,
            density=RadialMLPSpec(channels=(3, 1), **policy) if density else None,
        ),
        product.linear,
        LinearSpec(
            irreps_in=expanded_irreps(convolution.irreps_paths, 2),
            irreps_out="2x0e+2x1o",
            **policy,
        ),
        FullyConnectedTPSpec(
            irreps_in1="2x0e+2x1o",
            irreps_in2="2x0e",
            irreps_out="2x0e+2x1o",
            **policy,
        ),
        product,
    )


class ProductComposition(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        backend = ReferenceBackend()
        self.contraction = backend.make_symmetric_contraction(spec.contraction)
        self.linear = backend.make_linear(spec.linear)
        self.readout = (
            None if spec.readout is None else backend.make_linear(spec.readout)
        )

    def roles(self):
        return {name: getattr(self, name) for name in self.spec.weight_roles}

    def forward(self, inputs):
        features = self.linear(self.contraction(inputs[0], inputs[1]))
        if self.spec.residual:
            features = features + inputs[2]
        outputs = (features,) if self.spec.return_features else ()
        return outputs + (() if self.readout is None else (self.readout(features),))


class InteractionComposition(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

        class LayoutBackend(ReferenceBackend):
            def capabilities(self):
                return replace(
                    super().capabilities(),
                    native_activation_layout=spec.linear_up.activation_layout
                    or "mul_ir",
                )

        backend = CompositeBackend(LayoutBackend(), ReferenceBackend())
        shape = spec.edge
        transform = None
        if shape.distance_transform is not None:
            transform = build_distance_transform(
                AgnesiTransformConfig()
                if shape.distance_transform.kind == "agnesi"
                else SoftTransformConfig()
            )
        self.radial = RadialEmbeddingBlock(
            backend.make_radial_basis(shape.radial_basis), transform
        )
        self.harmonics = backend.make_spherical_harmonics(shape.spherical_harmonics)
        block = ResidualInteractionBlock if spec.product.residual else InteractionBlock
        self.interaction = block(
            backend,
            irreps_node="0e+1o",
            irreps_edge="0e+1o",
            irreps_target="0e+1o",
            num_radial=3,
            num_features=2,
            num_elements=2,
            avg_num_neighbors=4.5,
            radial_hidden=(4,),
            precision=spec.linear_up.precision,
            learned_density=shape.density is not None,
            envelope_outside=not shape.radial_basis.apply_cutoff,
            **({"irreps_skip_out": "0e+1o"} if spec.product.residual else {}),
        )
        self.product = ProductComposition(spec.product)

    def roles(self):
        body = self.interaction.body
        members = {
            "linear_up": body.linear_up,
            "radial_basis": self.radial.radial,
            "radial_network": body.radial_mlp,
            "linear_down": body.linear,
            "skip": self.interaction.skip,
            **{f"product.{key}": value for key, value in self.product.roles().items()},
        }
        if self.spec.edge.density is not None:
            members["density"] = body.density_mlp
        if self.radial.distance_transform is not None:
            members["distance_transform"] = self.radial.distance_transform
        return members

    def forward(self, inputs):
        nodes, vectors, attributes, numbers, element, sender, receiver = inputs
        edges = EdgeInputs(
            vectors,
            numbers,
            torch.stack((sender, receiver)),
            self.harmonics,
            self.radial,
        )
        message, skip = self.interaction(
            nodes, edges, attributes, sender, receiver, len(nodes)
        )
        return self.product(
            (message, element) if skip is None else (message, element, skip)
        )


def composition(spec):
    model = (
        ProductComposition(spec)
        if isinstance(spec, ProductReadoutSpec)
        else InteractionComposition(spec)
    )
    initialize_model_weights(model, 17)
    return model.eval()


def region_weights(model):
    weights = {role: canonical_state(member) for role, member in model.roles().items()}
    if (
        isinstance(model.spec, InteractionFusionSpec)
        and model.spec.edge.density is None
    ):
        weights["normalization"] = {"": model.interaction.body.to_canonical()}
    return weights


class PreparedComposition(nn.Module):
    capabilities = SpanCapabilities(True, False)

    def __init__(self, template):
        super().__init__()
        self.template = template
        entries = [*template.named_parameters(), *template.named_buffers()]
        self.names = tuple(name for name, _ in reversed(entries))
        self.preparation_tensors = tuple(
            value.detach().clone() for _, value in reversed(entries)
        )
        for name, _ in entries:
            path, _, attribute = name.rpartition(".")
            owner = template.get_submodule(path)
            if attribute in owner._parameters:
                owner.register_parameter(attribute, None)
            else:
                owner.register_buffer(attribute, None)

    def forward(self, inputs, tensors):
        state = dict(zip(self.names, tensors, strict=True))
        return torch.func.functional_call(self.template, state, (inputs,))


class CompositionBackend(ReferenceBackend):
    name = "test_inference_composition"

    def __init__(self):
        self.calls = 0

    def prepare_inference(self, spec, weights):
        self.calls += 1
        model = composition(spec)
        for role, module in model.roles().items():
            load_canonical_state(module, weights[role])
        if "normalization" in weights:
            model.interaction.body.load_canonical(weights["normalization"][""])
        operation = PreparedComposition(model)
        tensors = operation.preparation_tensors
        del operation.preparation_tensors
        return InferencePreparation(operation, tensors)


def inputs(spec, count=4):
    generator = torch.Generator().manual_seed(57 + count)
    nodes = torch.randn(count, 8, generator=generator)
    element = torch.arange(count) % 2
    if isinstance(spec, ProductReadoutSpec):
        return (nodes, element) + ((nodes * 0.3,) if spec.residual else ())
    positions = torch.randn(count, 3, generator=generator)
    sender = torch.arange(count).repeat_interleave(count)
    receiver = torch.arange(count).repeat(count)
    mask = sender != receiver
    sender, receiver = sender[mask], receiver[mask]
    return (
        nodes,
        positions[receiver] - positions[sender],
        torch.nn.functional.one_hot(element, 2).to(nodes.dtype),
        torch.where(element == 0, 1, 8),
        element,
        sender,
        receiver,
    )
