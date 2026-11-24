"""Carrying legacy weights into the reference backend's canonical layout.

Shared by the parity suites that copy a frozen module's weights into a v1 one
and compare the two. Each map is derived from the legacy module's own
instruction list rather than from the reference's code, so a reference that
stored its weights in another order fails the comparison instead of agreeing
with itself.
"""

import itertools

import numpy as np
import torch
from e3nn import o3
from mace_torch.nn.backbone import MACEBackbone

from tests.golden.harness import tolerance

CLOSED_FORM_ATOL = tolerance("closed_form_fp64").atol


def canonical_linear_index(irreps_in: o3.Irreps, irreps_out: o3.Irreps) -> dict:
    """``(i_out, copy_out, i_in, copy_in) -> canonical position``.

    The canonical order as the layout states it: output copies outermost, and
    within one output copy the matching input copies in declaration order.
    """
    index, position = {}, 0
    for i_out, (multiplicity_out, irrep_out) in enumerate(irreps_out):
        for copy_out in range(multiplicity_out):
            for i_in, (multiplicity_in, irrep_in) in enumerate(irreps_in):
                if irrep_in != irrep_out:
                    continue
                for copy_in in range(multiplicity_in):
                    index[(i_out, copy_out, i_in, copy_in)] = position
                    position += 1
    return index


def linear_permutation(legacy: o3.Linear) -> torch.Tensor:
    """For each flat legacy weight, the canonical weight it is."""
    canonical = canonical_linear_index(legacy.irreps_in, legacy.irreps_out)
    order = []
    for instruction in legacy.instructions:
        if instruction.i_in == -1:
            continue
        multiplicity_in, multiplicity_out = instruction.path_shape
        for copy_in in range(multiplicity_in):
            for copy_out in range(multiplicity_out):
                order.append(
                    canonical[(instruction.i_out, copy_out, instruction.i_in, copy_in)]
                )
    return torch.tensor(order)


def skip_permutation(legacy: o3.FullyConnectedTensorProduct) -> torch.Tensor:
    """For each flat legacy weight, its flat position in ``[scalar, weight]``."""
    canonical = canonical_linear_index(legacy.irreps_in1, legacy.irreps_out)
    per_scalar = len(canonical)
    scalar_offsets = np.cumsum([0] + [mul for mul, _ in legacy.irreps_in2])
    order = []
    for instruction in legacy.instructions:
        multiplicity_1, multiplicity_2, multiplicity_out = instruction.path_shape
        for copy_1 in range(multiplicity_1):
            for copy_2 in range(multiplicity_2):
                scalar = int(scalar_offsets[instruction.i_in2]) + copy_2
                for copy_out in range(multiplicity_out):
                    weight = canonical[
                        (instruction.i_out, copy_out, instruction.i_in1, copy_1)
                    ]
                    order.append(scalar * per_scalar + weight)
    return torch.tensor(order)


def legacy_contraction_blocks(
    contraction,
) -> list[tuple[torch.Tensor, torch.nn.Parameter, int]]:
    """``(coupling matrix, weight, body order)`` for every legacy block."""
    blocks = [
        (
            contraction.U_tensors(contraction.correlation),
            contraction.weights_max,
            contraction.correlation,
        )
    ]
    for position, weight in enumerate(contraction.weights):
        order = contraction.correlation - 1 - position
        blocks.append((contraction.U_tensors(order), weight, order))
    return blocks


def symmetric_part(tensor: np.ndarray, order: int) -> np.ndarray:
    """Average over permutations of the ``order`` input axes after axis 0."""
    total = np.zeros_like(tensor)
    permutations = list(itertools.permutations(range(1, order + 1)))
    for permutation in permutations:
        total += np.transpose(tensor, (0, *permutation))
    return total / len(permutations)


def legacy_to_reference_map(
    coupling: torch.Tensor, basis: torch.Tensor, order: int
) -> np.ndarray:
    """``M[p, q]`` with ``sum_q M[p, q] basis_q = sym(coupling_p)``.

    Legacy contracts its coupling matrices with a symmetric power of the
    features, so only their symmetric part acts. The reference basis is
    symmetric already, so a legacy weight ``w_p`` is the reference weight
    ``sum_p M[p, q] w_p``.
    """
    legacy = coupling.detach().numpy()
    if legacy.ndim == order + 1:  # a scalar output carries no output axis
        legacy = legacy[None]
    legacy = np.moveaxis(legacy, -1, 0)
    targets = np.stack([symmetric_part(path, order).reshape(-1) for path in legacy])
    flat_basis = basis.detach().numpy().reshape(basis.shape[0], -1)
    solution, *_ = np.linalg.lstsq(flat_basis.T, targets.T, rcond=None)
    np.testing.assert_allclose(
        flat_basis.T @ solution, targets.T, atol=CLOSED_FORM_ATOL
    )
    return solution.T


#: The legacy class of each interaction kind.
LEGACY_CLASS = {
    "plain": "RealAgnosticInteractionBlock",
    "residual": "RealAgnosticResidualInteractionBlock",
    "density": "RealAgnosticDensityInteractionBlock",
    "density_residual": "RealAgnosticDensityResidualInteractionBlock",
    "nonlinear": "RealAgnosticResidualNonLinearInteractionBlock",
}


# ---------------------------------------------------------------------------
# Carrying whole modules
# ---------------------------------------------------------------------------


def carry_linear(legacy: o3.Linear, op) -> None:
    assert str(legacy.irreps_in) == op.kernel_spec.irreps_in, (
        legacy.irreps_in,
        op.kernel_spec.irreps_in,
    )
    assert str(legacy.irreps_out) == op.kernel_spec.irreps_out, (
        legacy.irreps_out,
        op.kernel_spec.irreps_out,
    )
    state = op.to_canonical()
    weight = torch.zeros_like(state["weight"])
    weight[linear_permutation(legacy)] = legacy.weight.detach()
    op.load_canonical({"weight": weight, "bias": state["bias"]})


def _carry_skip(legacy: o3.FullyConnectedTensorProduct, op) -> None:
    assert str(legacy.irreps_in1) == op.kernel_spec.irreps_in1
    assert str(legacy.irreps_out) == op.kernel_spec.irreps_out
    weight = torch.zeros_like(op.to_canonical()["weight"])
    weight.view(-1)[skip_permutation(legacy)] = legacy.weight.detach()
    op.load_canonical({"weight": weight})


def _carry_fully_connected_net(legacy, op) -> None:
    layers = len(op.kernel_spec.channels) - 1
    op.load_canonical(
        {
            f"layer_{i}": getattr(legacy, f"layer{i}").weight.detach()
            for i in range(layers)
        }
    )


def _carry_radial_mlp(legacy, op) -> None:
    linears = [m for m in legacy.net if isinstance(m, torch.nn.Linear)]
    norms = [m for m in legacy.net if isinstance(m, torch.nn.LayerNorm)]
    state = {}
    for i, linear in enumerate(linears):
        state[f"layer_{i}"] = linear.weight.detach().T
        state[f"bias_{i}"] = linear.bias.detach()
    for i, norm in enumerate(norms):
        state[f"norm_weight_{i}"] = norm.weight.detach()
        state[f"norm_bias_{i}"] = norm.bias.detach()
    op.load_canonical(state)


def _carry_contraction(legacy, op) -> None:
    correlation = op.kernel_spec.correlation
    with torch.no_grad():
        for position, contraction in enumerate(legacy.contractions):
            tables = list(op.bases[position])
            for coupling, weight, order in legacy_contraction_blocks(contraction):
                if weight.numel() == 0:
                    continue
                target = op.weights[position * correlation + order - 1]
                mapping = torch.from_numpy(
                    legacy_to_reference_map(coupling, tables[order - 1], order)
                )
                target.copy_(torch.einsum("pq,zpc->zqc", mapping, weight))


def _carry_interaction(legacy, block) -> None:
    name = type(legacy).__name__
    if name == LEGACY_CLASS["nonlinear"]:
        carry_linear(legacy.source_embedding, block.sender_embedding)
        carry_linear(legacy.target_embedding, block.receiver_embedding)
        carry_linear(legacy.linear_up, block.linear_up)
        carry_linear(legacy.skip_tp, block.skip)
        carry_linear(legacy.linear_res, block.linear_residual)
        carry_linear(legacy.linear_1, block.linear_mix)
        carry_linear(legacy.linear_2, block.linear_out)
        _carry_radial_mlp(legacy.conv_tp_weights, block.radial_mlp)
        _carry_radial_mlp(legacy.density_fn, block.density_mlp)
        block.load_canonical(
            {"alpha": legacy.alpha.detach(), "beta": legacy.beta.detach()}
        )
        return
    body = block.body
    carry_linear(legacy.linear_up, body.linear_up)
    _carry_fully_connected_net(legacy.conv_tp_weights, body.radial_mlp)
    carry_linear(legacy.linear, body.linear)
    _carry_skip(legacy.skip_tp, block.skip)
    if hasattr(legacy, "density_fn"):
        _carry_fully_connected_net(legacy.density_fn, body.density_mlp)
    if body.neighbours is not None:
        with torch.no_grad():
            body.neighbours.fill_(float(legacy.avg_num_neighbors))


def _carry_element_embedding(legacy: o3.Linear, embedding) -> None:
    """Legacy's ``Nx0e -> Cx0e`` linear holds one weight per element and
    channel, ``[N, C]`` flattened, which is the element embedding's matrix."""
    state = embedding.to_canonical()
    embedding.load_canonical(
        {
            "atomic_numbers": state["atomic_numbers"],
            "weight": legacy.weight.detach().reshape(state["weight"].shape),
        }
    )


def carry_weights(legacy, backbone: MACEBackbone) -> None:
    """Every weight of the modules the backbone rebuilds, legacy to v1."""
    _carry_element_embedding(legacy.node_embedding.linear, backbone.element_embedding)
    for legacy_block, block in zip(
        legacy.interactions, backbone.interactions, strict=True
    ):
        _carry_interaction(legacy_block, block)
    for legacy_product, product in zip(legacy.products, backbone.products, strict=True):
        _carry_contraction(legacy_product.symmetric_contractions, product.contraction)
        carry_linear(legacy_product.linear, product.linear)


def carry_readout(legacy, readout) -> None:
    """A legacy model's readouts into a v1 readout, layer by layer.

    The legacy model holds one readout per layer, or one on the last layer
    alone when it reads out only that one. A gated v1 map takes the legacy
    readout's two linear maps, a linear one its one.
    """
    every_layer = len(legacy.readouts) == len(legacy.interactions)
    for layer, readout_map in zip(readout.layers, readout.maps, strict=True):
        legacy_readout = legacy.readouts[layer if every_layer else -1]
        if hasattr(legacy_readout, "linear_1"):
            carry_linear(legacy_readout.linear_1, readout_map.linear_in)
            carry_linear(legacy_readout.linear_2, readout_map.linear_out)
        else:
            carry_linear(legacy_readout.linear, readout_map)


def carry_joint_embedding(legacy, embedding) -> None:
    """The frozen tree's ``GenericJointEmbedding`` into a v1
    ``GraphFeatureEmbedding`` of the same features: each feature's table or
    two maps, and the projection."""
    ours = {
        feature.name: module
        for feature, module in zip(
            embedding.features, embedding.embeddings, strict=True
        )
    }
    for name, embedder in legacy.embedders.items():
        target = ours[name]
        if isinstance(embedder, torch.nn.Embedding):
            target.load_canonical({"table": embedder.weight.detach()})
            continue
        first, _, second = embedder
        state = {
            "weight_in": first.weight.detach(),
            "weight_out": second.weight.detach(),
        }
        if first.bias is not None:
            state |= {"bias_in": first.bias.detach(), "bias_out": second.bias.detach()}
        target.load_canonical(state)
    embedding.load_canonical({"projection": legacy.project[0].weight.detach()})
