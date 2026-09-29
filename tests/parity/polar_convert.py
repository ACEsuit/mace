"""Reading a legacy charge-aware model's weights into the rewrite's blocks.

The energy half is the anchor's conversion, unchanged: the backbone, the
readouts and the energy constants go through :mod:`tests.parity.fm00_convert`.
What is added here is the charge-aware half, block by block.

Two layouts differ and are converted rather than assumed:

* An equivariant linear, with or without a bias, as for the anchor: the
  per-instruction normalization is folded into the weights.
* The frozen tree's sparse channel-pair products store one flat weight over
  their instructions and multiply each path by a normalization inside the
  forward. The rewrite's hold the product of the two, path by path, and keep
  only the Clebsch-Gordan factor in the op.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from mace_core.clebsch_gordan.irreps import Irreps

from tests.parity.fm00_convert import linear_weights_to_canonical, transfer_weights

__all__ = ["polar_backbone_config", "polar_settings", "transfer_polar_weights"]


def polar_backbone_config(legacy, build_config) -> dict:
    """The backbone settings of a legacy charge-aware model.

    ``build_config`` is the anchor's reader. It reads the node declaration off
    the first product, which in a charge-aware model is already the full
    one, so it serves unchanged.
    """
    config = build_config(legacy)
    config.pop("pair_repulsion")
    # The anchor's reader takes the envelope order from the repulsion, which a
    # charge-aware model does not carry; the radial envelope has the same one.
    config["cutoff_order"] = int(legacy.radial_embedding.cutoff_fn.p)
    config["element_agnostic_product"] = bool(legacy.products[0].use_agnostic_product)
    return config


def polar_settings(legacy, **overrides):
    """The charge-aware settings, read off the trained model."""
    from mace_torch.models.electrostatics import PolarSettings

    widths = tuple(float(width) for width in legacy.field_feature_widths)
    norms = legacy.field_feature_norms.detach().tolist()
    # The buffer holds one entry per component; the settings hold one per
    # order and width.
    per_block, offset = [], 0
    for degree in range(legacy.field_feature_max_l + 1):
        for _ in widths:
            per_block.append(float(norms[offset]))
            offset += 2 * degree + 1
    fukui_hidden = Irreps.parse(str(legacy.fukui_source_map.linear_1.irreps_out))
    return PolarSettings(
        multipole_max_l=int(legacy.atomic_multipoles_max_l),
        multipole_width=float(legacy.atomic_multipoles_smearing_width),
        feature_max_l=int(legacy.field_feature_max_l),
        feature_widths=widths,
        feature_norms=tuple(per_block),
        num_recursion_steps=int(legacy.num_recursion_steps),
        kspace_cutoff_factor=float(legacy.kspace_cutoff_factor),
        feature_self_interaction=bool(legacy.field_si),
        energy_self_interaction=bool(legacy.include_electrostatic_self_interaction),
        add_local_electron_energy=bool(legacy.add_local_electron_energy),
        quadrupole_feature_corrections=bool(legacy.quadrupole_feature_corrections),
        fukui_hidden=fukui_hidden.dimension,
        **overrides,
    )


def _linear(source, destination) -> None:
    # A biased e3nn linear lists its biases among its instructions, with no
    # input, and holds their values apart from the weights.
    weights = SimpleNamespace(
        weight=source.weight,
        instructions=[ins for ins in source.instructions if ins.i_in >= 0],
    )
    destination.weight.copy_(
        torch.tensor(
            linear_weights_to_canonical(
                weights, str(source.irreps_in), str(source.irreps_out)
            ),
            dtype=destination.weight.dtype,
        )
    )
    bias = getattr(source, "bias", None)
    if bias is not None and bias.numel():
        destination.bias.copy_(bias.detach().to(destination.bias.dtype))
    elif destination.bias.numel():
        destination.bias.zero_()


def _paths(legacy_product):
    """Each path's weight block with its normalization folded in, keyed by
    where its two inputs start."""
    if not bool((legacy_product.output_mask == 1).all()):
        raise ValueError("a masked output: this layout assumes every one is reached")
    blocks = {}
    for (
        in1_start,
        _,
        in2_start,
        _,
        _,
        _,
        w_start,
        w_stop,
        path_weight,
        _,
        mul1,
        mul2,
        _,
    ) in legacy_product._path_meta:
        blocks[(in1_start, in2_start)] = (
            legacy_product.weight[w_start:w_stop].view(mul1, mul2) * path_weight
        )
    return blocks


def _invariant_products(source, destination) -> None:
    blocks = _paths(source)
    pieces = [
        blocks.pop((left_start, right_start))
        for _, _, left_start, right_start, _ in destination.paths
    ]
    if blocks:
        raise ValueError(f"paths the rewrite has no place for: {sorted(blocks)}")
    destination.weight.copy_(torch.cat([piece.reshape(-1) for piece in pieces]))


def _scalar_modulation(source, destination) -> None:
    blocks = _paths(source)
    pieces, offset = [], 0
    for mul, dimension in destination.spans:
        pieces.append(blocks.pop((offset, 0)).reshape(-1))
        offset += mul * dimension
    if blocks:
        raise ValueError(f"paths the rewrite has no place for: {sorted(blocks)}")
    destination.weight.copy_(torch.cat(pieces))


def _bias_readout(source, destination) -> None:
    _linear(source.linear_1, destination.first)
    _linear(source.linear_mid, destination.middle)
    _linear(source.linear_2, destination.last)


def transfer_polar_weights(legacy, model, correlation: int) -> None:
    """Copy every trained number of a charge-aware model into the rewrite."""
    transfer_weights(legacy, model, correlation)
    with torch.no_grad():
        for source, destination in zip(
            legacy.lr_source_maps, model.source_maps, strict=True
        ):
            _linear(source.linear, destination)
        for source, destination in zip(
            legacy.layer_feature_mixer.linears, model.layer_mixer, strict=True
        ):
            _linear(source, destination)
        _bias_readout(legacy.fukui_source_map, model.fukui_readout)
        for source, destination in zip(
            legacy.field_dependent_charges_maps, model.updates, strict=True
        ):
            embedding = source.potential_embedding
            _linear(embedding.potential_linear, destination.from_potential)
            _linear(embedding.node_feats_linear, destination.from_features)
            _linear(embedding.charge_embedding, destination.from_density)
            _linear(source.source_embedding, destination.element_embedding)
            _invariant_products(source.dot_products, destination.products)
            destination.mlp.net.load_state_dict(source.nonlinearity.net.state_dict())
            _scalar_modulation(source.tp_out, destination.modulation)
            _bias_readout(source.readout, destination.readout)
        readout = legacy.local_electron_energy
        _linear(readout.linear_up_q, model.electron_energy.from_density)
        _linear(readout.linear_up_v, model.electron_energy.from_potential)
        _invariant_products(
            readout.dot_products_q, model.electron_energy.density_products
        )
        _invariant_products(
            readout.dot_products_v, model.electron_energy.potential_products
        )
        model.electron_energy.mlp.net.load_state_dict(readout.mlp.net.state_dict())
