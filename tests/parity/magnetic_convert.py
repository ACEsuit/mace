"""Reading a legacy magnetic model's weights into the rewrite.

The operators are the ones the standard converter already reads, a linear, a
skip against the element attributes, a symmetric contraction and the radial
networks, and each is read the same way. What is added here is where each sits
in the magnetic blocks, and the constants the magnetic model carries beside the
standard ones: the moment saturations and the one-body coefficients.

The radial networks are copied as they are. Both trees scale each layer by the
inverse root of its fan-in inside the forward, and both lay the output out one
path at a time with the channels inside, in the paths' sorted order.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models.energy import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.magnetic import MagneticModel

from tests.parity.fm00_convert import (
    contraction_weights_to_canonical,
    copy_linear,
    energy_constants_to_canonical,
    fully_connected_tp_weights_to_canonical,
)

__all__ = ["convert_magnetic", "magnetic_config", "transfer_magnetic_weights"]


def _per_channel(irreps: str) -> str:
    """One channel's declaration of a grouped one, ``16x0e+16x1o`` -> ``0e+1o``."""
    return "+".join(str(irrep) for _, irrep in Irreps.parse(irreps).terms)


def magnetic_config(legacy) -> dict:
    """The rewrite's model settings, read off a legacy magnetic model."""
    first = legacy.interactions[0]
    hidden = str(legacy.products[0].linear_ori.irreps_out)
    readout = legacy.readouts[-1]
    one_body = getattr(legacy, "one_body_cheb_basis_with_const", None)
    repulsion = getattr(legacy, "pair_repulsion_fn", None)
    return {
        "atomic_numbers": [int(z) for z in legacy.atomic_numbers.tolist()],
        "saturation": [float(v) for v in legacy.m_max.tolist()],
        "num_layers": len(legacy.interactions),
        "num_features": Irreps.parse(hidden).terms[0][0],
        "lmax": max(
            irrep.degree
            for _, irrep in Irreps.parse(str(first.edge_attrs_irreps)).terms
        ),
        "moment_lmax": max(
            irrep.degree
            for _, irrep in Irreps.parse(str(first.magmom_node_attrs_irreps)).terms
        ),
        "hidden_irreps": _per_channel(hidden),
        "num_radial": int(legacy.radial_embedding.out_dim),
        "num_moment_basis": int(legacy.mag_radial_embedding.num_basis),
        "one_body_basis": int(one_body.num_basis) if one_body is not None else 0,
        "cutoff": float(legacy.r_max),
        "cutoff_order": int(legacy.radial_embedding.cutoff_fn.p),
        "correlation": legacy.products[0]
        .symmetric_contractions.contractions[0]
        .correlation,
        "pair_repulsion": repulsion is not None,
        "readout_hidden": str(readout.hidden_irreps),
    }


def convert_magnetic(legacy, observables=None) -> MagneticModel:
    """A rewrite model computing what ``legacy`` computes."""
    config = magnetic_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        ["default"],
        AtomicNumberTable(config["atomic_numbers"]),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
    )
    model = MagneticModel(
        ReferenceBackend(),
        observables=observables or [DEFAULT_CATALOGUE.observable("energy")],
        energy_head=head,
        **config,
    )
    transfer_magnetic_weights(legacy, model)
    return model


def _copy_radial(source, destination) -> None:
    for index, weight in enumerate(destination.weights):
        weight.copy_(getattr(source, f"layer{index}").weight)


def _copy_skip(source, destination) -> None:
    destination.weight.copy_(
        torch.tensor(
            fully_connected_tp_weights_to_canonical(
                source,
                str(source.irreps_in1),
                str(source.irreps_out),
                Irreps.parse(str(source.irreps_in2)).dimension,
            ),
            dtype=destination.weight.dtype,
        )
    )


def transfer_magnetic_weights(legacy, model: MagneticModel) -> None:
    """Copy every trained number of a legacy magnetic model into ``model``."""
    backbone = model.backbone
    with torch.no_grad():
        copy_linear(legacy.node_embedding.linear, backbone.node_embedding)
        backbone.moments.saturation.copy_(legacy.m_max)
        for source, block in zip(
            legacy.interactions, backbone.interactions, strict=True
        ):
            copy_linear(source.linear_up, block.linear_up)
            _copy_radial(source.conv_tp_weights, block.edge_weights)
            _copy_radial(source.conv_tp_weights_magmom, block.moment_weights)
            _copy_radial(source.density_fn, block.density)
            copy_linear(source.magmom_linear, block.linear)
            _copy_skip(
                source.skip_tp if block.residual else source.magmom_skip_tp, block.skip
            )

        channel_in = _per_channel(
            str(legacy.products[0].symmetric_contractions.irreps_in)
        )
        for source, product in zip(legacy.products, backbone.products, strict=True):
            correlation = source.symmetric_contractions.contractions[0].correlation
            targets = [
                str(irrep)
                for _, irrep in Irreps.parse(str(source.linear_ori.irreps_out)).terms
            ]
            for position, contraction in enumerate(
                source.symmetric_contractions.contractions
            ):
                carried = contraction_weights_to_canonical(
                    contraction, channel_in, targets[position], correlation
                )
                for order, weights in enumerate(carried):
                    product.contraction.weights[position * correlation + order].copy_(
                        torch.tensor(weights, dtype=torch.float64)
                    )
            _copy_radial(source.conv_tp_weights, product.moment_weights)
            copy_linear(source.linear, product.linear)
            copy_linear(source.linear_ori, product.linear_ori)

        head = model.outputs.heads["energy"]
        for source, destination in zip(legacy.readouts, head.readouts, strict=True):
            if hasattr(source, "linear_1"):
                copy_linear(source.linear_1, destination.first)
                copy_linear(source.linear_2, destination.second)
            else:
                copy_linear(source.linear, destination)

        if model.one_body is not None:
            model.one_body.coefficients.copy_(legacy.onebody_magmombasis_coeffs)
            model.one_body.offset.copy_(legacy.one_body_magmom_const_correction)
