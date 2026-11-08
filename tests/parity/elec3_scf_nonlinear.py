"""Conversion and independent references for nonlinear FixedPointSCF."""

import math

import torch

from tests.parity.fm00_convert import (
    fully_connected_tp_weights_to_canonical,
)


def linear(source, target):
    # Probe its linear map; bias and adjacent simplified irreps are both
    # represented exactly, with V1's normalization applied only once.
    bias = source(torch.zeros(1, source.irreps_in.dim))
    matrix = source(torch.eye(source.irreps_in.dim)) - bias
    target.weight[target.source] = (
        matrix[target.column, target.row] / target.path_normalization[target.source]
    )
    if target.bias.numel():
        target.bias.copy_(bias[0, target.bias_row])


def element(source, target):
    target.load_canonical(
        {
            "weight": torch.tensor(
                fully_connected_tp_weights_to_canonical(
                    source, str(source.irreps_in1), str(source.irreps_out), 2
                )
            )
        }
    )


def sparse(source, target):
    target.weight.copy_(
        torch.cat(
            [
                (block * instruction.path_weight).reshape(-1)
                for instruction, block in zip(
                    source.instructions, source.weight_views(), strict=True
                )
            ]
        )
    )


def bilinear(source, target):
    from e3nn import o3
    from mace_core.clebsch_gordan.real_basis import wigner_3j_real

    for index, (i, j, k) in enumerate(target.paths):
        oldindex = next(
            n
            for n, inst in enumerate(source.instructions)
            if (inst.i_in1, inst.i_in2, inst.i_out) == (i, j, k)
        )
        inst = source.instructions[oldindex]
        weight = source.weight_view_for_instruction(oldindex)
        degrees = (
            source.irreps_in1[i].ir.l,
            source.irreps_in2[j].ir.l,
            source.irreps_out[k].ir.l,
        )
        sign = float(
            (
                o3.wigner_3j(*degrees) * torch.tensor(wigner_3j_real(*degrees).copy())
            ).sum()
        )
        if target.mode == "uvw":
            weight = weight.permute(2, 0, 1)
        target.weights[index].copy_(weight * inst.path_weight * sign)


def scalar(source, target):
    target.weight.copy_(
        torch.cat(
            [
                (
                    block.permute(2, 0, 1)
                    * instruction.path_weight
                    * math.sqrt(target.fan_in)
                ).reshape(-1)
                for instruction, block in zip(
                    source.instructions, source.weight_views(), strict=True
                )
            ]
        )
    )


def mlp(source, target):
    from tests.parity.test_backbone_parity import (
        _carry_fully_connected_net,
        _carry_radial_mlp,
    )

    if target.descriptor.kind == "normalized_silu":
        _carry_fully_connected_net(source, target)
    else:
        _carry_radial_mlp(source, target)


def activation(source, target):
    if target.mlp is not None:
        mlp(source.mlp, target.mlp)


def embedding(source, target):
    if target.kind == "tanh":
        linear(source.linear, target.potential)
    else:
        element(source.prod, target.potential)
    if target.bias is not None:
        linear(source.node_feats_linear, target.bias)


def mixer(source, target):
    if target.kind == "additive":
        linear(source.linear, target.embedding)
    elif target.kind == "product":
        linear(source.linear_up, target.embedding)
        bilinear(source.conv_tp, target.product)
        linear(source.linear_out, target.down)


def many_body(source, target):
    from tests.parity.test_backbone_parity import _carry_contraction

    old, new = source.interaction, target.interaction
    linear(old.linear_up, new.body.linear_up)
    mlp(old.conv_tp_weights, new.body.radial_mlp)
    linear(old.linear, new.body.linear)
    if new.skip is not None:
        element(old.skip_tp, new.skip)
    mlp(old.density_fn, new.body.density_mlp)
    if target.central is not None:
        linear(source.interaction.linear_2, target.central)
    mixer(source.central_atom_mixer, target.mixer)
    if hasattr(source, "central_atom_node_feats_mixer"):
        mixer(source.central_atom_node_feats_mixer, target.node_mixer)
    _carry_contraction(
        source.product.symmetric_contractions, target.product.contraction
    )
    linear(source.product.linear, target.product.linear)


@torch.no_grad()
def carry_update(source, target):
    embedding(source.potential_embedding, target.embedding)
    if target.products is not None:
        sparse(source.dot_products, target.products)
    activation(
        source.activation if target.many_body is not None else source.nonlinearity,
        target.activation,
    )
    sparse(source.tp_out, target.modulation)
    element(source.element_select_out, target.readout)
    if target.skip is not None:
        bilinear(source.tp_skip, target.skip)
    if target.many_body is not None:
        many_body(source, target.many_body)


@torch.no_grad()
def carry_energy(source, target):
    if target.kind == "mlp_field":
        linear(source.linear_up_q, target.charge)
        linear(source.linear_up_v, target.field)
        sparse(source.dot_products_q, target.charge_products)
        sparse(source.dot_products_v, target.field_products)
        mlp(source.mlp, target.mlp)
    elif target.kind == "quadratic_charges":
        linear(source.linear_qa, target.charge)
        linear(source.linear_qb, target.local)
        bilinear(source.qq_tp, target.products)
        scalar(source.tp_down, target.down)
    else:
        embedding(source.charges_embedding, target.charge)
        if target.field is not None:
            embedding(source.field_embedding, target.field)
        many_body(source, target.many_body)
        sparse(source.dot_products, target.products)
        activation(source.activation, target.activation)
        sparse(source.tp_out, target.modulation)
        element(source.element_select_out, target.down)


# Each architecture is exercised independently. The scaled skip fixtures keep
# their ordinary learned readout scale; the older affine fixture multiplied it
# by 200 solely to escape the reference's small-Fukui emergency clamp.
VARIANTS = {
    "shallow": {"update_nonlinearity": "shallow"},
    "mlp": {"update_nonlinearity": "mlp"},
    "skip_mlp": {"update_nonlinearity": "skip_mlp"},
    "tanh": {"potential_embedding": "tanh"},
    "scaled": {
        "update": "scaled",
        "update_nonlinearity": "mlp",
        "update_scales": (0.8, 1.2),
    },
    "scaled_dot": {"update": "scaled_dot", "update_nonlinearity": "skip_mlp"},
    "many_body": {"update": "many_body"},
    "augmented": {
        "update": "many_body",
        "update_correlation": 2,
        "update_interaction": "augmented",
        "node_mixer": "additive",
    },
    "product_mixers": {
        "update": "many_body",
        "central_mixer": "product",
        "node_mixer": "product",
    },
    "mlp_field": {"electron_energy": "mlp_field"},
    "quadratic_charges": {"electron_energy": "quadratic_charges"},
    "many_body_charges": {"electron_energy": "many_body_charges"},
    "many_body_charges_field": {"electron_energy": "many_body_charges_field"},
    "fixed_charge": {"update_nonlinearity": "mlp", "constant_charge": True},
    "dipoles": {
        "update_nonlinearity": "mlp",
        "feature_max_l": 1,
        "multipole_max_l": 1,
        "local_source": "linear",
        "electron_energy": "mlp_field",
    },
}
for _kind in ("quadratic_charges", "many_body_charges", "many_body_charges_field"):
    VARIANTS[_kind + "_vectors"] = {
        "electron_energy": _kind,
        "feature_max_l": 1,
        "multipole_max_l": 1,
        "local_source": "linear",
    }
VARIANTS = {
    name: {"constant_charge": False, **options} for name, options in VARIANTS.items()
}


def reference_model(variant, profile):
    from tests.parity.elec3_fixed_point import reference_model as build

    model = build(profile, **VARIANTS[variant])
    if (
        variant in ("scaled", "scaled_dot", "product_mixers", "dipoles", "augmented")
        or variant.endswith("_vectors")
        or variant.startswith("many_body_charges")
    ):
        with torch.no_grad():
            model.field_dependent_charges_map.element_select_out.weight.div_(200)
    return model


def anchor_path(variant):
    from tests.parity.elec3_fixed_point import ROOT

    return ROOT / "tests/golden/models" / f"elec3_scf_nonlinear_{variant}.safetensors"


def reference_path():
    from tests.parity.elec3_fixed_point import ROOT

    return ROOT / "tests/golden/references/elec3_scf_nonlinear.json"


def case_graph(variant, profile="molecular", dtype=torch.float64):
    from tests.parity.elec3_fixed_point import graph_for

    graph = graph_for(profile, dtype)
    if variant == "fixed_charge":
        graph["total_charge"] *= 0.1
    return graph


def load_model(variant, profile, dtype=torch.float64, backend="reference"):
    from tests.parity.anchors import load_anchor_weights
    from tests.parity.elec3_fixed_point import build_model

    model = build_model(profile, dtype, backend, **VARIANTS[variant])
    load_anchor_weights(anchor_path(variant), model)
    return model


def capture():
    import json

    from mace_torch.serialization import save_checkpoint

    from tests.golden.harness import FP64_CPU_REFERENCE as row
    from tests.parity.elec3_fixed_point import (
        PINS,
        PROFILES,
        build_model,
        carry_weights,
        oracle_outputs,
        outputs,
    )

    torch.set_default_dtype(torch.float64)
    reference = {"pins": PINS, "tolerance_row": row.name, "cases": {}}
    for variant, options in VARIANTS.items():
        for profile in PROFILES:
            print("CASE", variant, profile, flush=True)
            old = reference_model(variant, profile)
            new = build_model(profile, **options)
            carry_weights(old, new)
            graph = case_graph(variant, profile)
            if profile == "per_structure":
                individual = []
                for index, physical in enumerate(
                    ("molecular", "full_periodic", "z_slab")
                ):
                    one = case_graph(variant, physical)
                    for key in ("total_charge", "fermi_level"):
                        one[key] = graph[key][index : index + 1]
                    individual.append(
                        oracle_outputs(
                            reference_model(variant, physical),
                            one,
                            options["constant_charge"],
                        )
                    )
                expected = {
                    key: sum(item[key] for item in individual)
                    if key == "force_loss_response_gradient"
                    else torch.cat([item[key] for item in individual])
                    for key in individual[0]
                }
            else:
                expected = oracle_outputs(old, graph, options["constant_charge"])
            actual = outputs(new, graph)
            for key in actual:
                print(key, (actual[key] - expected[key]).abs().max().item(), flush=True)
                torch.testing.assert_close(
                    actual[key],
                    expected[key],
                    atol=row.atol,
                    rtol=row.rtol,
                    check_dtype=False,
                    msg=f"{variant}/{profile}/{key}",
                )
            reference["cases"][f"{variant}/{profile}"] = {
                key: value.detach().tolist() for key, value in expected.items()
            }
            if profile == "molecular":
                save_checkpoint(
                    anchor_path(variant),
                    new,
                    {"model": new.config.model_dump(mode="json"), "pins": PINS},
                )
            print("PASS", variant, profile, flush=True)
    reference_path().write_text(json.dumps(reference, indent=2) + "\n")
    capture_blocks()


def block_outputs(model, legacy=False):
    """Nonzero scalar/vector probes independent of SCF convergence dynamics."""
    torch.manual_seed(42)
    if legacy:
        source = model.field_dependent_charges_map
        field_dim = source.potential_irreps.dim
        density_dim = source.charges_irreps.dim
        dtype = source.element_select_out.weight.dtype
        device = source.element_select_out.weight.device
    else:
        source = model.updates[0]
        field_dim = (model.settings.feature_max_l + 1) ** 2 * len(
            model.settings.feature_widths
        )
        density_dim = (model.settings.multipole_max_l + 1) ** 2
        parameter = next(source.parameters())
        dtype = parameter.dtype
        device = parameter.device

    # Draw on CPU in float64 so the GPU receives identical probes.
    def draw(shape):
        return torch.randn(shape, dtype=torch.float64).to(device=device, dtype=dtype)

    nodes = draw((3, 8)).requires_grad_()
    attrs = torch.eye(2, device=device, dtype=dtype)[
        torch.tensor([0, 1, 1], device=device)
    ]
    field = (draw((3, field_dim)) * 0.3).requires_grad_()
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    edge_attributes = draw((4, 9))
    radial = draw((4, 4))
    local = draw((3, density_dim)) * 0.3
    density = (local + 0.1).requires_grad_()
    if legacy:
        delta = source(
            attrs, nodes, edge_attributes, radial, edge_index, field, local, density
        )
        energy = model.local_electron_energy(
            attrs, nodes, edge_attributes, radial, edge_index, field, local, density
        )
    else:
        delta = (
            source(nodes, attrs, field)
            if model.settings.affine_response
            else source(nodes, attrs, field, edge_attributes, radial, edge_index)
        )
        energy = (
            nodes.new_zeros(3)
            if model.electron_energy is None
            else model.electron_energy(
                nodes, field, local, density, attrs, edge_attributes, radial, edge_index
            )
        )
    variables = (nodes, field, density)
    gradient = torch.autograd.grad(
        delta.square().sum() + energy.square().sum(),
        variables,
        allow_unused=True,
        create_graph=True,
        retain_graph=True,
    )
    gradient = tuple(
        torch.zeros_like(value) if grad is None else grad
        for grad, value in zip(gradient, variables, strict=True)
    )
    second = torch.autograd.grad(
        sum(grad.sum() for grad in gradient),
        variables,
        allow_unused=True,
        retain_graph=True,
    )
    second = tuple(
        torch.zeros_like(value) if grad is None else grad
        for grad, value in zip(second, variables, strict=True)
    )
    return {
        "update": delta,
        "electron_energy": energy,
        "gradient": torch.cat([value.flatten() for value in gradient]),
        "second": torch.cat([value.flatten() for value in second]),
    }


def capture_blocks():
    import json

    from tests.golden.harness import FP64_CPU_REFERENCE as row
    from tests.parity.elec3_fixed_point import build_model, carry_weights

    torch.set_default_dtype(torch.float64)
    reference = json.loads(reference_path().read_text())
    reference["blocks"] = {}
    for variant, options in VARIANTS.items():
        old = reference_model(variant, "molecular")
        new = build_model(**options)
        carry_weights(old, new)
        expected = block_outputs(old, legacy=True)
        actual = block_outputs(new)
        for name in expected:
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=row.atol,
                rtol=row.rtol,
                msg=f"{variant}/block/{name}",
            )
        reference["blocks"][variant] = {
            name: value.detach().tolist() for name, value in expected.items()
        }
        print("BLOCK PASS", variant, flush=True)
    reference_path().write_text(json.dumps(reference, indent=2) + "\n")
