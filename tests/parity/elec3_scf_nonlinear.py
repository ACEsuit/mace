"""Conversion and independent references for nonlinear FixedPointSCF."""

import torch

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


def block_outputs(model):
    """Nonzero scalar/vector probes independent of SCF convergence dynamics."""
    torch.manual_seed(42)
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
