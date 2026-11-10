"""Initial external references for LSC oxidation and response extensions."""

import torch

from tests.parity import elec3

VARIANTS = {
    "sum": {"oxidation_mixer": "sum"},
    "product": {"oxidation_mixer": "product"},
    "oxidation": {"transfer": "oxidation_dependent"},
    "oxidation_sum": {"transfer": "oxidation_dependent", "oxidation_mixer": "sum"},
    "oxidation_product": {
        "transfer": "oxidation_dependent",
        "oxidation_mixer": "product",
    },
    "polarizability": {"polarizability": True},
    "combined_atomic": {
        "transfer": "oxidation_dependent",
        "oxidation_mixer": "product",
        "oxidation_range": (-3.0, 5.0),
        "polarizability": True,
        "formal_charge_input": "oxidation_state",
        "formal_charges": None,
    },
}


def build_model(variant, degree, profile, dtype=torch.float64, backend="reference"):
    return elec3.build_model(
        "lsc_mace", degree, profile, dtype, backend, **VARIANTS[variant]
    )


def graph_for(variant, profile, dtype=torch.float64):
    graph = elec3.graph_for(profile, dtype)
    if variant == "combined_atomic":
        graph["charges"] = (
            graph["charges"] * 0.7
            + torch.arange(graph["charges"].numel(), dtype=dtype).remainder(4) * 0.05
        )
    graph["oxidation_state"] = graph["charges"].clone()
    return graph


def outputs(model, graph):
    result = elec3.outputs(model, graph)
    if model.settings.polarizability:
        result["polarizability"] = model(graph).extras["polarizability"]
    op = model.backbone.node_embedding
    parameter = op.operation.weight if hasattr(op, "operation") else op.weight
    gradient = torch.autograd.grad(response_loss(result), parameter, retain_graph=True)[
        0
    ].reshape(-1)
    if hasattr(op, "order"):
        canonical = torch.empty_like(gradient)
        canonical[op.order] = gradient / op.scale
        gradient = canonical
    result["response_gradient"] = gradient
    return result


def response_loss(result):
    loss = result["forces"].square().sum() + result["stress"].square().sum()
    if result.get("polarizability") is not None:
        loss = loss + result["polarizability"].square().sum()
    return loss


def anchor_path(variant, degree):
    return (
        elec3.ROOT
        / "tests/golden/models"
        / f"elec3_lsc_{variant}_l{degree}.safetensors"
    )


def reference_path():
    return elec3.ROOT / "tests/golden/references/elec3_local_extensions.json"


def load_model(variant, degree, profile, dtype=torch.float64, backend="reference"):
    model = build_model(variant, degree, profile, dtype, backend)
    elec3.load_anchor_weights(anchor_path(variant, degree), model)
    return model
