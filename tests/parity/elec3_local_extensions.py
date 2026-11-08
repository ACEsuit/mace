"""Initial external references for LSC oxidation and response extensions."""

import json

import torch

from tests.golden.targets._electrostatics_oracle import PINS
from tests.parity import elec3
from tests.parity.elec3_scf_nonlinear import linear, sparse
from tests.parity.fm00_convert import fully_connected_tp_weights_to_canonical

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


def reference_model(variant, degree, profile):
    options = VARIANTS[variant]
    return elec3.reference_model(
        "lsc_mace",
        degree,
        profile,
        static_bond_transfer_block=(
            "OxidationDependentSymmetricPredictionSourceBlock"
            if options.get("transfer")
            else "NoFieldSymmetricPredictionSourceBlock"
        ),
        oxidation_state_mixer={
            "none": "NoOxidationStateMixer",
            "sum": "SumOxidationStateMixer",
            "product": "ProductOxidationStateMixer",
        }[options.get("oxidation_mixer", "none")],
        oxidation_state_range=options.get("oxidation_range", (-4.0, 4.0)),
        compute_polarizability=options.get("polarizability", False),
        formal_charges_from_data=bool(options.get("formal_charge_input")),
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


def carry_weights(old, new):
    elec3.carry_weights(old, new)
    with torch.no_grad():
        if new.settings.oxidation_mixer != "none":
            source, target = old.oxidation_state_mixer, new.backbone.graph_embedding
            linear(source.ox_embedding.oxidation_state_linear, target.oxidation.linear)
            if target.product is not None:
                target.product.load_canonical(
                    {
                        "weight": torch.tensor(
                            fully_connected_tp_weights_to_canonical(
                                source.oxidation_state_tp,
                                str(source.oxidation_state_tp.irreps_in1),
                                str(source.oxidation_state_tp.irreps_out),
                                new.config.num_channels,
                            )
                        )
                    }
                )
        for source, target in zip(old.lr_source_maps, new.source_maps, strict=True):
            if target.oxidation is not None:
                linear(source.oxidation_state_linear, target.oxidation.linear)
                sparse(source.oxidation_state_tp, target.modulation)
        if new.polarizability_readouts is not None:
            for source, target in zip(
                old.polarizability_readouts, new.polarizability_readouts, strict=True
            ):
                linear(source.linear1, target.linear)
                linear(source.bypass_linear, target.skip)
                columns = [
                    i
                    for span, (_, ir) in zip(
                        source.tp.irreps_out.slices(), source.tp.irreps_out, strict=True
                    )
                    if ir.l in (0, 2)
                    for i in range(span.start, span.stop)
                ]
                matrix = source.linear2(torch.eye(source.linear2.irreps_in.dim))[
                    columns
                ]
                op = target.readout
                op.weight[op.source] = (
                    matrix[op.column, op.row] / op.path_normalization[op.source]
                )


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


def oracle(variant, degree, profile):
    if profile == "per_structure":
        parts = [
            oracle(variant, degree, p) for p in ("molecular", "full_periodic", "z_slab")
        ]
        return {
            name: (
                torch.stack([p[name] for p in parts]).sum(0)
                if name == "response_gradient"
                else torch.cat([p[name] for p in parts])
            )
            for name in parts[0]
        }
    old = reference_model(variant, degree, profile)
    graph = graph_for(variant, profile)
    result = old(
        graph,
        training=True,
        compute_force=True,
        compute_stress=True,
        compute_virials=True,
    )
    result["stress"] = result["stress"].clone()
    result["stress"][~graph["pbc"].any(-1)] = 0
    gradient = torch.autograd.grad(
        response_loss(result), old.node_embedding.linear.weight
    )[0]
    names = [
        "energy",
        "node_energy",
        "forces",
        "stress",
        "virials",
        "dipole",
        "density_coefficients",
    ]
    if VARIANTS[variant].get("polarizability"):
        names.append("polarizability")
    return {
        **{name: result[name] for name in names},
        "response_gradient": gradient.reshape(2, 2).T.reshape(-1),
    }


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


def capture():
    from tests.golden.harness import FP64_CPU_REFERENCE

    torch.set_default_dtype(torch.float64)
    reference = {"pins": PINS, "tolerance_row": FP64_CPU_REFERENCE.name, "cases": {}}
    for variant in VARIANTS:
        for degree in (0, 1):
            for profile in elec3.PROFILES:
                old, new = (
                    reference_model(variant, degree, profile),
                    build_model(variant, degree, profile),
                )
                carry_weights(old, new)
                graph = graph_for(variant, profile)
                actual = outputs(new, graph)
                expected = oracle(variant, degree, profile)
                captured = {}
                for name, value in actual.items():
                    torch.testing.assert_close(
                        value,
                        expected[name],
                        atol=FP64_CPU_REFERENCE.atol,
                        rtol=FP64_CPU_REFERENCE.rtol,
                        check_dtype=False,
                        msg=f"{variant} l{degree} {profile} {name}",
                    )
                    captured[name] = expected[name].detach().tolist()
                reference["cases"][f"{variant}/l{degree}/{profile}"] = captured
                if profile == "molecular":
                    elec3.save_checkpoint(
                        anchor_path(variant, degree),
                        new,
                        {"model": new.config.model_dump(mode="json"), "pins": PINS},
                    )
                print(variant, degree, profile, "PASS", flush=True)
    reference_path().write_text(json.dumps(reference, indent=2) + "\n")
