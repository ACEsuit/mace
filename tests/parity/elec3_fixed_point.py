"""FixedPointSCF's pinned external oracle; no legacy runtime dependency."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import widest
from mace_torch.backends import resolve_backend
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import FixedPointSCFModel
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import load_checkpoint, save_checkpoint

from tests.parity.elec3 import ENERGY, PROFILES, configuration as local_configuration
from tests.parity.elec3_qeq import graph_for as charge_graph
from tests.golden.targets._electrostatics_oracle import PINS

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = (
    "energy",
    "forces",
    "stress",
    "virials",
    "dipole",
    "density_coefficients",
    "fermi_level",
    "electrostatic_features",
    "electrostatic_energy",
    "electron_energy",
    "field_independent_charge_density",
)
VARIANTS = {
    "charge": {},
    "fermi": {"constant_charge": False, "fermi_level_offset": 0.2},
    "dipoles": {
        "multipole_max_l": 1,
        "feature_max_l": 1,
        "local_source": "linear",
        "density_scales": (0.8, 1.1),
        "feature_widths": (0.9, 1.3),
        "feature_norms": (1.2, 0.8, 1.1, 0.9),
        "electron_energy": "quadratic_field",
        "energy_self_interaction": True,
    },
    "linear": {
        "potential_embedding": "linear",
        "electron_energy": "linear_charges",
        "feature_self_interaction": True,
        "constant_charge": False,
    },
}


def configuration(**options):
    document = local_configuration("lc_mace", 0).model_dump(mode="json")
    document.update(
        model="fixed_point_scf",
        local_sources=None,
        fixed_point_scf={"kind": "fixed_point_scf", **options},
    )
    if options.get("update") == "many_body" or options.get(
        "electron_energy", ""
    ).startswith("many_body"):
        document["max_ell"] = 2
    return ModelConfig.model_validate(document)


def build_model(
    profile="molecular",
    dtype=torch.float64,
    backend="reference",
    solver="reference",
    trains_derivatives=False,
    **options,
):
    precision = str(dtype).removeprefix("torch.")
    head = EnergyOutputHead(
        ResolvedE0s({"default": {1: -0.2, 8: -0.7}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("none", (1.0,), (0.0,)),
        PrecisionConfig(model=precision),
        widen_node_energy=False,
    )
    config = configuration(**options)
    descriptor = FixedPointSCFModel.solver_descriptor(
        config,
        ElectrostaticsConfig(
            enabled=True,
            solver=solver,
            periodicity_profile="molecular"
            if profile == "molecular_rotated"
            else profile,
            slab_normal=2 if profile == "z_slab" else None,
        ),
        widest(precision, head.accumulate),
    )
    return FixedPointSCFModel(
        resolve_backend(backend),
        config,
        [1, 8],
        [ENERGY],
        head,
        settings=descriptor,
        precision=precision,
        solver=solver,
        trains_derivatives=trains_derivatives,
    )


def graph_for(profile="molecular", dtype=torch.float64):
    graph = charge_graph(profile, dtype)
    graph["total_charge"] *= 0.1
    graph["fermi_level"] = torch.linspace(-0.1, 0.3, len(graph["ptr"]) - 1, dtype=dtype)
    return graph


def outputs(model, graph):
    output = DerivativeEngine(model, ENERGY)(
        dict(graph), compute=("forces", "stress", "virials"), training=True
    )
    result = {
        "energy": output.total_energy,
        "forces": output.forces,
        "stress": output.stress,
        "virials": output.virials,
        "dipole": output.dipole,
    }
    result.update({key: output.extras[key] for key in OUTPUTS[5:]})
    (weight,) = tuple(model.updates[0].readout.parameters())
    gradient = torch.autograd.grad(
        output.forces.square().sum() + output.stress.square().sum(),
        weight,
        retain_graph=True,
    )[0]
    result["force_loss_response_gradient"] = (gradient * weight).sum().reshape(1)
    return result


def reference_model(profile="molecular", **options):
    from e3nn import o3
    from mace_scf.electrostatics.fixed_point_core import FixedPointCore
    from mace_scf.electrostatics import field_blocks as blocks
    from mace.modules import interaction_classes

    torch.manual_seed(123)
    model = FixedPointCore(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2
        if options.get("update") == "many_body"
        or options.get("electron_energy", "").startswith("many_body")
        else 1,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("2x0e+2x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.array([[-0.2, -0.7]]),
        avg_num_neighbors=1.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_MLP=[8],
        pbc_handling=PROFILES[profile],
        atom_density_scaling=np.array(
            options.get("update_scales", options.get("density_scales", (1.0, 1.0)))
        ),
        atomic_multipoles_max_l=options.get("multipole_max_l", 0),
        atomic_multipoles_smearing_width=options.get("multipole_width", 1.0),
        field_feature_max_l=options.get("feature_max_l", 0),
        field_feature_widths=list(options.get("feature_widths", (1.0,))),
        field_feature_norms=options.get("feature_norms"),
        field_si=options.get("feature_self_interaction", False),
        include_electrostatic_self_interaction=options.get(
            "energy_self_interaction", False
        ),
        quadrupole_feature_corrections=options.get("quadrupole_corrections", False),
        fermi_level_offset=options.get("fermi_level_offset", 0.0),
        use_linear_local_charges=options.get("local_source") == "linear",
        add_local_electron_energy=options.get("electron_energy", "none") != "none",
        fixedpoint_update_config={
            "type": {
                "one_body": blocks.OneBodyVariableUpdate,
                "scaled": blocks.OneBodyScaledNonLinearUpdate,
                "scaled_dot": blocks.OneBodyScaledDPNonLinearUpdate,
                "many_body": blocks.ManyBodyUpdate,
            }[options.get("update", "one_body")],
            "potential_embedding_cls": {
                "linear": blocks.LinearPotentialEmbedding,
                "biased_linear": blocks.BiasedLinearPotentialEmbedding,
                "tanh": blocks.TanhPotentialEmbedding,
            }[options.get("potential_embedding", "biased_linear")],
            "nonlinearity_cls": {
                "none": blocks.NoNonLinearity,
                "shallow": blocks.ShallowNonLinearity,
                "mlp": blocks.MLPNonLinearity,
                "skip_mlp": blocks.SkipMLPLinearity,
            }[options.get("update_nonlinearity", "none")],
            **(
                {
                    "correlation": options.get("update_correlation", 1),
                    "use_sc": options.get("update_use_skip", True),
                    "radial_MLP": list(
                        options.get("update_radial_hidden", (64, 64, 64))
                    ),
                    "central_atom_mixer_cls": {
                        "none": blocks.NoMixer,
                        "additive": blocks.AdditiveFieldMixer,
                        "product": blocks.UncoupledProductMixer,
                    }[options.get("central_mixer", "additive")],
                    "central_atom_feats_mixer_cls": {
                        "none": blocks.NoMixer,
                        "additive": blocks.AdditiveFieldMixer,
                        "product": blocks.UncoupledProductMixer,
                    }[options.get("node_mixer", "none")],
                    "interaction_cls": blocks.AugRealAgnosticResidualInteractionBlock
                    if options.get("update_interaction") == "augmented"
                    else blocks.RealAgnosticDensityResidualInteractionBlock,
                }
                if options.get("update") == "many_body"
                else {}
            ),
        },
        field_readout_config={
            "type": {
                "none": blocks.NullFieldReadout,
                "quadratic_field": blocks.StrictQuadraticFieldEnergyReadout,
                "linear_charges": blocks.LinearChargesEnergyReadout,
                "mlp_field": blocks.OneBodyMLPFieldReadout,
                "quadratic_charges": blocks.QuadraticChargesEnergyReadout,
                "many_body_charges": blocks.ManyBodyChargesReadout,
                "many_body_charges_field": blocks.ManyBodyChargesFieldReadout,
            }[options.get("electron_energy", "none")]
        },
    )
    # Tiny random backbones have a vanishing fourth-order field response.
    # Scale this fixture's learned response before freezing it, so the legacy
    # Newton update is outside its 1e-6 emergency clamp.
    with torch.no_grad():
        model.field_dependent_charges_map.element_select_out.weight.mul_(200)
    return model


@torch.no_grad()
def carry_weights(old, new):
    from tests.parity.fm00_convert import (
        copy_linear,
        transfer_weights,
        fully_connected_tp_weights_to_canonical,
    )

    transfer_weights(old, new, 2)

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
        weights = []
        for instruction, block in zip(
            source.instructions, source.weight_views(), strict=True
        ):
            weights.append((block * instruction.path_weight).reshape(-1))
        with torch.no_grad():
            target.weight.copy_(torch.cat(weights))

    def bilinear(source, target):
        weights = []
        for instruction, block in zip(
            source.instructions, source.weight_views(), strict=True
        ):
            weights.append(
                (
                    block.permute(2, 0, 1)
                    * instruction.path_weight
                    * math.sqrt(target.fan_in)
                ).reshape(-1)
            )
        with torch.no_grad():
            target.weight.copy_(torch.cat(weights))

    for before, after in zip(old.lr_source_maps, new.source_maps, strict=True):
        if hasattr(before, "linear"):
            copy_linear(before.linear, after)
        else:
            for name, dest in (
                ("linear_1", "first"),
                ("linear_mid", "middle"),
                ("linear_2", "last"),
            ):
                copy_linear(getattr(before, name), getattr(after, dest))
    for before, after in zip(
        old.layer_feature_mixer.linears, new.layer_mixer, strict=True
    ):
        copy_linear(before, after)
    if new.settings.affine_response:
        source, target = old.field_dependent_charges_map, new.updates[0]
        element(source.potential_embedding.prod, target.potential)
        if target.bias is not None:
            copy_linear(source.potential_embedding.node_feats_linear, target.bias)
        sparse(source.dot_products, target.products)
        sparse(source.tp_out, target.modulation)
        element(source.element_select_out, target.readout)
    else:
        from tests.parity.elec3_scf_nonlinear import carry_update

        carry_update(old.field_dependent_charges_map, new.updates[0])
    if new.electron_energy is not None:
        source, target = old.local_electron_energy, new.electron_energy
        if target.kind == "quadratic_field":
            copy_linear(source.linear_qa, target.charge)
            copy_linear(source.linear_va, target.field)
            bilinear(source.qv_tp, target.products)
            bilinear(source.tp_down, target.down)
        elif target.kind == "linear_charges":
            bilinear(source.tp, target.down)
        else:
            from tests.parity.elec3_scf_nonlinear import carry_energy

            carry_energy(source, target)


def oracle_outputs(
    model, graph, constant_charge=True, *, num_scf_steps=1000, mixing_parameter=0.25
):
    from mace_scf.electrostatics.fixed_point_runner import FixedPointSCFRunner
    from mace_scf.electrostatics.fixed_point_state import FixedPointSCFOptions

    data = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }
    count = len(graph["ptr"]) - 1
    strain = torch.zeros(
        (count, 3, 3), dtype=graph["positions"].dtype, requires_grad=True
    )
    symmetric = (strain + strain.transpose(-1, -2)) / 2
    batch = graph["batch"]
    data["positions"] = data["positions"] + torch.einsum(
        "ni,nij->nj", data["positions"], symmetric[batch]
    )
    cell = graph["cell"].reshape(-1, 3, 3)
    deformed = cell + cell @ symmetric
    data["cell"] = deformed.reshape(-1, 3)
    data["rcell"] = (
        2 * math.pi * torch.linalg.inv(deformed).transpose(-1, -2)
    ).reshape(-1, 3)
    data["volume"] = torch.linalg.det(deformed).abs()
    data["pbc"] = graph["pbc"].reshape(-1)
    edge_batch = batch[graph["edge_index"][0]]
    data["shifts"] = data["shifts"] + torch.einsum(
        "ni,nij->nj", data["shifts"], symmetric[edge_batch]
    )
    runner = FixedPointSCFRunner(
        FixedPointSCFOptions(
            num_scf_steps=num_scf_steps,
            scf_tolerance=1e-15,
            mixing_parameter=mixing_parameter,
            constant_charge=constant_charge,
            use_autograd_forces=True,
        )
    )
    local = model.local_part(data, compute_force=True)
    state = runner.converge(
        model,
        data,
        local,
        runner.get_initial_density(local, data),
        runner.get_initial_fermi(model, local, data),
        compute_force=True,
    )
    print(
        "SCF",
        state.status,
        state.terminated_step,
        state.final_avg_abs_change.detach().tolist(),
        flush=True,
    )
    assert state.status == "converged", state.status
    result = model.build_observables(
        data,
        local,
        state.density,
        state.fermi_level,
        state.field_feats,
        training=True,
        compute_force=True,
    )
    gradient = torch.autograd.grad(result["energy"].sum(), strain, create_graph=True)[0]
    mask = graph["pbc"].view(-1, 3).any(-1)
    stress = (
        gradient
        / torch.where(mask, torch.linalg.det(cell).abs(), torch.ones(count))[
            :, None, None
        ]
    )
    result["stress"] = torch.where(
        mask[:, None, None], stress, torch.zeros_like(stress)
    )
    result["virials"] = -gradient
    weight = model.field_dependent_charges_map.element_select_out.weight
    gradient = torch.autograd.grad(
        result["forces"].square().sum() + result["stress"].square().sum(),
        weight,
        retain_graph=True,
    )[0]
    result["force_loss_response_gradient"] = (gradient * weight).sum().reshape(1)
    return {key: result[key] for key in (*OUTPUTS, "force_loss_response_gradient")}


def anchor_path(variant):
    return ROOT / "tests/golden/models" / f"elec3_fixed_point_{variant}.safetensors"


def reference_path():
    return ROOT / "tests/golden/references/elec3_fixed_point.json"


def load_model(variant, profile, dtype=torch.float64, backend="reference"):
    model = build_model(profile, dtype, backend, **VARIANTS[variant])
    load_checkpoint(anchor_path(variant), lambda _: model)
    return model


def capture():
    from tests.golden.harness import FP64_CPU_REFERENCE

    torch.set_default_dtype(torch.float64)
    reference = {"pins": PINS, "tolerance_row": FP64_CPU_REFERENCE.name, "cases": {}}
    for variant, options in VARIANTS.items():
        for profile in PROFILES:
            print("CASE", variant, profile, flush=True)
            old = reference_model(profile, **options)
            new = build_model(profile, **options)
            carry_weights(old, new)
            graph = graph_for(profile)
            if profile == "per_structure":
                individual = []
                for index, physical in enumerate(
                    ("molecular", "full_periodic", "z_slab")
                ):
                    one = graph_for(physical)
                    for key in ("total_charge", "fermi_level"):
                        one[key] = graph[key][index : index + 1]
                    individual.append(
                        oracle_outputs(
                            reference_model(physical, **options),
                            one,
                            options.get("constant_charge", True),
                        )
                    )
                expected = {
                    key: (
                        sum(item[key] for item in individual)
                        if key == "force_loss_response_gradient"
                        else torch.cat([item[key] for item in individual])
                    )
                    for key in individual[0]
                }
            else:
                expected = oracle_outputs(
                    old, graph, options.get("constant_charge", True)
                )
            actual = outputs(new, graph)
            for key in actual:
                print(key, (actual[key] - expected[key]).abs().max().item(), flush=True)
                torch.testing.assert_close(
                    actual[key],
                    expected[key],
                    atol=FP64_CPU_REFERENCE.atol,
                    rtol=FP64_CPU_REFERENCE.rtol,
                    check_dtype=False,
                    msg=f"{variant} {profile} {key}",
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
            print(variant, profile, "PASS", flush=True)
    reference_path().write_text(json.dumps(reference, indent=2) + "\n")
