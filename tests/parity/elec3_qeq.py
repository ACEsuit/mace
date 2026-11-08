"""QEq's pinned external oracle and tiny canonical fixture (test-only)."""

from __future__ import annotations

import json
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
from mace_torch.models.electrostatics import QEqModel
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import save_checkpoint

from tests.golden.targets.elec3 import PINS
from tests.parity.anchors import load_anchor_weights
from tests.parity.elec3 import ENERGY, PROFILES
from tests.parity.elec3 import configuration as local_configuration
from tests.parity.elec3 import graph_for as local_graph

ROOT = Path(__file__).resolve().parents[2]


def configuration(**options):
    document = local_configuration("lc_mace", 0).model_dump(mode="json")
    document.update(model="qeq", local_sources=None, qeq=options)
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
    return QEqModel(
        resolve_backend(backend),
        config,
        [1, 8],
        [ENERGY],
        head,
        settings=QEqModel.solver_descriptor(
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
        ),
        precision=precision,
        solver=solver,
        trains_derivatives=trains_derivatives,
    )


def graph_for(profile="molecular", dtype=torch.float64):
    graph = local_graph(profile, dtype)
    count = graph["ptr"].shape[0] - 1
    # Nonzero total charges, distinct in a mixed batch.
    graph["total_charge"] = torch.tensor([-0.6, 0.4, 1.1][:count], dtype=dtype)
    graph["input_chi"] = torch.linspace(
        -0.15, 0.20, graph["positions"].shape[0], dtype=dtype
    )
    graph["input_hardness"] = torch.linspace(
        1.8, 2.4, graph["positions"].shape[0], dtype=dtype
    )
    return graph


def outputs(model, graph):
    graph = dict(graph)
    if model.settings.electronegativity_input is not None:
        for name in ("input_chi", "input_hardness"):
            graph[name] = graph[name].clone().requires_grad_()
    output = DerivativeEngine(model, ENERGY)(
        graph, compute=("forces", "stress", "virials"), training=True
    )
    result = {
        "energy": output.total_energy,
        "forces": output.forces,
        "stress": output.stress,
        "virials": output.virials,
        "dipole": output.dipole,
        **{
            name: output.extras[name]
            for name in (
                "density_coefficients",
                "electronegativity",
                "hardness",
                "qeq_energy",
            )
        },
    }

    if model.settings.electronegativity_input is not None:
        gradients = torch.autograd.grad(
            output.forces.square().sum() + output.stress.square().sum(),
            (graph["input_chi"], graph["input_hardness"]),
            retain_graph=True,
        )
        result.update(
            zip(
                ("force_loss_chi_gradient", "force_loss_hardness_gradient"),
                gradients,
                strict=True,
            )
        )
    return result


def reference_model(profile="molecular", **options):
    from e3nn import o3
    from mace_scf.electrostatics.qeq import MACEQEq

    from mace.modules import interaction_classes

    torch.manual_seed(123)
    return MACEQEq(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
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
        train_hardness=options.get("train_hardness", False),
        read_enegs=options.get("electronegativity_input") is not None,
        read_hardness=options.get("hardness_input") is not None,
        default_hardness=options.get("default_hardness", 2.0),
        atomic_multipoles_smearing_width=options.get("multipole_width", 1.0),
        include_electrostatic_self_interaction=options.get(
            "energy_self_interaction", True
        ),
        qeq_charges="nonPBC"
        if options.get("charge_method") == "nonperiodic"
        else "autograd",
    )


def carry_weights(old, new):
    from tests.parity.fm00_convert import copy_linear, transfer_weights

    transfer_weights(old, new, 2)
    with torch.no_grad():
        for name, source in (
            ("electronegativity", old.enegs_readouts),
            ("hardness", old.hardness_readouts),
        ):
            if name not in new.source_maps:
                continue
            for before, after in zip(
                source, new.source_maps[name].readouts, strict=True
            ):
                if hasattr(before, "linear_1"):
                    copy_linear(before.linear_1, after.first)
                    copy_linear(before.linear_2, after.second)
                else:
                    copy_linear(before.linear, after)


def external_graph(graph):
    result = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }
    # QEq's oracle expects the old flat batching of these graph quantities.
    result["cell"] = result["cell"].reshape(-1, 3)
    result["pbc"] = result["pbc"].reshape(-1)
    result["enegs"] = result["input_chi"]
    result["hardness"] = result["input_hardness"]
    return result


VARIANTS = {
    "fixed": {},
    "learned_inputs": {
        "train_hardness": True,
        "electronegativity_input": "input_chi",
        "hardness_input": "input_hardness",
    },
    "nonperiodic": {"charge_method": "nonperiodic"},
    "no_self": {
        "energy_self_interaction": False,
        "default_hardness": 12.0,
        "multipole_width": 1.3,
    },
}


def anchor_path(variant):
    family = "learned" if VARIANTS[variant].get("train_hardness", False) else "fixed"
    return ROOT / "tests/golden/models" / f"elec3_qeq_{family}.safetensors"


def reference_path():
    return ROOT / "tests/golden/references/elec3_qeq.json"


def load_model(variant, profile, dtype=torch.float64, backend="reference"):
    model = build_model(profile, dtype, backend, **VARIANTS[variant])
    load_anchor_weights(anchor_path(variant), model)
    return model


def oracle_outputs(model, graph):
    """Differentiate the unchanged oracle energy under an external strain.

    The upstream QEq forward returns no stress/virial and never applies strain.
    This wrapper evaluates its energy on strained inputs, without modifying its
    code; forces and stress derive from the same energy used for charge solving.
    """
    data = external_graph(graph)
    if model.read_enegs:
        data["enegs"] = data["enegs"].requires_grad_()
        data["hardness"] = data["hardness"].requires_grad_()
    count = graph["ptr"].shape[0] - 1
    strain = torch.zeros(
        (count, 3, 3), dtype=graph["positions"].dtype, requires_grad=True
    )
    symmetric = (strain + strain.transpose(-1, -2)) / 2
    batch = graph["batch"]
    data["positions"] = data["positions"] + torch.einsum(
        "ni,nij->nj", data["positions"], symmetric[batch]
    )
    cell = graph["cell"].reshape(-1, 3, 3)
    data["cell"] = (cell + cell @ symmetric).reshape(-1, 3)
    edge_batch = batch[graph["edge_index"][0]]
    data["shifts"] = data["shifts"] + torch.einsum(
        "ni,nij->nj", data["shifts"], symmetric[edge_batch]
    )
    result = model(data, training=True, compute_force=True)
    gradient = torch.autograd.grad(result["energy"].sum(), strain, create_graph=True)[0]
    mask = graph["pbc"].view(-1, 3).any(-1)
    volume = torch.linalg.det(cell).abs()
    stress = (
        gradient / torch.where(mask, volume, torch.ones_like(volume))[:, None, None]
    )
    result["stress"] = torch.where(
        mask[:, None, None], stress, torch.zeros_like(stress)
    )
    result["virials"] = -gradient
    result["electronegativity"] = result.pop("enegs")
    if model.read_enegs:
        gradients = torch.autograd.grad(
            result["forces"].square().sum() + result["stress"].square().sum(),
            (data["enegs"], data["hardness"]),
            retain_graph=True,
        )
        result.update(
            zip(
                ("force_loss_chi_gradient", "force_loss_hardness_gradient"),
                gradients,
                strict=True,
            )
        )
    return result


def capture():
    from tests.golden.harness import FP64_CPU_REFERENCE

    torch.set_default_dtype(torch.float64)
    reference = {"pins": PINS, "tolerance_row": FP64_CPU_REFERENCE.name, "cases": {}}
    for variant, options in VARIANTS.items():
        for profile in PROFILES:
            old = reference_model(profile, **options)
            new = build_model(profile, **options)
            carry_weights(old, new)
            graph = graph_for(profile)
            if profile == "per_structure":
                # Preserve the agreed per-structure boundary convention. Use
                # exactly the same input values/slices in the individual oracle.
                individual = []
                start = 0
                for index, physical in enumerate(
                    ("molecular", "full_periodic", "z_slab")
                ):
                    one = graph_for(physical)
                    stop = start + one["positions"].shape[0]
                    for key in ("input_chi", "input_hardness"):
                        one[key] = graph[key][start:stop]
                    one["total_charge"] = graph["total_charge"][index : index + 1]
                    individual.append(
                        oracle_outputs(reference_model(physical, **options), one)
                    )
                    start = stop
                expected = {
                    name: torch.cat([item[name] for item in individual], dim=0)
                    for name in individual[0]
                }
            else:
                expected = oracle_outputs(old, graph)
            actual = outputs(new, graph)
            for name, value in actual.items():
                torch.testing.assert_close(
                    value,
                    expected[name],
                    atol=FP64_CPU_REFERENCE.atol,
                    rtol=FP64_CPU_REFERENCE.rtol,
                    check_dtype=False,
                    msg=f"{variant} {profile} {name}",
                )
            reference["cases"][f"{variant}/{profile}"] = {
                name: value.detach().tolist() for name, value in expected.items()
            }
            if profile == "molecular" and variant in ("fixed", "learned_inputs"):
                save_checkpoint(
                    anchor_path(variant),
                    new,
                    {"model": new.config.model_dump(mode="json"), "pins": PINS},
                )
            print(variant, profile, "PASS", flush=True)
    reference_path().write_text(json.dumps(reference, indent=2) + "\n")
