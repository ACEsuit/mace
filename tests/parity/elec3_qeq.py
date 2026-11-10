"""QEq's pinned external oracle and tiny canonical fixture (test-only)."""

from __future__ import annotations

from pathlib import Path

import torch
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import widest
from mace_torch.backends import resolve_backend
from mace_torch.models import EnergyReadout, ScaleShiftSpec
from mace_torch.models.electrostatics import QEqModel
from mace_torch.physics import DerivativeEngine

from tests.parity.anchors import load_anchor_weights
from tests.parity.elec3 import ENERGY
from tests.parity.elec3 import PROFILES as PROFILES
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
    head = EnergyReadout(
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
        settings=QEqModel.make_solver_spec(
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
