"""FixedPointSCF's pinned external oracle; no legacy runtime dependency."""

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
from mace_torch.models.electrostatics import FixedPointSCFModel
from mace_torch.physics import DerivativeEngine

from tests.parity.anchors import load_anchor_weights
from tests.parity.elec3 import ENERGY
from tests.parity.elec3 import PROFILES as PROFILES
from tests.parity.elec3 import configuration as local_configuration
from tests.parity.elec3_qeq import graph_for as charge_graph

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
        fixed_point_scf=options,
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
    head = EnergyReadout(
        ResolvedE0s({"default": {1: -0.2, 8: -0.7}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("none", (1.0,), (0.0,)),
        PrecisionConfig(model=precision),
        widen_node_energy=False,
    )
    config = configuration(**options)
    solver_spec = FixedPointSCFModel.make_solver_spec(
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
        settings=solver_spec,
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


def anchor_path(variant):
    return ROOT / "tests/golden/models" / f"elec3_fixed_point_{variant}.safetensors"


def reference_path():
    return ROOT / "tests/golden/references/elec3_fixed_point.json"


def load_model(variant, profile, dtype=torch.float64, backend="reference"):
    model = build_model(profile, dtype, backend, **VARIANTS[variant])
    load_anchor_weights(anchor_path(variant), model)
    return model
