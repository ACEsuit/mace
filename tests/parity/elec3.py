"""ELEC-3 fixtures and a test-only bridge to the pinned external oracle.

No runtime converter is shipped: the bridge only supplies frozen test weights.
The external packages load in a child process with their required MACE version.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import widest
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends import resolve_backend
from mace_torch.graph import build_batch
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.electrostatics import LocalChargeModel
from mace_torch.models.registry import registered_models
from mace_torch.neighbors import resolve_neighbor_backend
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import load_checkpoint, save_checkpoint
from scipy.spatial.transform import Rotation

from tests.golden.targets.elec3 import PINS

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "molecular": "realspace",
    "molecular_rotated": "realspace",
    "full_periodic": "pbc",
    "z_slab": "slab",
    "partial": "mixed_periodic",
    "per_structure": "auto",
}
ENERGY = DEFAULT_CATALOGUE.observable("energy")


def configuration(kind, degree, **options):
    return ModelConfig(
        model=kind,
        local_sources={
            "kind": kind,
            "multipole_max_l": degree,
            **({"formal_charges": {1: 1.0, 8: -2.0}} if kind == "lsc_mace" else {}),
            **options,
        },
        num_channels=2,
        num_interactions=2,
        max_L=1,
        max_ell=1,
        correlation=2,
        radial_mlp=(8,),
        radial_basis={"kind": "bessel", "num_basis": 4},
        scaling="none",
        readout={"mlp_irreps": "4x0e"},
        interaction_first="plain",
    )


def build_model(
    kind, degree, profile, dtype=torch.float64, backend="reference", **options
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
    config = configuration(kind, degree, **options)
    return registered_models()[kind](
        resolve_backend(backend),
        config,
        [1, 8],
        [ENERGY],
        head,
        settings=LocalChargeModel.solver_descriptor(
            config,
            ElectrostaticsConfig(
                enabled=True,
                periodicity_profile="molecular"
                if profile == "molecular_rotated"
                else profile,
                slab_normal=2 if profile == "z_slab" else None,
            ),
            precision=widest(precision, head.accumulate),
        ),
        precision=precision,
    )


def configurations(profile):
    positions = np.array(
        [[0.5, 0.4, 0.3], [1.45, 0.4, 0.3], [0.26, 1.33, 0.3], [3.1, 0.8, 0.5]]
    )
    numbers = np.array([8, 1, 1, 1])
    fields = {"external_field": np.array([0.01, -0.02, 0.005])}
    molecule = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
        properties=fields,
    )
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    rotated = Configuration(
        atomic_numbers=numbers,
        positions=positions @ rotation.T,
        cell=np.zeros((3, 3)),
        pbc=(False, False, False),
        properties={"external_field": fields["external_field"] @ rotation.T},
    )
    crystal = Configuration(
        atomic_numbers=numbers,
        positions=positions,
        cell=np.array([[5.2, 0.2, 0], [0, 5.4, 0.1], [0.2, 0, 5.6]]),
        pbc=(True, True, True),
        properties=fields,
    )
    slab = Configuration(
        atomic_numbers=numbers,
        positions=positions + np.array([0, 0, 4.0]),
        cell=np.diag([5.2, 5.4, 12.0]),
        pbc=(True, True, False),
        properties=fields,
    )
    return {
        "molecular": [molecule],
        "molecular_rotated": [rotated],
        "full_periodic": [crystal],
        "z_slab": [slab],
        "partial": [molecule, crystal, slab],
        "per_structure": [molecule, crystal, slab],
    }[profile]


def graph_for(profile, dtype=torch.float64):
    graph = build_batch(
        configurations(profile),
        atomic_number_table=AtomicNumberTable([1, 8]),
        cutoff=5.0,
        neighbor_backend=resolve_neighbor_backend(),
        dtype=dtype,
    )
    graph["charges"] = graph["node_attrs"] @ torch.tensor([1.0, -2.0], dtype=dtype)
    return graph


def outputs(model, graph):
    result = DerivativeEngine(model, ENERGY)(
        graph, compute=("forces", "stress", "virials"), training=True
    )
    return {
        "energy": result.total_energy,
        "node_energy": result.node_energies,
        "forces": result.forces,
        "stress": result.stress,
        "virials": result.virials,
        "dipole": result.dipole,
        "density_coefficients": result.extras["density_coefficients"],
    }


def anchor_path(kind, degree):
    return ROOT / "tests/golden/models" / f"elec3_{kind}_l{degree}.safetensors"


def reference_path(kind):
    return ROOT / "tests/golden/references" / f"elec3_{kind}.json"


def load_model(kind, degree, profile, dtype=torch.float64, backend="reference"):
    model = build_model(kind, degree, profile, dtype, backend)
    load_checkpoint(anchor_path(kind, degree), lambda _: model)
    return model


def reference_model(kind, degree, profile, **options):
    from e3nn import o3
    from mace_scf.electrostatics.localsources import LocalCharges, LocalSplitCharges

    from mace.modules import interaction_classes

    torch.manual_seed(123)
    cls = LocalCharges if kind == "lc_mace" else LocalSplitCharges
    return cls(
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
        atomic_multipoles_max_l=degree,
        pbc_handling=PROFILES[profile],
        **{
            **(
                {
                    "formal_charges_from_data": False,
                    "atomic_formal_charges": np.array([1.0, -2.0]),
                }
                if kind == "lsc_mace"
                else {}
            ),
            **options,
        },
    )


def carry_weights(source, destination):
    from tests.parity.fm00_convert import (
        copy_linear,
        linear_weights_to_canonical,
        transfer_weights,
    )

    transfer_weights(source, destination, 2)
    with torch.no_grad():
        for old, new in zip(
            source.lr_source_maps, destination.source_maps, strict=True
        ):
            if destination.kind == "lc_mace":
                copy_linear(old.linear, new)
                continue
            if new.density_map is not None:
                # Canonical output blocks are ordered by degree; the monopole
                # has two weights and is identically discarded by the source.
                linear = old.multipole_block.linear
                weights = linear_weights_to_canonical(
                    linear, str(linear.irreps_in), str(linear.irreps_out)
                )
                new.density_map.weight.copy_(torch.from_numpy(weights[2:]))
            transfer = new.transfer
            copy_linear(old.linear1, transfer.sender)
            copy_linear(old.linear2, transfer.receiver)
            offsets = []
            start = 0
            for instruction in old.conv_tp.instructions:
                size = int(np.prod(instruction.path_shape))
                if old.conv_tp.irreps_out[instruction.i_out].ir.l == 0:
                    offsets.extend(range(start, start + size))
                start += size
            state = {
                f"layer_{i}": getattr(old.conv_tp_weights, f"layer{i}").weight.detach()
                for i in range(4)
            }
            state["layer_3"] = state["layer_3"][:, offsets]
            transfer.radial.load_canonical(state)
            weights = linear_weights_to_canonical(
                old.linear, str(old.linear.irreps_in), str(old.linear.irreps_out)
            )
            transfer.readout.weight.copy_(torch.from_numpy(weights))


def capture():
    """Capture new external references once; called only by the golden target."""
    from tests.golden.harness import FP64_CPU_REFERENCE

    torch.set_default_dtype(torch.float64)
    for kind in ("lc_mace", "lsc_mace"):
        reference = {
            "pins": PINS,
            "tolerance_row": FP64_CPU_REFERENCE.name,
            "cases": {},
        }
        for degree in (0, 1):
            for profile in PROFILES:
                old = reference_model(kind, degree, profile)
                new = build_model(kind, degree, profile)
                carry_weights(old, new)
                graph = graph_for(profile)
                old_graph = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in graph.items()
                }
                expected = old(
                    old_graph,
                    training=True,
                    compute_force=True,
                    compute_stress=True,
                    compute_virials=True,
                )
                if profile == "per_structure":
                    # Upstream 'auto' picks one regime for the entire batch.
                    # V1's agreed per_structure regime evaluates each physical
                    # system independently; compare that function explicitly.
                    individual = []
                    for physical in ("molecular", "full_periodic", "z_slab"):
                        separate = reference_model(kind, degree, physical)
                        individual.append(
                            separate(
                                graph_for(physical),
                                training=True,
                                compute_force=True,
                                compute_stress=True,
                                compute_virials=True,
                            )
                        )
                    expected = {
                        name: torch.cat([item[name] for item in individual], dim=0)
                        for name in (
                            "energy",
                            "node_energy",
                            "forces",
                            "stress",
                            "virials",
                            "dipole",
                            "density_coefficients",
                        )
                    }
                actual = outputs(new, graph)
                # The established V1 contract masks aperiodic stress. Virials
                # remain physical and are compared for every structure.
                expected["stress"] = expected["stress"].clone()
                expected["stress"][~graph["pbc"].any(-1)] = 0
                captured = {}
                for name, value in actual.items():
                    torch.testing.assert_close(
                        value,
                        expected[name],
                        atol=FP64_CPU_REFERENCE.atol,
                        rtol=FP64_CPU_REFERENCE.rtol,
                        check_dtype=False,
                        msg=f"{kind} l{degree} {profile} {name}",
                    )
                    captured[name] = expected[name].detach().tolist()
                reference["cases"][f"l{degree}/{profile}"] = captured
                if profile == "molecular":
                    save_checkpoint(
                        anchor_path(kind, degree),
                        new,
                        {"model": new.config.model_dump(mode="json"), "pins": PINS},
                    )
                print(kind, degree, profile, "PASS", flush=True)
        reference_path(kind).write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    capture()
