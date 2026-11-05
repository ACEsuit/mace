"""Isolated differential evaluation and replay for PAR-2 (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from mace_core.config.precision import PrecisionConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.graph.collate import collate
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.neighbour_oracle import canonical_edges
from tests.parity.fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    model_arguments,
    transfer_weights,
)
from tests.parity.fuzz_cases import CUTOFF, MODEL_SEED, Case
from tests.parity.process_state import capture_state, restore_state, state_differences

ENERGY = DEFAULT_CATALOGUE.observable("energy")


@contextmanager
def isolated():
    before = capture_state()
    try:
        yield
    finally:
        restore_state(before)
        assert not state_differences(before, capture_state())


@lru_cache(maxsize=8)
def models(lmax, dtype, constants_dtype):
    with isolated():
        from e3nn import o3

        from mace import modules

        torch.set_default_dtype(torch.float64)
        torch.manual_seed(MODEL_SEED)
        legacy = modules.ScaleShiftMACE(
            r_max=CUTOFF,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=lmax,
            interaction_cls=modules.RealAgnosticResidualInteractionBlock,
            interaction_cls_first=modules.RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("2x0e + 2x1o"),
            MLP_irreps=o3.Irreps("2x0e"),
            atomic_energies=np.array([-1.5, -8.75]),
            avg_num_neighbors=4,
            atomic_numbers=[1, 8],
            correlation=2,
            gate=torch.nn.functional.silu,
            pair_repulsion=False,
            use_reduced_cg=False,
            atomic_inter_scale=1.25,
            atomic_inter_shift=-0.125,
        ).eval()
        settings = build_config(legacy)
        values, scale, shift = energy_constants_to_canonical(legacy)
        head = EnergyOutputHead(
            ResolvedE0s(values),
            ["default"],
            AtomicNumberTable([1, 8]),
            ScaleShiftSpec("std", scale, shift),
            PrecisionConfig(model=constants_dtype, accumulate="float64"),
        )
        native = MACEModel(
            ReferenceBackend(),
            observables=[ENERGY],
            energy_head=head,
            precision=dtype,
            **model_arguments(settings),
        ).eval()
        # Project the full legacy basis in fp64, before narrowing any weights.
        native.double()
        transfer_weights(legacy, native, settings["correlation"])
        native.to(getattr(torch, dtype))
        native.outputs.energy_head.to(getattr(torch, constants_dtype))
        native.outputs.energy_head.e0_table = (
            native.outputs.energy_head.e0_table.double()
        )
        legacy.to(getattr(torch, dtype))
        legacy.atomic_energies_fn.double()
        legacy.scale_shift.to(getattr(torch, constants_dtype))
        return legacy, native


def compare(case: Case, *, perturbation=None):
    """Compare E [eV], F [eV/A], stress [eV/A^3] and graph construction."""
    legacy, native = models(case.lmax, case.dtype, case.constants_dtype)
    dtype = getattr(torch, case.dtype)
    atoms = Atoms(case.numbers, positions=case.positions, cell=case.cell, pbc=case.pbc)
    # Restore after each stack, so neither inherits the other's global state.
    with isolated():
        from tests.golden.anchors import anchor_graph

        legacy_graph = anchor_graph(legacy, atoms, dtype)
        reference = legacy(
            legacy_graph,
            training=False,
            compute_force=True,
            compute_stress=True,
        )
    with isolated():
        # Deliberately disagree with the model dtype: graph and precision APIs
        # must not depend on the process default.
        torch.set_default_dtype(
            torch.float64 if dtype == torch.float32 else torch.float32
        )
        configuration = Configuration(
            atomic_numbers=np.asarray(case.numbers),
            positions=np.asarray(case.positions),
            cell=np.asarray(case.cell),
            pbc=tuple(case.pbc),
        )
        sample = graph_from_configuration(
            configuration,
            cutoff=CUTOFF,
            z_table=AtomicNumberTable([1, 8]),
        )
        graph = collate([sample], dtype=dtype)
        if perturbation == "drop_self_images":
            keep = graph["edge_index"][0] != graph["edge_index"][1]
            graph["edge_index"] = graph["edge_index"][:, keep]
            for key in ("shifts", "unit_shifts"):
                graph[key] = graph[key][keep]
        elif perturbation == "physical_open_cell":
            graph["cell"] = torch.tensor(case.cell, dtype=dtype)[None]
        elif perturbation is not None:
            raise ValueError(f"unknown perturbation: {perturbation}")
        actual = DerivativeEngine(native, ENERGY)(graph, compute=("forces", "stress"))

    tolerance = FP32 if case.dtype == "float32" else FP64_CPU_REFERENCE
    pairs = {
        "energy": (actual.total_energy, reference["energy"]),
        "forces": (actual.forces, reference["forces"]),
        "stress": (actual.stress, reference["stress"]),
        "node_energy": (actual.node_energies, reference["node_energy"]),
        "cell": (graph["cell"], legacy_graph["cell"].reshape(-1, 3, 3)),
    }
    errors = {}
    for name, (value, expected) in pairs.items():
        assert value is not None and expected is not None, name
        assert value.dtype == expected.dtype, (
            f"{name}: {value.dtype} != {expected.dtype}"
        )
        value, expected = value.detach().numpy(), expected.detach().numpy()
        assert np.isfinite(value).all() and np.isfinite(expected).all(), name
        np.testing.assert_allclose(
            value, expected, atol=tolerance.atol, rtol=tolerance.rtol, err_msg=name
        )
        errors[name] = float(np.max(np.abs(value - expected), initial=0))
    assert canonical_edges(
        graph["edge_index"].numpy(), graph["unit_shifts"].numpy()
    ) == canonical_edges(
        legacy_graph["edge_index"].numpy(), legacy_graph["unit_shifts"].numpy()
    ), "neighbor edges differ"
    return errors


def save_case(path: Path, case: Case, hypothesis_seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_seed": MODEL_SEED,
                "hypothesis_seed": hypothesis_seed,
                "case": case.record(),
            },
            indent=2,
        )
        + "\n"
    )


def check_case(case, root, hypothesis_seed, *, perturbation=None):
    """Persist the exact failing input before Hypothesis attempts shrinking."""
    try:
        return compare(case, perturbation=perturbation)
    except Exception as exc:
        digest = hashlib.sha256(
            json.dumps(case.record(), sort_keys=True).encode()
        ).hexdigest()[:16]
        path = root / f"failure-{case.input_class}-{digest}.json"
        save_case(path, case, hypothesis_seed)
        command = f"python -m tests.parity.fuzz_parity {shlex.quote(str(path))}"
        if perturbation is not None:
            command += f" --perturbation {shlex.quote(perturbation)}"
        raise AssertionError(
            f"PAR-2 {case.input_class}; Hypothesis seed={hypothesis_seed}; "
            f"geometry seed={case.geometry_seed}. Replay: {command}\n{exc}"
        ) from exc


def replay(path: Path, *, perturbation=None):
    record = json.loads(path.read_text())
    assert record["model_seed"] == MODEL_SEED, "model recipe seed changed"
    print(
        f"Hypothesis seed: {record['hypothesis_seed']}; model seed: {MODEL_SEED}",
        flush=True,
    )
    return compare(Case(**record["case"]), perturbation=perturbation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument(
        "--perturbation", choices=("drop_self_images", "physical_open_cell")
    )
    args = parser.parse_args()
    print(json.dumps(replay(args.case, perturbation=args.perturbation), indent=2))


if __name__ == "__main__":
    main()
