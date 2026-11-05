"""JAX-1's source-tree conformance entry point, using the frozen golden harness."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms
from mace_jax.models.registry import SUPPORTED_MODELS

from tests.golden import harness

TINY = Path(__file__).parent / "fixtures" / "tiny.json"


class ModelSurface:
    golden_surface = "model"

    def __init__(self, apply, params):
        self.apply, self.params = apply, params

    def golden_outputs(self, atoms):
        graph = graph_from_atoms(
            atoms,
            elements=self.apply.elements,
            cutoff=self.apply.config.model.r_max,
            dtype=self.apply.dtype,
            head=next(iter(self.apply.config.data.heads)),
        )
        output = self.apply(self.params, graph)
        return {
            "energy": np.asarray(output.total_energy)[0],
            "node_energy": np.asarray(output.node_energies),
            "forces": np.asarray(output.forces),
            "stress": np.asarray(output.stress)[0],
            "virials": np.asarray(output.virials)[0],
        }


def run_jax_model_conformance(
    model_id,
    *,
    dtypes=("float64", "float32"),
    devices=("cpu",),
    tol=None,
    artifact_path=None,
):
    """Committed golden parity and O(3)/translation/permutation checks, eager.

    A local converted artifact may be provided while FM-2 publication is
    deferred. It never substitutes freshly evaluated predictions for a golden.
    """
    anchors = {"tiny": "tiny_scaleshift", "tiny_plain": "tiny_mace"}
    reference_name = (
        anchors[model_id] + "_e3nn_cpu_fp64.json"
        if model_id in anchors
        else SUPPORTED_MODELS[model_id].golden_ref
    )
    reference = harness.load_reference(harness.REFERENCES_DIR / reference_name)
    if model_id == "mace-off23-medium":
        periodic = harness.load_reference(
            harness.REFERENCES_DIR / "off_medium_periodic_e3nn_cpu_fp64.json"
        )
        assert periodic["provenance"]["sha256"] == reference["provenance"]["sha256"]
        reference = {
            **reference,
            "fixtures": {**reference["fixtures"], **periodic["fixtures"]},
        }
    fixtures = harness.load_fixtures(names=tuple(reference["fixtures"]))
    results = []
    for device in devices:
        backend_name = "gpu" if device == "cuda" else device
        accelerator = jax.devices(backend_name)[0]
        for dtype in dtypes:
            row = (
                harness.tolerance(tol)
                if tol
                else (
                    harness.FP32 if dtype == "float32" else harness.FP64_CPU_REFERENCE
                )
            )
            with jax.default_device(accelerator):
                apply, params = load_model(
                    artifact_path or (TINY if model_id == "tiny" else model_id),
                    dtype=dtype,
                )
                if model_id in SUPPORTED_MODELS:
                    model = apply.config.model
                    blocks = {
                        model.interaction_first,
                        model.interaction,
                        model.radial_basis.kind,
                        "polynomial_cutoff",
                        "linear_readout",
                        "nonlinear_readout",
                    }
                    if model.distance_transform.kind != "none":
                        blocks.add(model.distance_transform.kind)
                    if model.pair_repulsion:
                        blocks.add("zbl")
                    assert blocks == SUPPORTED_MODELS[model_id].required_blocks
                surface = ModelSurface(apply, params)
                snapshot = harness.snapshot_outputs(
                    surface, fixtures, dtype=dtype, device=device, backend="reference"
                )
                channels = ["energy", "forces", "stress"]
                if (
                    "node_energy"
                    in next(iter(reference["fixtures"].values()))["outputs"]
                ):
                    channels.append("energies")
                harness.compare_to_reference(
                    snapshot, reference, row=row.name, channels=channels
                )
                errors = {
                    f"{fixture}/{key}": float(
                        np.max(
                            np.abs(
                                np.asarray(entry["outputs"][key]["value"])
                                - np.asarray(
                                    reference["fixtures"][fixture]["outputs"][key][
                                        "value"
                                    ]
                                )
                            )
                        )
                    )
                    for fixture, entry in snapshot["fixtures"].items()
                    for key in channels
                    if key in entry["outputs"]
                    and key in reference["fixtures"][fixture]["outputs"]
                }
                # The committed water geometry is asymmetric; an improper
                # rotation also checks the irrep parity convention.
                atoms = fixtures.get("water_cluster", next(iter(fixtures.values())))
                initial = surface.golden_outputs(atoms)
                rotation, _ = np.linalg.qr(
                    np.random.default_rng(51).normal(size=(3, 3))
                )
                rotation[:, 0] *= -np.linalg.det(rotation)
                moved = atoms.copy()
                moved.positions = atoms.positions @ rotation + np.asarray(
                    [1.3, -0.6, 0.8]
                )
                moved.set_cell(np.asarray(atoms.cell) @ rotation)
                permutation = np.random.default_rng(74).permutation(len(atoms))
                moved = moved[permutation]
                changed = surface.golden_outputs(moved)
                for got, expected in (
                    (changed["energy"], initial["energy"]),
                    (changed["forces"], initial["forces"][permutation] @ rotation),
                    (changed["stress"], rotation.T @ initial["stress"] @ rotation),
                ):
                    np.testing.assert_allclose(
                        got, expected, atol=row.atol, rtol=row.rtol
                    )
                results.append(
                    {
                        "model_id": model_id,
                        "dtype": dtype,
                        "device": device,
                        "tolerance_row": row.name,
                        "max_abs_difference": errors,
                        "equivariance": "passed",
                    }
                )
            jax.clear_caches()
    return results
