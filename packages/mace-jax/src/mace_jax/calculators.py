"""ASE adapter for compiled single-head energy models."""

from __future__ import annotations

import jax
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms
from mace_jax.inference import JITPipeline


class MACECalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        model,
        *,
        dtype=None,
        device="cpu",
        backend="reference",
        padding=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = jax.devices("gpu" if device == "cuda" else device)[0]
        with jax.default_device(self.device):
            self.apply, params = load_model(model, dtype=dtype, backend=backend)
            self.pipeline = JITPipeline(self.apply, params, padding)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise ValueError("ASE inference requires an Atoms object")
        with jax.default_device(self.device):
            graph = graph_from_atoms(
                self.atoms,
                elements=self.apply.elements,
                cutoff=self.apply.config.model.r_max,
                dtype=self.apply.dtype,
                head=next(iter(self.apply.config.data.heads)),
            )
            output = self.pipeline(graph)
        self.results = {
            "energy": float(output.total_energy[0]),
            "forces": np.asarray(output.forces),
        }
        if self.atoms.pbc.any():
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                np.asarray(output.stress[0])
            )
