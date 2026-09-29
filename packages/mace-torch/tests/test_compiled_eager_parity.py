"""Compiled and padded against eager and unpadded, on the real atoms.

Energy, forces and stress of each structure, from a calculator that compiles
its model and pads every batch to a budget, against one that does neither.
Three structure sizes under each of two budgets: the one the first structure
sets, and a fixed one well above it. ``aot_eager`` runs everything a compile
does short of generating code, so it stays in the fast suite; Inductor itself
is the slow case.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.kernels.precision import Precision
from mace_torch.calculators import MACECalculator, ase_calculator
from mace_torch.calculators.padding import PaddingPolicy
from mace_torch_compile_fixtures import cluster, deployed

#: The model's dtype is the axis here, so the process default is held at one.
pytestmark = fp64_only

SIZES = (3, 1, 2)
BUDGETS = {
    "auto": None,
    "fixed": PaddingPolicy(mode="fixed", nodes_budget=12, edges_budget=1024),
}
_TORCH = {"float64": torch.float64, "float32": torch.float32}


def results(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    return {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(),
        "stress": atoms.get_stress(),
    }


def compare(
    model: Precision, budget: str, backend: str, monkeypatch, device: str = "cpu"
) -> None:
    torch._dynamo.reset()

    def compiled(engine, mode):
        engine = copy.copy(engine)
        engine.compile_model(backend=backend)
        return engine

    monkeypatch.setattr(ase_calculator, "_compiled", compiled)
    deployment = deployed(model)
    deployment.engine.to(device)
    eager = MACECalculator(models=deployment, device=device)
    padded = MACECalculator(
        models=deployment,
        device=device,
        compile_mode="default",
        padding=BUDGETS[budget],
    )
    atol, rtol = TOLERANCES[_TORCH[model]]
    for molecules in SIZES:
        atoms = cluster(molecules, periodic=True, seed=molecules)
        expected, actual = results(eager, atoms), results(padded, atoms)
        for name, value in expected.items():
            np.testing.assert_allclose(
                actual[name],
                value,
                atol=atol,
                rtol=rtol,
                err_msg=f"{name} of {molecules} waters",
            )
    torch._dynamo.reset()


@pytest.mark.parametrize("budget", sorted(BUDGETS))
@pytest.mark.parametrize("model", ["float64", "float32"])
def test_compiled_padded_matches_eager_unpadded(model, budget, monkeypatch):
    compare(model, budget, "aot_eager", monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("budget", sorted(BUDGETS))
@pytest.mark.parametrize("model", ["float64", "float32"])
def test_inductor_matches_eager_unpadded(model, budget, monkeypatch):
    compare(model, budget, "inductor", monkeypatch)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("budget", sorted(BUDGETS))
@pytest.mark.parametrize("model", ["float64", "float32"])
def test_inductor_on_cuda_matches_eager_unpadded(model, budget, monkeypatch):
    compare(model, budget, "inductor", monkeypatch, device="cuda")
