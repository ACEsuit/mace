from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from mace.calculators import MACECalculator

pytest.importorskip("graph_longrange.energy")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "mace-polar-spin-2L.model"
REF_PATH = Path(__file__).with_name("polar_regression_reference.json")

if REF_PATH.exists():
    _REF = json.loads(REF_PATH.read_text())
    STRUCTURE_KEYS = sorted(_REF.get("structures", {}).keys())
    BENCH_ROOT = Path(_REF.get("bench_root", ""))
else:
    _REF = {"structures": {}}
    STRUCTURE_KEYS = []
    BENCH_ROOT = Path("")

ATOL_BY_DTYPE = {
    "float32": 1e-6,
    "float64": 1e-9,
}


@pytest.fixture(scope="module", params=["float32", "float64"])
def polar_calc(request):
    dtype = request.param
    calc = MACECalculator(
        model_paths=str(MODEL_PATH),
        model_type="PolarMACE",
        device="cpu",
        default_dtype=dtype,
    )
    return dtype, calc


@pytest.mark.skipif(not REF_PATH.exists(), reason="Regression reference JSON not available")
@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Polar 2L model file not available")
@pytest.mark.skipif(not BENCH_ROOT.exists(), reason="benchmarks-mp X23 structures not available")
@pytest.mark.parametrize("structure_relpath", STRUCTURE_KEYS)
def test_polar_2l_regression_hardcoded_values(polar_calc, structure_relpath):
    dtype, calc = polar_calc
    expected = _REF["structures"][structure_relpath][dtype]

    at = read(BENCH_ROOT / structure_relpath, index=0)
    at.info["charge"] = 0
    at.info["spin"] = 1
    at.calc = calc

    energy = float(at.get_potential_energy())
    forces = at.get_forces()
    stress = at.get_stress()

    atol = ATOL_BY_DTYPE[dtype]
    np.testing.assert_allclose(energy, expected["energy"], rtol=0.0, atol=atol)
    np.testing.assert_allclose(forces, np.array(expected["forces"]), rtol=0.0, atol=atol)
    np.testing.assert_allclose(stress, np.array(expected["stress"]), rtol=0.0, atol=atol)
