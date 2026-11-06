"""The public ASE contract. Head and basis checks belong to the shared loader."""

import numpy as np
import pytest
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress
from jax_conformance import TINY, ModelSurface
from mace_jax.calculators import MACECalculator
from mace_jax.deploy.neutral_io import load_model

from tests.golden import harness


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_periodic_and_molecular_results_and_md_reuse(dtype):
    row = harness.FP32 if dtype == "float32" else harness.FP64_CPU_REFERENCE
    calc = MACECalculator(TINY, dtype=dtype)
    eager = ModelSurface(*load_model(TINY, dtype=dtype))
    fixtures = harness.load_fixtures(names=["triclinic_bulk", "dimer_short"])
    for name in ("triclinic_bulk", "dimer_short"):
        atoms = fixtures[name]
        atoms.calc = calc
        for offset in (0.0, 0.01):
            atoms.positions[0, 1] += offset
            ref = eager.golden_outputs(atoms)
            np.testing.assert_allclose(
                atoms.get_potential_energy(),
                ref["energy"],
                atol=row.atol,
                rtol=row.rtol,
            )
            np.testing.assert_allclose(
                atoms.get_forces(), ref["forces"], atol=row.atol, rtol=row.rtol
            )
            if atoms.pbc.any():
                np.testing.assert_allclose(
                    atoms.get_stress(),
                    full_3x3_to_voigt_6_stress(ref["stress"]),
                    atol=row.atol,
                    rtol=row.rtol,
                )
                assert set(calc.results) == {"energy", "forces", "stress"}
            else:
                with pytest.raises(PropertyNotImplementedError):
                    atoms.get_stress()
                assert set(calc.results) == {"energy", "forces"}
    assert calc.pipeline.output_traces == 1
