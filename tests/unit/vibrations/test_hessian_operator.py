import numpy as np
import pytest
import torch
from ase.io import read
from scipy.sparse.linalg import eigsh

from mace.calculators import MACECalculator
from mace.calculators.vibrations import hessian_operator
from tests.golden.anchors import load_anchor
from tests.golden.harness import FIXTURES_DIR


@pytest.mark.parametrize(
    "name, lowest_curvature",
    [("dimer_short", -125.7085057214), ("triclinic_bulk", -4.5001065218)],
)
def test_products_match_force_differences_and_lowest_mode(name, lowest_curvature):
    atoms = read(FIXTURES_DIR / f"{name}.xyz")
    atoms.calc = MACECalculator(
        models=load_anchor("tiny_scaleshift"), device="cpu", default_dtype="float64"
    )
    operator = hessian_operator(atoms.calc, atoms)
    dense = atoms.calc.get_hessian(atoms).reshape(operator.shape)
    direction = np.linspace(-0.4, 0.7, 3 * len(atoms))
    step = 1e-5
    offsets = []
    for sign in (-1, 1):
        displaced = atoms.copy()
        displaced.positions += sign * step * direction.reshape(-1, 3)
        offsets.append(atoms.calc.get_forces(displaced))
    finite_difference = (offsets[0] - offsets[1]).ravel() / (2 * step)
    np.testing.assert_allclose(operator @ direction, finite_difference, atol=1e-7)
    np.testing.assert_allclose(operator.T @ direction, dense @ direction, atol=1e-10)
    lowest = eigsh(operator, k=1, which="SA", v0=direction, return_eigenvectors=False)
    np.testing.assert_allclose(lowest[0], lowest_curvature, rtol=0, atol=1e-8)


def test_committee_and_isolated_atom():
    calculator = MACECalculator(
        models=[load_anchor("tiny_mace"), load_anchor("tiny_scaleshift")],
        device="cpu",
        default_dtype="float64",
    )
    atoms = read(FIXTURES_DIR / "water_cluster.xyz")
    operator = hessian_operator(calculator, atoms)
    dense = np.mean(calculator.get_hessian(atoms), axis=0).reshape(operator.shape)
    atoms.positions[0, 0] += 0.1
    vector = np.arange(3 * len(atoms)) * (1 - 0.5j)
    np.testing.assert_allclose(operator @ vector, dense @ vector, atol=1e-9)

    atom = read(FIXTURES_DIR / "isolated_atom.xyz")
    np.testing.assert_array_equal(hessian_operator(calculator, atom) @ np.ones(3), 0)


def test_single_precision_product_in_inference_mode():
    calculator = MACECalculator(
        models=load_anchor("tiny_scaleshift", dtype=torch.float32),
        device="cpu",
        default_dtype="float32",
    )
    atoms = read(FIXTURES_DIR / "dimer_short.xyz")
    vector = np.linspace(-0.4, 0.7, 6, dtype=np.float32)
    with torch.inference_mode():
        product = hessian_operator(calculator, atoms) @ vector
    assert product.dtype == np.float32
    np.testing.assert_allclose(product[0], 41.4838068881, rtol=2e-5)
