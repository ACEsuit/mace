"""The response models behind the one ASE calculator.

What a dipole or dielectric model gives through the calculator: the frozen
tree's result names and shapes, the same numbers as the model evaluated
directly, no energy, forces or stress, its inputs read from ``atoms``, padding
that changes nothing, and dielectric derivatives in the frozen tree's shapes
for one model and for a committee. What the calculator writes is decided by
what the model declares, so a capability it does not declare is refused by
the observable's name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.calculators import MACECalculator
from mace_torch.calculators.padding import PaddingPolicy
from mace_torch.deploy.loader import DeployedModel
from mace_torch.train.model_stage import build_model

NUMBERS = [1, 6, 8]
METHANOL = Atoms(
    "COHHHH",
    positions=[
        [0.0, 0.0, 0.0],
        [1.42, 0.0, 0.0],
        [1.75, 0.9, 0.1],
        [-0.36, 1.03, 0.05],
        [-0.36, -0.52, 0.89],
        [-0.36, -0.5, -0.9],
    ],
)
OBSERVABLES = {"dipole": ["dipole"], "dielectric": ["dipole", "polarizability"]}


def deployed(family: str, seed: int = 0) -> DeployedModel:
    config = ResolvedConfig.model_validate(
        {
            "model": {
                "model": family,
                "observables": OBSERVABLES[family],
                "r_max": 4.0,
                "num_channels": 4,
                "max_ell": 2,
                "hidden_irreps": "0e+1o+2e",
                "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"},
            },
            "runtime": {"seed": seed},
        }
    )
    engine, outputs = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(NUMBERS),
        heads=("default",),
        e0s=ResolvedE0s({"default": dict.fromkeys(NUMBERS, 0.0)}),
        statistics=DatasetStatistics(avg_num_neighbors=3.0),
    )
    return DeployedModel(
        engine=engine.eval(),
        config=config,
        metadata=ModelMetadata(
            config=ConfigRecord(),
            provenance=Provenance(code_version="0", git_commit=None),
        ),
        z_table=AtomicNumberTable(NUMBERS),
        heads=("default",),
        e0s={"default": dict.fromkeys(NUMBERS, 0.0)},
        outputs=outputs,
        path=Path(f"{family}-{seed}"),
    )


def evaluate(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    calculator.calculate(atoms)
    return dict(calculator.results)


def charged(atoms, charges=(-0.1, -0.6, 0.4, 0.1, 0.1, 0.1), total=0.0):
    atoms = atoms.copy()
    atoms.arrays["Qs"] = np.asarray(charges)
    atoms.info["charge"] = total
    return atoms


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize(
    ("family", "shapes"),
    [
        ("dipole", {"dipole": (3,)}),
        (
            "dielectric",
            {
                "dipole": (3,),
                "charges": (6,),
                "polarizability": (3, 3),
                "polarizability_sh": (6,),
            },
        ),
    ],
)
def test_a_response_model_writes_the_frozen_tree_s_keys(family, shapes):
    calculator = MACECalculator(models=deployed(family))
    results = evaluate(calculator, charged(METHANOL))
    assert set(results) == set(shapes) == set(calculator.implemented_properties)
    for key, shape in shapes.items():
        assert np.shape(results[key]) == shape, key


@fp64_only
@pytest.mark.parametrize("family", ["dipole", "dielectric"])
def test_the_results_are_the_model_s_own_on_the_same_graph(family):
    model = deployed(family)
    calculator = MACECalculator(models=model)
    atoms = charged(METHANOL)
    results = evaluate(calculator, atoms)
    graph, _ = calculator._graph(atoms, padded=False)
    direct = model.compute(graph, compute=())
    assert direct.dipole is not None
    np.testing.assert_array_equal(results["dipole"], direct.dipole[0].detach().numpy())
    if family == "dielectric":
        for key in ("charges", "polarizability"):
            value = direct.extras[key].detach().numpy()
            expected = value if key == "charges" else value[0]
            np.testing.assert_array_equal(results[key], expected)


@fp64_only
def test_a_response_model_has_no_energy_to_give():
    atoms = charged(METHANOL)
    atoms.calc = MACECalculator(models=deployed("dielectric"))
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energy()
    assert atoms.get_dipole_moment().shape == (3,)


def test_the_hessian_is_refused_by_the_observable_it_needs():
    calculator = MACECalculator(models=deployed("dipole"))
    with pytest.raises(NotImplementedError, match="'energy'"):
        calculator.get_hessian(charged(METHANOL))


# ---------------------------------------------------------------------------
# Inputs from atoms
# ---------------------------------------------------------------------------


@fp64_only
def test_the_fixed_charges_are_read_from_the_charges_key():
    calculator = MACECalculator(models=deployed("dipole"))
    plain = evaluate(calculator, charged(METHANOL, charges=np.zeros(6)))
    moved = evaluate(calculator, charged(METHANOL))
    assert np.abs(moved["dipole"] - plain["dipole"]).max() > 1e-3


@fp64_only
@pytest.mark.parametrize("total", [0.0, -1.0])
def test_the_predicted_charges_add_up_to_the_charge_in_info(total):
    results = evaluate(
        MACECalculator(models=deployed("dielectric")), charged(METHANOL, total=total)
    )
    assert abs(results["charges"].sum() - total) < 1e-12


@fp64_only
@pytest.mark.parametrize("family", ["dipole", "dielectric"])
def test_padding_changes_no_result(family):
    model = deployed(family)
    atoms = charged(METHANOL, total=-1.0)
    plain = evaluate(MACECalculator(models=model), atoms)
    padded = evaluate(
        MACECalculator(
            models=model, padding=PaddingPolicy.requested(16, 256, environ={})
        ),
        atoms,
    )
    for key, value in plain.items():
        np.testing.assert_allclose(padded[key], value, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Dielectric derivatives
# ---------------------------------------------------------------------------


@fp64_only
def test_one_dielectric_model_gives_both_derivatives_as_arrays():
    calculator = MACECalculator(models=deployed("dielectric"))
    dmu_dr, dalpha_dr = calculator.get_dielectric_derivatives(charged(METHANOL))
    assert dmu_dr.shape == (3, 6, 3)
    assert dalpha_dr.shape == (9, 6, 3)


@fp64_only
def test_one_dipole_model_gives_the_dipole_derivative_alone():
    calculator = MACECalculator(models=deployed("dipole"))
    dmu_dr = calculator.get_dielectric_derivatives(charged(METHANOL))
    assert isinstance(dmu_dr, np.ndarray)
    assert dmu_dr.shape == (3, 6, 3)


@fp64_only
def test_a_committee_gives_one_entry_per_model():
    members = [deployed("dielectric", seed) for seed in (1, 2)]
    calculator = MACECalculator(models=members)
    dmu_dr, dalpha_dr = calculator.get_dielectric_derivatives(charged(METHANOL))
    assert isinstance(dmu_dr, list) and len(dmu_dr) == 2
    assert isinstance(dalpha_dr, list) and len(dalpha_dr) == 2
    assert np.abs(dmu_dr[0] - dmu_dr[1]).max() > 1e-6
    for member, expected in zip(members, dmu_dr, strict=True):
        single = MACECalculator(models=member).get_dielectric_derivatives(
            charged(METHANOL)
        )[0]
        np.testing.assert_array_equal(single, expected)
    results = evaluate(calculator, charged(METHANOL))
    assert results["dipole_comm"].shape == (2, 3)
    assert results["dipole_var"].shape == (3,)


@fp64_only
def test_the_dipole_derivative_is_the_derivative_of_the_reported_dipole():
    calculator = MACECalculator(models=deployed("dielectric"))
    atoms = charged(METHANOL)
    dmu_dr, _ = calculator.get_dielectric_derivatives(atoms)
    step = 1e-5
    for atom, axis in [(0, 0), (2, 1), (5, 2)]:
        shifted = []
        for sign in (1, -1):
            moved = atoms.copy()
            moved.positions[atom, axis] += sign * step
            shifted.append(evaluate(calculator, moved)["dipole"])
        numeric = (shifted[0] - shifted[1]) / (2 * step)
        np.testing.assert_allclose(dmu_dr[:, atom, axis], numeric, atol=1e-8)


def test_a_committee_of_different_families_is_refused():
    with pytest.raises(ValueError, match="different observables"):
        MACECalculator(models=[deployed("dipole"), deployed("dielectric")])


def test_the_dtype_of_the_results_is_float64():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        results = evaluate(
            MACECalculator(models=deployed("dielectric")), charged(METHANOL)
        )
    finally:
        torch.set_default_dtype(previous)
    assert all(np.asarray(value).dtype == np.float64 for value in results.values())
