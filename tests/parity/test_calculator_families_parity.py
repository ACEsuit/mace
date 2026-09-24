"""The response models through the one v1 calculator, against the frozen one.

The frozen calculator is driven with ``model_type="DipolePolarizabilityMACE"``
or ``"DipoleMACE"``, the v1 one with the converted model and no model type at
all: what it writes is read off what the model declares. The results and the
dielectric derivatives are compared, for one model and for a committee, whose
shapes the frozen tree splits.

**The frozen dipole model has no dielectric derivatives through its
calculator.** ``get_dielectric_derivatives`` with ``"DipoleMACE"`` calls the
model with an argument its forward does not take, and raises ``TypeError``. The
v1 calculator returns the derivative, and it is compared with the frozen tree's
own gradient function over the frozen model's own dipole.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import AtomicNumberTable
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.loader import DeployedModel
from mace_torch.physics import DerivativeEngine

from tests.parity.test_dipole_parity import (
    ANCHOR,
    CASES,
    convert,
    legacy_anchor,
    legacy_batch,
    legacy_dielectric,
)

OBSERVABLES = {"dipole": ["dipole"], "dielectric": ["dipole", "polarizability"]}
LEGACY_TYPE = {"dipole": "DipoleMACE", "dielectric": "DipolePolarizabilityMACE"}


def deployed(legacy, family: str) -> DeployedModel:
    model, config = convert(legacy, family)
    resolved = ResolvedConfig.model_validate(
        {
            "model": {
                "model": family,
                "observables": OBSERVABLES[family],
                "r_max": config["cutoff"],
            }
        }
    )
    outputs = resolve_requested(OBSERVABLES[family], DEFAULT_CATALOGUE)
    engine = DerivativeEngine(model, None, responses=outputs.observables)
    numbers = config["atomic_numbers"]
    return DeployedModel(
        engine=engine.eval(),
        config=resolved,
        metadata=ModelMetadata(
            config=ConfigRecord(),
            provenance=Provenance(code_version="0", git_commit=None),
        ),
        z_table=AtomicNumberTable(numbers),
        heads=("default",),
        e0s={"default": dict.fromkeys(numbers, 0.0)},
        outputs=outputs,
        path=Path(family),
    )


def legacy_calculator(paths, family):
    from mace.calculators import MACECalculator as LegacyCalculator

    return LegacyCalculator(
        model_paths=[str(path) for path in paths],
        model_type=LEGACY_TYPE[family],
        device="cpu",
        default_dtype="float64",
    )


def with_inputs(name):
    atoms, charges, total_charge = CASES[name]
    atoms = atoms.copy()
    atoms.arrays["Qs"] = np.asarray(charges, dtype=float)
    atoms.info["charge"] = total_charge
    return atoms


def dielectric_committee(tmp_path, seeds):
    legacies, paths = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        legacy = legacy_dielectric("8x0e + 8x1o + 8x2e")
        # A distinct model per seed: the builder reseeds, so perturb here.
        with torch.no_grad():
            generator = torch.Generator().manual_seed(seed)
            for parameter in legacy.parameters():
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
        path = tmp_path / f"dielectric_{seed}.model"
        torch.save(legacy, path)
        legacies.append(legacy)
        paths.append(path)
    return legacies, paths


def results_of(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    calculator.calculate(atoms)
    return dict(calculator.results)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("seeds", [(3,), (3, 4)], ids=["single", "committee"])
def test_the_dielectric_surface_is_the_frozen_calculator_s(fp64, tmp_path, name, seeds):
    legacies, paths = dielectric_committee(tmp_path, seeds)
    frozen = legacy_calculator(paths, "dielectric")
    v1 = MACECalculator(models=[deployed(legacy, "dielectric") for legacy in legacies])
    atoms = with_inputs(name)

    expected = results_of(frozen, atoms)
    got = results_of(v1, atoms)
    assert set(got) <= set(expected), sorted(set(got) - set(expected))
    assert set(got) == set(v1.implemented_properties)
    for key in got:
        np.testing.assert_allclose(got[key], expected[key], atol=1e-12, err_msg=key)

    frozen_derivatives = frozen.get_dielectric_derivatives(atoms)
    v1_derivatives = v1.get_dielectric_derivatives(atoms)
    assert type(v1_derivatives) is type(frozen_derivatives)
    for mine, theirs in zip(v1_derivatives, frozen_derivatives, strict=True):
        assert type(mine) is type(theirs)
        np.testing.assert_allclose(np.asarray(mine), np.asarray(theirs), atol=1e-12)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_dipole_surface_is_the_frozen_calculator_s(fp64, name):
    from mace.modules.utils import compute_dielectric_gradients

    legacy = legacy_anchor()
    frozen = legacy_calculator([ANCHOR], "dipole")
    v1 = MACECalculator(models=deployed(legacy, "dipole"))
    atoms = with_inputs(name)

    expected = results_of(frozen, atoms)
    got = results_of(v1, atoms)
    assert set(got) == {"dipole"} == set(v1.implemented_properties)
    np.testing.assert_allclose(got["dipole"], expected["dipole"], atol=1e-12)

    with pytest.raises(TypeError, match="compute_dielectric_derivatives"):
        frozen.get_dielectric_derivatives(atoms)
    _, charges, total_charge = CASES[name]
    batch = legacy_batch(legacy, atoms, charges, total_charge)
    reference = compute_dielectric_gradients(
        legacy(batch, training=False)["dipole"], batch["positions"]
    )
    np.testing.assert_allclose(
        v1.get_dielectric_derivatives(atoms), reference.detach().numpy(), atol=1e-12
    )
