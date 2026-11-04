"""Polar field inputs keep their native path independently of the LES surface."""

import numpy as np
import pytest

from tests.golden.harness import FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.test_deployment_output_parity import polar_deployed
from tests.parity.test_polar_parity import CASES, legacy_polar, with_inputs

pytestmark = [pytest.mark.polar, pytest.mark.usefixtures("fp64", "isolated")]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("name", ["neutral molecule", "crystal"])
def test_polar_field_inputs_preserve_predictions_and_cache(
    device, padded, override, name
):
    from mace_torch.calculators import MACECalculator

    from mace.calculators import MACECalculator as LegacyCalculator

    legacy = legacy_polar(False)
    deployed = polar_deployed(legacy)
    field = [0.031, -0.012, 0.023] if override else None
    native = MACECalculator(
        models=deployed,
        device=device,
        external_field=field,
        pad_num_atoms=16 if padded else 0,
        pad_num_edges=512 if padded else 0,
    )
    assert all(tensor.device.type == "cpu" for tensor in deployed.engine.parameters())
    assert all(tensor.device.type == "cpu" for tensor in deployed.engine.buffers())
    frozen = LegacyCalculator(
        models=[legacy.to(device)],
        device=device,
        default_dtype="float64",
        model_type="PolarMACE",
        external_field=field,
    )
    atoms, _, inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    atoms.calc = native
    tolerance = FP64_ACCELERATED_BACKEND if device == "cuda" else FP64_CPU_REFERENCE
    first = atoms.get_potential_energy()
    assert native.check_state(atoms) == []
    atoms.info["external_field"] = np.array([-0.025, 0.018, -0.036])
    assert native.check_state(atoms)
    second = atoms.get_potential_energy()
    if override:
        np.testing.assert_allclose(
            first, second, atol=tolerance.atol, rtol=tolerance.rtol
        )
    else:
        assert abs(first - second) > tolerance.atol
    reference_atoms = atoms.copy()
    if override:
        # The frozen constructor sends [3] to a model requiring [graphs, 3].
        # Its supported atoms.info route supplies the same physical field.
        with pytest.raises(RuntimeError, match="same number of dimensions"):
            frozen.calculate(atoms)
        reference_atoms.info["external_field"] = np.asarray(field)
        frozen = LegacyCalculator(
            models=[legacy],
            device=device,
            default_dtype="float64",
            model_type="PolarMACE",
        )
    frozen.calculate(reference_atoms)
    for key in ("energy", "forces", "dipole", "charges", "fukui_functions"):
        np.testing.assert_allclose(
            native.results[key],
            frozen.results[key],
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            err_msg=key,
        )
    graph, _ = native._graph(atoms, padded=False)
    actual_field = graph["external_field"].detach().cpu().numpy()[0]
    np.testing.assert_array_equal(
        actual_field, field if override else atoms.info["external_field"]
    )
