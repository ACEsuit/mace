"""The TorchSim adapter reproduces the committed legacy energy anchors."""

import json

import numpy as np
import pytest
from ase.io import read
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.train import write_model

from tests.parity.test_fm_convert_legacy import (
    CATALOGUE,
    FIXTURES,
    REFERENCE,
    reference,
)
from tests.parity.test_fm_convert_legacy import (
    fixture_converted as fixture_converted,
)

pytestmark = pytest.mark.torchsim


def test_torchsim_bundle_matches_committed_legacy_goldens(converted, tmp_path):
    import torch
    from mace_torch.deploy.torchsim import MaceTorchSimModel
    from torch_sim.io import atoms_to_state

    name, neutral = converted
    imported = import_neutral(neutral, CATALOGUE)
    checkpoint = write_model(tmp_path / "v1", imported.engine, imported.metadata)
    bundle = export_lammps(checkpoint, halo_exchange=False)
    adapter = MaceTorchSimModel(bundle)
    goldens = json.loads(reference(name).read_text())["fixtures"]
    for fixture, record in goldens.items():
        atoms = read(FIXTURES / f"{fixture}.xyz")
        state = atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float64)
        result = adapter(state)
        for quantity in ("energy", "forces", "stress"):
            if quantity not in record["outputs"]:
                continue
            expected = np.asarray(record["outputs"][quantity]["value"])
            actual = result[quantity].numpy()
            if quantity != "forces":
                actual = actual[0]
            if quantity == "stress" and expected.shape == (6,):
                from ase.stress import full_3x3_to_voigt_6_stress

                actual = full_3x3_to_voigt_6_stress(actual)
            np.testing.assert_allclose(
                actual, expected, atol=REFERENCE.atol, rtol=REFERENCE.rtol
            )


@pytest.mark.polar
@pytest.mark.parametrize(
    "name", ["neutral molecule", "charged doublet in a field", "crystal", "slab"]
)
@pytest.mark.parametrize("systems", [1, 2])
def test_torchsim_polar_inputs_and_outputs_match_legacy(
    fp64, isolated, monkeypatch, name, systems
):
    import mace_torch.deploy.torchsim as integration
    import torch
    from mace_core.graph import FixedPadding
    from mace_torch.calculators import MACECalculator
    from torch_sim.io import atoms_to_state

    from tests.parity.test_deployment_output_parity import polar_deployed
    from tests.parity.test_polar_parity import (
        CASES,
        legacy_batch,
        legacy_polar,
        with_inputs,
    )

    legacy = legacy_polar(False)
    deployed = polar_deployed(legacy)
    # The normal loader is covered by both anchor/bundle tests above. Here the
    # existing parity converter supplies identical weights for every polar case.
    monkeypatch.setattr(integration, "load_deployed", lambda *args, **kwargs: deployed)
    model = integration.MaceTorchSimModel(
        "converted",
        padding=FixedPadding(nodes_budget=16, edges_budget=512, graphs_budget=4),
    )
    atoms, _, inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    frames = [atoms.copy() for _ in range(systems)]
    state = atoms_to_state(frames, device=torch.device("cpu"), dtype=torch.float64)
    state.system_extras.update(
        total_charge=torch.full((systems,), inputs[0]),
        total_spin=torch.full((systems,), inputs[1]),
        external_field=torch.tensor([inputs[2]] * systems),
    )
    actual = model(state)
    expected = legacy(legacy_batch(legacy, *frames), training=False, compute_force=True)
    for key in (
        "energy",
        "forces",
        "dipole",
        "charges",
        "spins",
        "density_coefficients",
        "spin_charge_density",
        "fukui_functions",
    ):
        np.testing.assert_allclose(
            actual[key].numpy(),
            expected[key].detach().numpy(),
            atol=REFERENCE.atol,
            rtol=REFERENCE.rtol,
        )
    # Legacy's frozen strain bug is already reconciled: stress is compared
    # with V1 ASE, whose derivative convention has independent finite differences.
    atoms.calc = MACECalculator(models=[deployed])
    expected_stress = (
        atoms.get_stress(voigt=False) if atoms.pbc.any() else np.zeros((3, 3))
    )
    for stress in actual["stress"]:
        np.testing.assert_allclose(
            stress.numpy(), expected_stress, atol=REFERENCE.atol, rtol=REFERENCE.rtol
        )
