"""CLI-trained V1 checkpoints through ASE, without importing the legacy engine.

These are deployment workflows, not comparisons of stochastic training paths.
The fixed-anchor comparisons with legacy live in tests/parity/test_calculator*.
The joint energy/dipole case also checks the physical charge contribution and
the vector features of the final layer after training and checkpoint reload.
"""

import copy
import json
import shutil

import numpy as np
import pytest
import yaml
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress
from ase.io import read, write

from tests.golden.harness import FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.workflows.test_data_sources_v1 import launch
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, tiny_task, training_records

pytestmark = needs_the_v1_engine


def close(actual, expected, row=FP64_CPU_REFERENCE):
    np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)


@pytest.fixture(scope="module")
def trained_calculators(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-calculators")
    base = yaml.safe_load(tiny_task(root).read_text())
    base["training"]["max_num_epochs"] = 2
    for seed in (1, 2, 3):
        document = copy.deepcopy(base)
        document["runtime"]["seed"] = seed
        launch(root, document, f"energy-{seed}")
    frames = read(root / "train.xyz", ":")
    for atoms in frames:
        charges = np.array([-0.8 if z == 8 else 0.4 for z in atoms.numbers])
        atoms.arrays["Qs"] = charges
        atoms.info["dipole"] = charges @ atoms.positions
    write(root / "responses.xyz", frames)
    for family in ("dipole", "energy-dipole"):
        document = copy.deepcopy(base)
        document["data"]["heads"]["default"]["train_file"] = str(root / "responses.xyz")
        document["model"]["readout"] = {"mlp_irreps": "4x0e+4x1o"}
        if family == "dipole":
            document["model"].update(model="dipole", observables=["dipole"])
            document["loss"] = {"weights": {"dipole": 1.0}}
        else:
            document["model"]["observables"].append("dipole")
            document["loss"]["weights"]["dipole"] = 1.0
        launch(root, document, family)
    for name in ("energy-1", "energy-2", "energy-3", "dipole", "energy-dipole"):
        rows = training_records(root / name)
        assert {row["epoch"] for row in rows if row["mode"] == "opt"} == {0, 1}
        assert all(np.isfinite(row["loss"]) for row in rows)
        assert (root / name / "tiny.safetensors").is_file()
    return root, frames


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    return request.param


def calculator(trained, device, family="energy-1", **kwargs):
    from mace_torch.calculators import MACECalculator

    return MACECalculator(model_paths=trained[0] / family / "tiny.safetensors", device=device, **kwargs)


def molecule(trained, periodic=False):
    atoms = trained[1][4].copy()
    atoms.info.pop("dipole")  # A training label must not shadow ASE's prediction on write.
    atoms.set_cell([4.3, 4.7, 5.1])
    atoms.set_pbc(periodic)
    return atoms


def roundtrip(atoms, path):
    """Standard ASE results survive extxyz; extra V1 properties stay declared."""
    from ase.calculators.calculator import all_properties

    expected = copy.deepcopy(atoms.calc.results)
    assert expected.keys() <= set(atoms.calc.implemented_properties)
    write(path, atoms)
    restored = read(path)
    for key in expected.keys() & set(all_properties):
        close(restored.calc.results[key], expected[key])


@pytest.mark.parametrize("quantity", ["forces", "stress"])
def test_trained_energy_derivatives_and_xyz(trained_calculators, device, quantity, tmp_path):
    atoms = molecule(trained_calculators, periodic=quantity == "stress")
    atoms.calc = calculator(trained_calculators, device)
    if quantity == "forces":
        expected = atoms.get_forces()
        numerical = calculate_numerical_forces(atoms, eps=1e-4)
    else:
        expected = atoms.get_stress()
        numerical = calculate_numerical_stress(atoms, eps=1e-4)
    close(numerical, expected)
    atoms.get_potential_energy()
    roundtrip(atoms, tmp_path / "derivatives.xyz")


def test_trained_atomic_energies(trained_calculators, device):
    calc = calculator(trained_calculators, device)
    e0 = calc.models[0].e0s["default"]
    for source in trained_calculators[1]:
        atoms = source.copy()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        reference = np.array([e0[z] for z in atoms.numbers])
        close(calc.results["node_energy"] + reference, calc.results["energies"])
        close(calc.results["energies"].sum(), energy)


@pytest.mark.parametrize("source", ["paths", "models"])
def test_trained_committee_and_model_instances(trained_calculators, device, source, tmp_path):
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed

    paths = [trained_calculators[0] / f"energy-{i}/tiny.safetensors" for i in (1, 2, 3)]
    models = [load_deployed(path, device=device) for path in paths] if source == "models" else paths
    key = "models" if source == "models" else "model_paths"
    singles = []
    atoms = molecule(trained_calculators, periodic=True)
    for model in models:
        atoms.calc = MACECalculator(**{key: model}, device=device)
        atoms.get_potential_energy()
        singles.append(copy.deepcopy(atoms.calc.results))
    atoms.calc = MACECalculator(**{key: models}, device=device)
    atoms.get_potential_energy()
    for name in ("energy", "forces", "stress"):
        values = np.stack([result[name] for result in singles])
        close(atoms.calc.results[name], values.mean(axis=0))
        close(atoms.calc.results[f"{name}_var"], values.var(axis=0))
    assert atoms.calc.results["energy_var"] > 0
    close(calculate_numerical_forces(atoms, eps=1e-4), atoms.get_forces())
    roundtrip(atoms, tmp_path / "committee.xyz")


@pytest.mark.parametrize("family", ["dipole", "energy-dipole"])
def test_trained_response_calculators(trained_calculators, device, family, tmp_path):
    atoms = molecule(trained_calculators)
    atoms.calc = calculator(trained_calculators, device, family)
    dipole = atoms.get_dipole_moment()
    assert dipole.shape == (3,) and np.isfinite(dipole).all()
    if family == "energy-dipole":
        close(calculate_numerical_forces(atoms, eps=1e-4), atoms.get_forces())
        from mace_torch.models.dipoles import E_ANGSTROM_PER_DEBYE

        assert atoms.calc.models[0].model.backbone.layer_irreps[-1] == "4x0e+4x1o"
        energy = atoms.get_potential_energy()
        force = atoms.get_forces().copy()
        delta = np.array([0.1, -0.1, 0.0])
        atoms.arrays["Qs"] += delta
        close(atoms.get_dipole_moment() - dipole, delta @ atoms.positions / E_ANGSTROM_PER_DEBYE)
        close(atoms.get_potential_energy(), energy)
        close(atoms.get_forces(), force)
    roundtrip(atoms, tmp_path / "response.xyz")


def check_descriptors(calc, atoms):
    rotated = atoms.copy()
    rotated.rotate(90, "x", rotate_cell=True)
    for invariant, width in ((True, 4), (False, 16)):
        first = calc.get_descriptors(atoms, invariants_only=invariant, num_layers=1)
        full = calc.get_descriptors(atoms, invariants_only=invariant)
        other = calc.get_descriptors(rotated, invariants_only=invariant)
        assert first.shape == (3, width)
        assert full.shape == (3, width + 4)
        close(first, full[:, :width])
        if invariant:
            close(full, other)
        else:
            assert np.max(np.abs(first[:, 4:] - other[:, 4:width])) > FP64_CPU_REFERENCE.atol


def test_trained_descriptors_rotate_and_select_layers(trained_calculators, device):
    check_descriptors(calculator(trained_calculators, device), molecule(trained_calculators))


@pytest.mark.gpu
@pytest.mark.cueq
def test_trained_descriptors_on_cueq(trained_calculators, tmp_path):
    from mace_torch.backends.cueq import compiled_operations_available
    from mace_torch.calculators import MACECalculator

    # Backend selection is recorded in metadata; the canonical weights are shared.
    source = trained_calculators[0] / "energy-1/tiny.safetensors"
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata["config"]["config"]["resolved"]["model"]["backend"] = "cueq"
    target.with_suffix(".json").write_text(json.dumps(metadata))
    calc = MACECalculator(model_paths=target, device="cuda")
    reference = calculator(trained_calculators, "cpu")
    atoms = molecule(trained_calculators)
    called = set()
    backbone = calc.models[0].model.backbone
    ops = {name: module for name, module in backbone.named_modules()
           if type(module).__name__.startswith("CuEq")}
    assert ops and compiled_operations_available()
    hooks = [module.register_forward_hook(lambda m, a, o, name=name: called.add(name))
             for name, module in ops.items()]
    try:
        check_descriptors(calc, atoms)
        for invariant in (True, False):
            for layers in (1, -1):
                close(calc.get_descriptors(atoms, invariants_only=invariant, num_layers=layers),
                      reference.get_descriptors(atoms, invariants_only=invariant, num_layers=layers),
                      FP64_ACCELERATED_BACKEND)
    finally:
        for hook in hooks:
            hook.remove()
    assert called == ops.keys(), f"unused accelerated ops: {ops.keys() - called}"


def test_trained_calculator_padding(trained_calculators, device):
    atoms = molecule(trained_calculators, periodic=True)
    atoms.calc = calculator(trained_calculators, device)
    atoms.get_potential_energy()
    expected = copy.deepcopy(atoms.calc.results)
    atoms.calc = calculator(trained_calculators, device, pad_num_atoms=10, pad_num_edges=128)
    atoms.get_potential_energy()
    for name, value in expected.items():
        close(atoms.calc.results[name], value)
