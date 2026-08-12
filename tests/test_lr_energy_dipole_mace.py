import inspect
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import ase.io
from ase import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.calculators import MACECalculator
from mace.tools.scatter import scatter_sum
from mace.tools import torch_geometric
from mace.tools.model_script_utils import _build_model
from mace.tools.scripts_utils import get_optimizer, get_params_options

GTOElectrostaticEnergy = pytest.importorskip(
    "graph_longrange.energy"
).GTOElectrostaticEnergy


torch.set_default_dtype(torch.float64)


@pytest.fixture(name="model_config")
def fixture_model_config():
    return {
        "r_max": 2.5,
        "num_bessel": 4,
        "num_polynomial_cutoff": 3,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 2,
        "hidden_irreps": o3.Irreps("8x0e + 8x1o + 8x2e"),
        "MLP_irreps": o3.Irreps("8x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": np.zeros(2),
        "avg_num_neighbors": 2.0,
        "atomic_numbers": [1, 8],
        "correlation": 2,
    }


def _configuration(
    positions,
    numbers=(8, 1, 1),
    oxn=(-2, 1, 1),
    total_charge=0.0,
    cell=None,
    pbc=None,
):
    positions = np.asarray(positions, dtype=float)
    properties = {
        "energy": 0.0,
        "forces": np.zeros_like(positions),
        "dipole": np.zeros(3),
        "oxn": np.asarray(oxn),
        "total_charge": total_charge,
    }
    return data.Configuration(
        atomic_numbers=np.asarray(numbers),
        positions=positions,
        properties=properties,
        property_weights={key: 1.0 for key in properties},
        cell=cell,
        pbc=pbc,
    )


def _batch(configs, cutoff=2.5):
    table = tools.AtomicNumberTable([1, 8])
    atomic_data = [
        data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
        for config in configs
    ]
    return next(
        iter(
            torch_geometric.dataloader.DataLoader(
                atomic_data, batch_size=len(atomic_data), shuffle=False
            )
        )
    )


def _forward(model, configs, **kwargs):
    return model(_batch(configs).to_dict(), compute_force=False, **kwargs)


def test_drop_in_yaml_construction_and_backward(tmp_path, model_config):
    base_yaml = "name: drop_in\nmodel: EnergyDipoleMACE\nloss: energy_forces_dipole\n"
    lr_yaml = base_yaml.replace("model: EnergyDipoleMACE", "model: LREnergyDipoleMACE")
    assert base_yaml.replace("EnergyDipoleMACE", "LREnergyDipoleMACE") == lr_yaml
    config_path = tmp_path / "lr.yaml"
    config_path.write_text(lr_yaml, encoding="utf-8")
    args = tools.build_default_arg_parser().parse_args(["--config", str(config_path)])

    backbone_config = dict(model_config)
    backbone_config.pop("interaction_cls_first")
    backbone_config.pop("MLP_irreps")
    backbone_config.pop("gate")
    backbone_config.pop("correlation")
    model = _build_model(args, backbone_config, None, ["Default"])
    assert isinstance(model, modules.LREnergyDipoleMACE)
    param_options = get_params_options(args, model)
    assert param_options["params"][-1]["name"] == "long_range"
    optimizer = get_optimizer(args, param_options)

    config = _configuration([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
    weight_before = model.charge_readout.weight.detach().clone()
    output = model(_batch([config]).to_dict(), training=True)
    loss = output["energy"].sum() + output["forces"].square().sum()
    loss = loss + output["dipole"].square().sum()
    loss.backward()
    assert model.charge_readout.weight.grad is not None
    assert torch.isfinite(model.charge_readout.weight.grad).all()
    optimizer.step()
    assert not torch.equal(weight_before, model.charge_readout.weight)


def test_cli_yaml_trains_when_only_model_name_changes(tmp_path):
    isolated_o = Atoms("O", positions=[[0, 0, 0]])
    isolated_h = Atoms("H", positions=[[0, 0, 0]])
    isolated_o.info.update(REF_energy=-1.0, config_type="IsolatedAtom")
    isolated_h.info.update(REF_energy=-0.5, config_type="IsolatedAtom")
    water = Atoms(
        "OHH", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]
    )
    configs = [isolated_o, isolated_h]
    for index in range(4):
        current = water.copy()
        current.positions[2, 2] += 0.03 * index
        current.info["REF_energy"] = -2.0 + 0.02 * index
        current.info["REF_dipole"] = np.array([0.1, 0.2, 0.03 * index])
        current.info["total_charge"] = 0.0
        current.new_array("REF_forces", np.zeros((3, 3)))
        current.new_array("oxn", np.array([-2.0, 1.0, 1.0]))
        configs.append(current)
    train_file = tmp_path / "train.xyz"
    ase.io.write(train_file, configs)

    base_yaml = f"""\
name: lr_yaml
train_file: {train_file}
checkpoints_dir: {tmp_path}
model_dir: {tmp_path}
valid_fraction: 0.25
model: EnergyDipoleMACE
hidden_irreps: 8x0e + 8x1o
r_max: 2.5
batch_size: 2
max_num_epochs: 1
device: cpu
default_dtype: float64
seed: 7
loss: energy_forces_dipole
energy_key: REF_energy
forces_key: REF_forces
dipole_key: REF_dipole
error_table: EnergyDipoleRMSE
eval_interval: 1
plot: false
"""
    lr_yaml = base_yaml.replace("model: EnergyDipoleMACE", "model: LREnergyDipoleMACE")
    assert base_yaml.replace("EnergyDipoleMACE", "LREnergyDipoleMACE") == lr_yaml
    config_path = tmp_path / "train.yaml"
    config_path.write_text(lr_yaml, encoding="utf-8")

    repo_root = Path(__file__).parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "mace" / "cli" / "run_train.py"),
            f"--config={config_path}",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    trained = torch.load(tmp_path / "lr_yaml.model", map_location="cpu", weights_only=False)
    assert isinstance(trained, modules.LREnergyDipoleMACE)


def test_interface_compatibility(model_config):
    assert inspect.signature(modules.LREnergyDipoleMACE.__init__) == inspect.signature(
        modules.EnergyDipoleMACE.__init__
    )
    assert inspect.signature(modules.LREnergyDipoleMACE.forward) == inspect.signature(
        modules.EnergyDipoleMACE.forward
    )
    torch.manual_seed(4)
    short_model = modules.EnergyDipoleMACE(**model_config)
    torch.manual_seed(4)
    long_model = modules.LREnergyDipoleMACE(**model_config)
    config = _configuration([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]])
    short_output = _forward(short_model, [config])
    long_output = _forward(long_model, [config])

    assert long_output.keys() == short_output.keys()
    for key in short_output:
        if short_output[key] is None:
            assert long_output[key] is None
        else:
            assert long_output[key].shape == short_output[key].shape
            assert long_output[key].dtype == short_output[key].dtype


def test_actual_final_charges_are_conserved_for_a_batch(model_config):
    model = modules.LREnergyDipoleMACE(**model_config)
    configs = [
        _configuration([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]], total_charge=1.25),
        _configuration([[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]], total_charge=-0.75),
    ]
    captured = {}

    def capture_source(_module, _args, kwargs):
        captured["source"] = kwargs["source_feats"]
        captured["batch"] = kwargs["batch"]

    handle = model.coulomb_energy.register_forward_pre_hook(
        capture_source, with_kwargs=True
    )
    _forward(model, configs)
    handle.remove()
    charges = captured["source"][:, 0]
    totals = scatter_sum(charges, captured["batch"], dim=0, dim_size=2)
    assert torch.allclose(totals, torch.tensor([1.25, -0.75]), atol=2e-14)


def test_distant_atom_changes_dipole_only_with_field_update(model_config):
    torch.manual_seed(12)
    model = modules.LREnergyDipoleMACE(**model_config)
    near = _configuration([[0, 0, 0], [0.8, 0, 0], [7.0, 1.0, 0]])
    far = _configuration([[0, 0, 0], [0.8, 0, 0], [10.0, 2.0, 0]])
    dipole_near = _forward(model, [near])["atomic_dipoles"][0]
    dipole_far = _forward(model, [far])["atomic_dipoles"][0]
    assert not torch.allclose(dipole_near, dipole_far, atol=1e-10, rtol=1e-8)

    model.num_field_updates = 0
    local_near = _forward(model, [near])["atomic_dipoles"][0]
    local_far = _forward(model, [far])["atomic_dipoles"][0]
    assert torch.allclose(local_near, local_far, atol=1e-12, rtol=1e-12)

    model.num_field_updates = 2
    assert torch.isfinite(_forward(model, [near])["atomic_dipoles"]).all()


def _open_energy(coulomb, source, distance):
    dtype = source.dtype
    return coulomb(
        k_vectors=torch.zeros((1, 3), dtype=dtype),
        k_norm2=torch.zeros(1, dtype=dtype),
        k_vector_batch=torch.zeros(1, dtype=torch.long),
        k0_mask=torch.ones(1, dtype=dtype),
        source_feats=source,
        node_positions=torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        volume=torch.ones(1, dtype=dtype),
        pbc=torch.zeros((1, 3), dtype=torch.bool),
    )[0]


def _cross_energy(coulomb, source_a, source_b, distance):
    return (
        _open_energy(coulomb, source_a + source_b, distance)
        - _open_energy(coulomb, source_a, distance)
        - _open_energy(coulomb, source_b, distance)
    )


def test_multipolar_energy_and_large_distance_scaling(model_config):
    model = modules.LREnergyDipoleMACE(**model_config)
    zero = torch.zeros((2, 4))
    q0, q1 = zero.clone(), zero.clone()
    q0[0, 0], q1[1, 0] = 1.0, 1.0
    p0, p1 = zero.clone(), zero.clone()
    p0[0, 3], p1[1, 3] = 1.0, 1.0  # x dipoles in [q, y, z, x]

    interactions = [
        (q0, q1, 2.0),
        (q0, p1, 4.0),
        (p0, p1, 8.0),
    ]
    for source_a, source_b, expected_ratio in interactions:
        energy_20 = _cross_energy(model.coulomb_energy, source_a, source_b, 20.0)
        energy_40 = _cross_energy(model.coulomb_energy, source_a, source_b, 40.0)
        assert energy_20.abs() > 1e-10
        assert torch.allclose(
            energy_20.abs() / energy_40.abs(),
            torch.tensor(expected_ratio),
            rtol=4e-2,
            atol=4e-2,
        )


def test_rotation_translation_permutation_and_multivalued_dipole(model_config):
    torch.manual_seed(9)
    model = modules.LREnergyDipoleMACE(**model_config)
    positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.1, 0.2]])
    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    base = _forward(model, [_configuration(positions)])
    rotated = _forward(model, [_configuration(positions @ rotation.T)])
    translated = _forward(model, [_configuration(positions + [2.3, -1.1, 0.7])])
    permutation = [0, 2, 1]
    permuted = _forward(
        model,
        [
            _configuration(
                positions[permutation],
                numbers=np.asarray([8, 1, 1])[permutation],
                oxn=np.asarray([-2, 1, 1])[permutation],
            )
        ],
    )
    # graph_longrange's real-space l=1 projection uses a 0.1 A finite
    # displacement, so rotations are equivariant up to that approximation.
    assert torch.allclose(base["energy"], rotated["energy"], atol=2e-5)
    assert torch.allclose(
        rotated["dipole"][0],
        torch.as_tensor(rotation) @ base["dipole"][0],
        atol=1e-3,
    )
    assert torch.allclose(base["energy"], translated["energy"], atol=2e-8)
    assert torch.allclose(base["dipole"], translated["dipole"], atol=2e-8)
    assert torch.allclose(base["energy"], permuted["energy"], atol=2e-8)
    assert torch.allclose(base["dipole"], permuted["dipole"], atol=2e-8)

    cell = np.eye(3) * 8.0
    periodic = _configuration(
        [[0.1, 0, 0], [2.0, 0, 0], [0, 2.0, 0]],
        cell=cell,
        pbc=(True, True, True),
    )
    shifted = deepcopy(periodic)
    shifted.positions[1] += cell[0]
    output_0 = _forward(model, [periodic])
    output_1 = _forward(model, [shifted])
    assert torch.allclose(output_0["energy"], output_1["energy"], atol=2e-8)
    assert torch.allclose(
        output_1["dipole"] - output_0["dipole"],
        torch.tensor([[8.0, 0.0, 0.0]]),
        atol=2e-8,
    )


def test_forces_periodic_reference_and_serialization(tmp_path, model_config):
    torch.manual_seed(21)
    model = modules.LREnergyDipoleMACE(**model_config)
    config = _configuration([[0, 0, 0], [0.9, 0.2, 0], [0.1, 1.0, 0.3]])
    batch = _batch([config])
    output = model(batch.to_dict(), training=True)
    epsilon = 2e-5
    plus = deepcopy(config)
    minus = deepcopy(config)
    plus.positions[1, 0] += epsilon
    minus.positions[1, 0] -= epsilon
    finite_difference = -(
        _forward(model, [plus])["energy"] - _forward(model, [minus])["energy"]
    ) / (2 * epsilon)
    assert torch.allclose(
        output["forces"][1, 0], finite_difference[0], rtol=2e-3, atol=2e-4
    )

    periodic = _configuration(
        [[0.1, 0, 0], [1.5, 0.2, 0], [0.2, 1.6, 0.1]],
        cell=np.eye(3) * 7.0,
        pbc=(True, True, True),
    )
    captured = {}

    def capture(_module, _args, kwargs, result):
        captured.update(kwargs)
        captured["energy"] = result

    handle = model.coulomb_energy.register_forward_hook(capture, with_kwargs=True)
    periodic_output = _forward(model, [periodic], compute_stress=True)
    handle.remove()
    reference = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=1.5,
        kspace_cutoff=float(model.kspace_cutoff),
        include_self_interaction=True,
    )
    reference_inputs = {
        key: value.detach() for key, value in captured.items() if key != "energy"
    }
    reference_energy = reference(**reference_inputs)
    assert torch.allclose(captured["energy"], reference_energy, atol=2e-10)
    assert torch.isfinite(periodic_output["stress"]).all()
    assert periodic_output["stress"].abs().max() > 1e-10

    path = tmp_path / "lr.model"
    torch.save(model, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded.num_field_updates == 1
    assert loaded.atomic_multipoles_smearing_width == 1.5
    before = _forward(model, [config])
    after = _forward(loaded, [config])
    for key in before:
        if before[key] is not None:
            assert torch.allclose(before[key], after[key])

    atoms = Atoms("OHH", positions=config.positions)
    atoms.arrays["oxn"] = np.array([-2.0, 1.0, 1.0])
    atoms.info["charge"] = 0.0
    atoms.calc = MACECalculator(
        models=loaded,
        device="cpu",
        default_dtype="float64",
        model_type="LREnergyDipoleMACE",
        arrays_keys={"oxn": "oxn"},
    )
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_dipole_moment()).all()
