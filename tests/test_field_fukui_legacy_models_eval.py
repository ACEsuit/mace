import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from mace.calculators.mace import MACECalculator
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace import data as mace_data
from mace.tools import torch_geometric, utils

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODELS = [
    ("mace-fukui-spin-3L.model", -2079.864990234375),
    ("mace-fukui-spin-2L.model", -2079.86376953125),
    ("mace-fukui-spin-1L.model", -2079.86474609375),
]


def _water_atoms() -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2390, 0.9270, 0.0],
            ],
            dtype=float,
        ),
    )


def _build_model_batch(model: torch.nn.Module) -> dict:
    atoms = _water_atoms()
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    keyspec = mace_data.KeySpecification(
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        },
        arrays_keys={"Qs": "charges"},
    )
    config = mace_data.config_from_atoms(
        atoms, key_specification=keyspec, head_name="Default"
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace_data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=model.heads,
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to("cpu").to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            batch[key] = value.to(model_dtype)
    return batch


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


@pytest.mark.parametrize("model_name, expected_energy", LEGACY_MODELS)
def test_legacy_fieldfukui_model_evaluates(model_name, expected_energy):
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model = torch.load(str(model_path), map_location="cpu")
    assert model.__class__.__name__ == "FieldFukuiMACE"

    atoms = _water_atoms()
    calc = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="FieldFukuiMACE",
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert abs(float(energy) - expected_energy) < 1e-5


@pytest.mark.parametrize("model_name, _", LEGACY_MODELS)
def test_legacy_fieldfukui_models_run_with_and_without_cueq(model_name, _):
    if os.environ.get("SKIP_CUEQ_TESTS") == "1":
        pytest.skip("Skipping CuEq test by environment request")

    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    atoms = _water_atoms()
    calc_e3 = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="FieldFukuiMACE",
        enable_cueq=False,
    )
    atoms.calc = calc_e3
    energy_e3 = float(atoms.get_potential_energy())
    assert np.isfinite(energy_e3)

    calc_cueq = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="FieldFukuiMACE",
        enable_cueq=True,
    )
    atoms.calc = calc_cueq
    energy_cueq = float(atoms.get_potential_energy())
    assert np.isfinite(energy_cueq)

    # True CuEq on CPU should remain tightly matched to e3nn.
    assert abs(energy_cueq - energy_e3) <= 3.0e-4


@pytest.mark.parametrize("model_name, _", LEGACY_MODELS)
def test_legacy_fieldfukui_true_cueq_matches_e3nn(model_name, _):
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model_e3 = torch.load(str(model_path), map_location="cpu").eval()
    batch = _build_model_batch(model_e3)

    out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
    model_cueq = run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval()
    out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

    energy_max_abs = torch.max(torch.abs(out_cueq["energy"] - out_e3["energy"])).item()
    forces_max_abs = torch.max(torch.abs(out_cueq["forces"] - out_e3["forces"])).item()
    stress_max_abs = torch.max(torch.abs(out_cueq["stress"] - out_e3["stress"])).item()

    # Tight CPU parity bounds for true CuEq conversion (no calculator fallback).
    assert energy_max_abs <= 3.0e-4
    assert forces_max_abs <= 2.5e-4
    assert stress_max_abs <= 1.0e-8


def test_legacy_fieldfukui_2l_true_cueq_matches_e3nn_float64():
    model_name = "mace-fukui-spin-2L.model"
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model_e3 = torch.load(str(model_path), map_location="cpu").eval().double()
    torch.set_default_dtype(torch.float64)
    batch = _build_model_batch(model_e3)

    out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
    model_cueq = (
        run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval().double()
    )
    out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

    energy_max_abs = torch.max(torch.abs(out_cueq["energy"] - out_e3["energy"])).item()
    forces_max_abs = torch.max(torch.abs(out_cueq["forces"] - out_e3["forces"])).item()
    stress_max_abs = torch.max(torch.abs(out_cueq["stress"] - out_e3["stress"])).item()

    assert energy_max_abs <= 1.0e-7
    assert forces_max_abs <= 1.0e-7
    assert stress_max_abs <= 1.0e-11
