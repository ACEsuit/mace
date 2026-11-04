"""Public magnetic XYZ and polar ASE results against the frozen live models.

Conversion is confined to the parity harness. The public v1 interfaces consume
native models; no legacy checkpoint loader is added to the product.
"""

from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from mace_torch.calculators import MACECalculator

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.parity.test_calculator_families_parity import results_of


def close(actual, expected, row=FP64_CPU_REFERENCE):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)


@pytest.mark.magnetic
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("key", ["REF_magmom", "custom_moments"])
@pytest.mark.parametrize("magforces", [False, True])
def test_magnetic_xyz_values_keys_and_batch_slicing(
    fp64, isolated, tmp_path, dtype, batch_size, key, magforces
):
    from mace_torch.cli.eval_configs import main

    from tests.parity.magnetic_convert import write_magnetic_checkpoint
    from tests.parity.test_magnetic_parity import magnetic_surfaces
    from tests.parity.test_time_reversal_parity import moved, structure

    surfaces = magnetic_surfaces()
    legacy = surfaces.load_anchor()
    source, target = tmp_path / "input.xyz", tmp_path / "predictions.xyz"
    checkpoint = write_magnetic_checkpoint(legacy, tmp_path, source)
    structures = []
    for name in ("mag_fe_atom", "mag_fe3_canted", "crystal", "mag_fe_dimer_afm"):
        atoms = structure(name).copy()
        atoms.arrays[key] = atoms.arrays["REF_magmom"].copy()
        if key != "REF_magmom":
            # A plausible default must not hide a silently ignored custom key.
            atoms.arrays["REF_magmom"] = np.zeros((len(atoms), 3))
        structures.append(atoms)
    ase.io.write(source, structures, format="extxyz")
    dtype_name = str(dtype).split(".")[-1]
    assert (
        main(
            [
                "--configs",
                str(source),
                "--model",
                str(checkpoint),
                "--output",
                str(target),
                "--batch_size",
                str(batch_size),
                "--default_dtype",
                dtype_name,
                "--magmom_key",
                key,
                "--info_prefix",
                "prediction_",
            ]
            + (["--return_magforces"] if magforces else [])
        )
        == 0
    )
    original = ase.io.read(source, index=":")
    written = ase.io.read(target, index=":")
    assert len(written) == len(original)
    torch.set_default_dtype(dtype)
    legacy = legacy.to(dtype)
    row = FP32 if dtype == torch.float32 else FP64_CPU_REFERENCE
    for atoms, output in zip(original, written, strict=True):
        np.testing.assert_array_equal(output.arrays[key], atoms.arrays[key])
        atoms.arrays["REF_magmom"] = atoms.arrays[key].copy()
        reference = legacy(
            moved(surfaces.build_batch(legacy, atoms), dtype, "cpu"),
            training=False,
            compute_force=True,
            compute_magforces=True,
        )
        close(output.info["prediction_energy"], reference["energy"][0].detach(), row)
        close(output.arrays["prediction_forces"], reference["forces"].detach(), row)
        assert ("prediction_magforces" in output.arrays) == magforces
        assert "MACE_magforces" not in output.arrays
        if magforces:
            close(
                output.arrays["prediction_magforces"],
                reference["magforces"].detach(),
                row,
            )


@pytest.mark.magnetic
def test_missing_custom_moments_are_refused_before_writing(fp64, isolated, tmp_path):
    from mace_torch.cli.eval_configs import main

    from tests.parity.magnetic_convert import write_magnetic_checkpoint
    from tests.parity.test_magnetic_parity import magnetic_surfaces

    surfaces = magnetic_surfaces()
    source, target = tmp_path / "input.xyz", tmp_path / "output.xyz"
    ase.io.write(source, list(surfaces.magnetic_fixtures().values()))
    checkpoint = write_magnetic_checkpoint(surfaces.load_anchor(), tmp_path, source)
    with pytest.raises(ValueError, match="custom_moments"):
        main(
            [
                "--configs",
                str(source),
                "--model",
                str(checkpoint),
                "--output",
                str(target),
                "--magmom_key",
                "custom_moments",
                "--return_magforces",
            ]
        )
    assert not target.exists()


def polar_deployed(legacy):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.elements import AtomicNumberTable
    from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
    from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
    from mace_torch.deploy.loader import DeployedModel
    from mace_torch.physics import DerivativeEngine

    from tests.parity.test_polar_parity import ENERGY, convert

    model, settings = convert(legacy, "per_structure", None)
    names = ["energy", "forces", "stress"]
    config = ResolvedConfig.model_validate(
        {
            "model": {
                **model.config.model_dump(),
                "model": "polar",
                "observables": names,
            },
            "electrostatics": {"enabled": True, "periodicity_profile": "per_structure"},
        }
    )
    numbers = settings["atomic_numbers"]
    return DeployedModel(
        engine=DerivativeEngine(model, ENERGY).eval(),
        config=config,
        metadata=ModelMetadata(
            config=ConfigRecord(),
            provenance=Provenance(versions={"mace-torch-v1": "0"}, git_commit=None),
        ),
        z_table=AtomicNumberTable(numbers),
        heads=("default",),
        e0s={"default": dict.fromkeys(numbers, 0.0)},
        outputs=resolve_requested(names, DEFAULT_CATALOGUE),
        path=Path("polar"),
    )


@pytest.mark.polar
@pytest.mark.parametrize(
    "name", ["neutral molecule", "charged doublet in a field", "crystal", "slab"]
)
@pytest.mark.parametrize("agnostic", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("committee", [False, True])
def test_polar_ase_values_shapes_units_and_padding(
    fp64, isolated, name, agnostic, padded, committee
):
    from mace.calculators import MACECalculator as LegacyCalculator
    from tests.parity.test_polar_parity import CASES, legacy_polar, with_inputs

    legacies = [legacy_polar(agnostic)]
    if committee:
        second = legacy_polar(agnostic)
        with torch.no_grad():
            next(second.parameters()).mul_(1.01)
        legacies.append(second)
    # Legacy leaves dipoles in their model units regardless of length_units_to_A.
    # Direct dipole parity therefore uses the common unit domain.
    units = {"energy_units_to_eV": 2.0, "length_units_to_A": 1.0}
    native = MACECalculator(
        models=[polar_deployed(model) for model in legacies],
        pad_num_atoms=16 if padded else 0,
        pad_num_edges=512 if padded else 0,
        **units,
    )
    frozen = LegacyCalculator(
        models=legacies,
        device="cpu",
        default_dtype="float64",
        model_type="PolarMACE",
        **units,
    )
    # Reuse both calculators on a smaller structure: no stale or padding rows.
    for current in (name, "neutral molecule"):
        atoms, _, inputs = CASES[current]
        atoms = with_inputs(atoms, *inputs)
        actual, reference = results_of(native, atoms), results_of(frozen, atoms)
        shapes = {
            "energy": (),
            "forces": (len(atoms), 3),
            "dipole": (3,),
            "charges": (len(atoms),),
            "spins": (len(atoms),),
            "interaction_energy": (),
            "electrostatic_energy": (),
            "electron_energy": (),
            "density_coefficients": (len(atoms), 4),
            "spin_charge_density": (len(atoms), 2, 4),
            "fukui_functions": (len(atoms), 2),
        }
        for key, shape in shapes.items():
            assert np.shape(actual[key]) == shape, key
            close(actual[key], reference[key])
        assert set(actual) <= set(native.implemented_properties)
        if committee:
            assert reference["energy_var"] > 0
            for key in ("energy_var", "forces_comm", "dipole_var"):
                close(actual[key], reference[key])


@pytest.mark.parametrize("key", ["Qs", "REF_charges", "custom_charges"])
@pytest.mark.parametrize("padded", [False, True])
def test_charges_key_controls_the_dipole_baseline(fp64, isolated, key, padded):
    from mace.calculators import MACECalculator as LegacyCalculator
    from tests.parity.test_calculator_families_parity import deployed, with_inputs
    from tests.parity.test_dipole_parity import legacy_anchor

    legacy = legacy_anchor()
    atoms = with_inputs("water")
    charges = np.array([-0.8, 0.4, 0.4])
    atoms.arrays["Qs"] = np.zeros(len(atoms))
    atoms.arrays[key] = charges.copy()
    # The dedicated charges_key wins over arrays_keys in both stacks.
    kwargs = {"charges_key": key, "arrays_keys": {"charges": "unused"}}
    native = MACECalculator(
        models=deployed(legacy, "dipole"),
        **kwargs,
        pad_num_atoms=8 if padded else 0,
        pad_num_edges=64 if padded else 0,
    )
    frozen = LegacyCalculator(
        models=legacy,
        model_type="DipoleMACE",
        device="cpu",
        default_dtype="float64",
        **kwargs,
    )
    with_charge = results_of(native, atoms)["dipole"]
    close(with_charge, results_of(frozen, atoms)["dipole"])
    del atoms.arrays[key]
    without_charge = results_of(native, atoms)["dipole"]
    close(without_charge, results_of(frozen, atoms)["dipole"])
    # The frozen dipole family reports Debye; this conversion is its contract.
    from mace.modules.utils import compute_fixed_charge_dipole

    expected = compute_fixed_charge_dipole(
        torch.tensor(charges),
        torch.tensor(atoms.positions),
        torch.zeros(len(atoms), dtype=torch.long),
        num_graphs=1,
    )[0].numpy()
    assert np.linalg.norm(expected) > 0
    close(with_charge - without_charge, expected)


@pytest.mark.parametrize(
    "family", ["dipole", "dielectric", pytest.param("polar", marks=pytest.mark.polar)]
)
@pytest.mark.parametrize("committee", [False, True])
@pytest.mark.parametrize("length_factor", [0.5, 3.0])
def test_dipole_length_conversion_preserves_the_native_contract(
    fp64, isolated, family, committee, length_factor
):
    from mace.calculators import MACECalculator as LegacyCalculator
    from tests.parity.test_calculator_families_parity import (
        LEGACY_TYPE,
        deployed,
        with_inputs,
    )
    from tests.parity.test_dipole_parity import legacy_anchor, legacy_dielectric

    if family == "polar":
        from tests.parity.test_polar_parity import legacy_polar

        legacies = [legacy_polar() for _ in range(2 if committee else 1)]
        legacy_type = "PolarMACE"
    else:
        legacies = [
            legacy_anchor()
            if family == "dipole"
            else legacy_dielectric("8x0e + 8x1o + 8x2e")
            for _ in range(2 if committee else 1)
        ]
        legacy_type = LEGACY_TYPE[family]
    if committee:
        with torch.no_grad():
            next(legacies[1].parameters()).mul_(1.01)
    models = [
        polar_deployed(model) if family == "polar" else deployed(model, family)
        for model in legacies
    ]
    atoms = with_inputs("water")
    baseline = results_of(MACECalculator(models=models), atoms)
    converted = results_of(
        MACECalculator(models=models, length_units_to_A=length_factor), atoms
    )
    frozen = results_of(
        LegacyCalculator(
            models=legacies,
            model_type=legacy_type,
            device="cpu",
            default_dtype="float64",
        ),
        atoms,
    )
    frozen_scaled = results_of(
        LegacyCalculator(
            models=legacies,
            model_type=legacy_type,
            device="cpu",
            default_dtype="float64",
            length_units_to_A=length_factor,
        ),
        atoms,
    )
    assert np.linalg.norm(baseline["dipole"]) > FP64_CPU_REFERENCE.atol
    powers = {"dipole": 1}
    if committee:
        powers.update(dipole_comm=1, dipole_var=2)
        assert np.any(baseline["dipole_var"] > 0)
    for key, power in powers.items():
        close(baseline[key], frozen[key])
        close(frozen_scaled[key], frozen[key])
        close(converted[key], baseline[key] * length_factor**power)
