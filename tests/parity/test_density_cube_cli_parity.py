"""Actual cube files and reports from native and frozen polar checkpoints."""

import json
import sys

import ase.io
import numpy as np
import pytest
import torch
from ase.io.cube import read_cube_data

from tests.golden.harness import FP32, FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.process_state import capture_state, restore_state

pytestmark = pytest.mark.polar


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """A parity-only conversion; both public loaders must rebuild their model."""
    from ase.data import chemical_symbols
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.metadata import (
        ConfigRecord,
        E0Details,
        HeadSummary,
        ModelMetadata,
        Provenance,
    )
    from mace_torch.physics import DerivativeEngine
    from mace_torch.train import write_model

    from tests.parity.fm00_convert import energy_constants_to_canonical
    from tests.parity.test_polar_parity import ENERGY, convert, legacy_polar

    before = capture_state()
    try:
        torch.set_default_dtype(torch.float64)
        directory = tmp_path_factory.mktemp("density_cube_models")
        legacy = legacy_polar()
        legacy_path = directory / "legacy.model"
        torch.save(legacy, legacy_path)
        model, settings = convert(legacy, "per_structure", None)
        values, _, _ = energy_constants_to_canonical(legacy)
        config = ResolvedConfig.model_validate(
            {
                "data": {
                    "heads": {
                        "default": {
                            "train_file": str(directory / "unused.xyz"),
                            "e0s": {"kind": "table", "values": values["default"]},
                        }
                    }
                },
                "model": {
                    **model.config.model_dump(),
                    "model": "polar",
                },
                "electrostatics": {
                    "enabled": True,
                    "periodicity_profile": "per_structure",
                    "kspace_cutoff_factor": float(legacy.kspace_cutoff_factor),
                },
            }
        )
        metadata = ModelMetadata(
            config=ConfigRecord(resolved=config.model_dump(mode="json")),
            provenance=Provenance(versions={"mace-torch-v1": "parity"}),
            heads={
                "default": HeadSummary(
                    e0=E0Details(
                        source="explicit",
                        values={
                            chemical_symbols[z]: value
                            for z, value in values["default"].items()
                        },
                    )
                )
            },
            elements=settings["atomic_numbers"],
        )
        native_path = write_model(
            directory / "native", DerivativeEngine(model, ENERGY), metadata
        )
    finally:
        restore_state(before)
    return native_path, legacy_path


# Each recipe exercises a different interpolation/periodicity boundary. Custom
# settings deliberately differ from both parser defaults and checkpoint values.
RECIPES = [
    ("molecule", "auto", "all", False),
    ("partial", "auto", "all", False),
    ("slab", "auto", "all", False),
    ("crystal", "auto", "all", False),
    ("molecule", "realspace", "alpha", True),
    ("partial", "realspace", "beta", True),
    ("slab", "realspace", "spin", True),
    ("crystal", "realspace", "charge", True),
    ("slab", "fourier", "all", True),
    ("crystal", "fourier", "all", True),
]


def input_file(directory, periodicity):
    from tests.parity.test_polar_density_cube_parity import structure

    flags = {
        "molecule": (False, False, False),
        "partial": (True, False, True),
        "slab": (True, True, False),
        "crystal": (True, True, True),
    }
    selected = structure(flags[periodicity])
    selected.info.update(
        charge=1.0,
        spin=2.0,
        external_field=np.array([0.01, -0.02, 0.03]),
        fermi_level=-0.15,
    )
    first = selected.copy()
    first.positions[1] += [0.2, 0.1, -0.1]
    first.info.update(charge=0.0, spin=1.0)
    path = directory / "structures.xyz"
    ase.io.write(path, [first, selected])
    return path, selected


def close(actual, expected, row):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=row.atol, rtol=row.rtol)


def compare_written(native, frozen, selected, row, quantity, potential, report):
    quantities = (
        ["charge", "spin", "alpha", "beta"] if quantity == "all" else [quantity]
    )
    expected_names = []
    for name in quantities:
        stem = "density_" + name if quantity == "all" else "density"
        expected_names += [
            stem + suffix + ".cube"
            for suffix in (
                ("", "_potential", "_potential_corrected") if potential else ("",)
            )
        ]
    if report:
        expected_names.append("quality.json")
    assert [p.name for p in native] == expected_names
    assert [p.name for p in frozen] == expected_names
    for actual_path, reference_path in zip(native, frozen, strict=True):
        if actual_path.suffix == ".json":
            actual, reference = (
                json.loads(actual_path.read_text()),
                json.loads(reference_path.read_text()),
            )
            assert actual.keys() == reference.keys()
            assert actual["backend"] == reference["backend"]
            assert actual["grid"] == reference["grid"]
            assert actual["quantities"].keys() == reference["quantities"].keys()
            for name in quantities:
                assert (
                    actual["quantities"][name].keys()
                    == reference["quantities"][name].keys()
                )
                for key in actual["quantities"][name]:
                    close(
                        actual["quantities"][name][key],
                        reference["quantities"][name][key],
                        row,
                    )
        else:
            actual, actual_atoms = read_cube_data(actual_path)
            reference, reference_atoms = read_cube_data(reference_path)
            assert actual.shape == (6, 5, 7)
            close(actual, reference, row)
            np.testing.assert_array_equal(actual_atoms.numbers, selected.numbers)
            close(actual_atoms.positions, reference_atoms.positions, FP64_CPU_REFERENCE)
            close(actual_atoms.cell, reference_atoms.cell, FP64_CPU_REFERENCE)
            # Both writers can round coordinates, but must select the requested frame.
            close(actual_atoms.positions, selected.positions, FP64_CPU_REFERENCE)


@pytest.mark.parametrize(
    "recipe", RECIPES, ids=["-".join(map(str, row)) for row in RECIPES]
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_density_cli_matches_frozen_files_and_reports(
    checkpoints, tmp_path, monkeypatch, isolated, recipe, dtype, device
):
    from mace_torch.cli import polar_density_cube as native

    from mace.cli import polar_density_cube as frozen

    periodicity, backend, quantity, custom = recipe
    source, selected = input_file(tmp_path, periodicity)
    potential = periodicity in ("slab", "crystal") and backend != "realspace"
    report = quantity == "all"
    common = [
        "--configs",
        str(source),
        "--index",
        "-1" if custom else "1",
        "--quantity",
        quantity,
        "--grid",
        "6",
        "5",
        "7",
        "--default_dtype",
        dtype,
        "--device",
        device,
        "--backend",
        backend,
    ]
    if custom:
        common += [
            "--sigma",
            "0.85",
            "--kspace_cutoff",
            "2.2",
            "--realspace_cutoff_factor",
            "4.5",
            "--chunk_size",
            "11",
            "--external_field",
            "0.07",
            "-0.04",
            "0.08",
            "--fermi_level",
            "0.35",
        ]
        if backend == "fourier":
            common += ["--subtract_total_charge"]
    if potential:
        common += ["--write_potential"]
    # Observe the real loader, without substituting a calculator or its outputs.
    real_loader, loaded = native.load_calculator, []

    def checked_loader(model, requested_device, requested_dtype):
        calculator, settings = real_loader(model, requested_device, requested_dtype)
        parameter = next(calculator.models[0].engine.parameters())
        assert parameter.device.type == device
        assert parameter.dtype == getattr(torch, dtype)
        loaded.append((requested_device, requested_dtype))
        return calculator, settings

    monkeypatch.setattr(native, "load_calculator", checked_loader)
    written = []
    for module, checkpoint, name in (
        (native, checkpoints[0], "native"),
        (frozen, checkpoints[1], "frozen"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        args = [
            *common,
            "--model",
            str(checkpoint),
            "--output",
            str(directory / "density.cube"),
        ]
        if report:
            args += ["--quality_report", str(directory / "quality.json")]
        if module is native:
            paths = module.run(module.build_parser().parse_args(args))
        else:
            monkeypatch.setattr(sys, "argv", ["mace_polar_density_cube", *args])
            paths = module.run(module.parse_args())
        assert sorted(p.name for p in directory.iterdir()) == sorted(
            p.name for p in paths
        )
        written.append(paths)
    assert loaded == [(device, dtype)]
    row = (
        FP32
        if dtype == "float32"
        else (FP64_ACCELERATED_BACKEND if device == "cuda" else FP64_CPU_REFERENCE)
    )
    compare_written(*written, selected, row, quantity, potential, report)


@pytest.mark.parametrize(
    "flags,error,match",
    [
        (["--grid", "0", "5", "7"], ValueError, "positive"),
        (["--grid", "-1", "5", "7"], ValueError, "positive"),
        (["--chunk_size", "0"], ValueError, "chunk_size"),
        (["--backend", "fourier"], ValueError, "real-space"),
        (["--write_potential"], NotImplementedError, "write_potential"),
        (["--quantity", "typo"], SystemExit, "2"),
        (["--backend", "typo"], SystemExit, "2"),
        (["--index", "4"], StopIteration, ""),
    ],
)
def test_invalid_density_requests_write_no_files(
    checkpoints, tmp_path, isolated, flags, error, match
):
    from mace_torch.cli.polar_density_cube import main

    source, _ = input_file(tmp_path, "molecule")
    output = tmp_path / "outputs"
    output.mkdir()
    with pytest.raises(error, match=match):
        main(
            [
                "--configs",
                str(source),
                "--model",
                str(checkpoints[0]),
                "--output",
                str(output / "density.cube"),
                "--quality_report",
                str(output / "quality.json"),
                "--grid",
                "6",
                "5",
                "7",
                "--default_dtype",
                "float64",
                *flags,
            ]
        )
    assert list(output.iterdir()) == []
