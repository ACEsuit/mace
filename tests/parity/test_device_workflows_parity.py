"""Device placement through actual conversion and evaluation entry points."""

import hashlib
import subprocess
import sys

import numpy as np
import pytest
import torch
from ase.io import read, write
from mace_core.cli import main
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.deploy import loader
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.train.checkpoint import write_model

from tests.golden.harness import FP32, FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.process_state import capture_state, restore_state
from tests.parity.test_fm_convert_legacy import FIXTURES, GOLDEN, extract_here

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
pytestmark = pytest.mark.usefixtures("isolated")


@pytest.fixture(scope="module", params=["tiny_mace", "tiny_scaleshift"])
def checkpoints(request, tmp_path_factory):
    before = capture_state()
    try:
        torch.set_default_dtype(torch.float64)
        directory = tmp_path_factory.mktemp(request.param)
        legacy = GOLDEN / "models" / f"{request.param}.model"
        imported = import_neutral(
            extract_here(legacy, directory / "neutral"), DEFAULT_CATALOGUE
        )
        native = write_model(directory / "native", imported.engine, imported.metadata)
    finally:
        restore_state(before)
    return legacy, native


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_device_command_preserves_all_state_without_rewriting(
    checkpoints, tmp_path, monkeypatch, capsys, device, suffix
):
    from mace.cli.convert_device import main as legacy_main

    source, artifact = checkpoints
    original = torch.load(source, map_location="cpu", weights_only=False)
    destination = tmp_path / "moved.model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mace_convert_device",
            str(source),
            "--target_device",
            device,
            "--output_file",
            str(destination),
        ],
    )
    legacy_main()
    moved = torch.load(destination, weights_only=False)
    assert original.state_dict().keys() == moved.state_dict().keys()
    for name, expected in original.state_dict().items():
        actual = moved.state_dict()[name]
        assert actual.device.type == device and actual.dtype == expected.dtype
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    paths = [artifact.with_suffix(ext) for ext in (".json", ".safetensors")]
    before = {path: digest(path) for path in paths}
    directory_before = sorted(path.name for path in artifact.parent.iterdir())
    load = loader.load_deployed
    reference = load(artifact, device="cpu").engine.state_dict()
    observed = []

    def checked(*args, **kwargs):
        result = load(*args, **kwargs)
        for name, expected in reference.items():
            actual = result.engine.state_dict()[name]
            assert actual.device.type == device and actual.dtype == expected.dtype
            torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        observed.append(result)
        return result

    monkeypatch.setattr(loader, "load_deployed", checked)
    assert (
        main(
            [
                "model",
                "convert-device",
                "--model",
                str(artifact.with_suffix(suffix)),
                "--device",
                device,
            ]
        )
        == 0
    )
    assert len(observed) == 1
    assert "no rewrite is needed" in capsys.readouterr().out
    if suffix == ".safetensors":
        launched = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mace_launcher import convert_device; convert_device()",
                "--engine",
                "v1",
                "--model",
                str(artifact.with_suffix(suffix)),
                "--device",
                device,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "no rewrite is needed" in launched.stdout
    assert {path: digest(path) for path in paths} == before
    assert sorted(path.name for path in artifact.parent.iterdir()) == directory_before


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_eval_device_and_precision_match_the_actual_legacy_cli(
    checkpoints, tmp_path, monkeypatch, device, dtype
):
    from mace.cli.eval_configs import main as legacy_main

    legacy, native = checkpoints
    structures = [
        read(FIXTURES / name) for name in ("water_cluster.xyz", "triclinic_bulk.xyz")
    ]
    source = tmp_path / "structures.xyz"
    write(source, structures)
    expected_file, actual_file = tmp_path / "legacy.xyz", tmp_path / "native.xyz"
    flags = [
        "--configs",
        str(source),
        "--device",
        device,
        "--default_dtype",
        dtype,
        "--batch_size",
        "1",
        "--compute_stress",
        "--return_node_energies",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mace_eval_configs",
            "--model",
            str(legacy),
            "--output",
            str(expected_file),
            *flags,
        ],
    )
    legacy_main()
    observed = []
    load = loader.load_deployed

    def checked(*args, **kwargs):
        result = load(*args, **kwargs)
        assert next(result.engine.parameters()).device.type == device
        assert next(result.engine.parameters()).dtype == getattr(torch, dtype)
        observed.append(result)
        return result

    monkeypatch.setattr(loader, "load_deployed", checked)
    assert (
        main(["eval", "--model", str(native), "--output", str(actual_file), *flags])
        == 0
    )
    assert len(observed) == 1
    tolerance = (
        FP32
        if dtype == "float32"
        else FP64_ACCELERATED_BACKEND
        if device == "cuda"
        else FP64_CPU_REFERENCE
    )
    for actual, expected in zip(
        read(actual_file, index=":"), read(expected_file, index=":"), strict=True
    ):
        for key, store in (
            ("MACE_energy", "info"),
            ("MACE_forces", "arrays"),
            ("MACE_stress", "info"),
            ("MACE_node_energies", "arrays"),
        ):
            np.testing.assert_allclose(
                getattr(actual, store)[key],
                getattr(expected, store)[key],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )


def test_convert_device_does_not_accept_an_output_artifact(checkpoints, tmp_path):
    _, artifact = checkpoints
    output = tmp_path / "output.json"
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "model",
                "convert-device",
                "--model",
                str(artifact),
                "--device",
                "cpu",
                "--output_file",
                str(output),
            ]
        )
    assert stopped.value.code == 2
    assert not output.exists()
