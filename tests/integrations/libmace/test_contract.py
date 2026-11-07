"""The libmace seam requires a native package and a framework-free public ABI."""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from mace_torch.deploy.bundle import export_lammps

ROOT = Path(__file__).resolve().parents[3]


def test_native_package_declares_atomic_numbers_without_another_weight_format(
    tmp_path, monkeypatch
):
    captured = {}

    def capture(program, *, package_path, inductor_configs):
        captured.update(inductor_configs["aot_inductor.metadata"])
        Path(package_path).write_bytes(b"contract-test")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", capture)
    bundle = export_lammps(
        ROOT / "packages/mace-jax/tests/fixtures/tiny.json",
        tmp_path / "bundle",
        halo_exchange=True,
        native_capacity=(8, 32),
    )
    assert captured["mace_atomic_numbers"] == "1 6 8"
    assert captured["mace_species"] == "H C O"
    assert {path.name for path in bundle.iterdir()} == {
        "deploy.json",
        "model.json",
        "model.safetensors",
        "native.pt2",
    }


@pytest.mark.parametrize("compiler,language", [("cc", "c"), ("c++", "c++")])
def test_public_header_compiles_without_torch_headers(compiler, language):
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"requires {compiler}")
    completed = subprocess.run(
        [
            executable,
            "-x",
            language,
            "-fsyntax-only",
            "-",
            "-I",
            str(ROOT / "packages/mace-torch/libmace"),
        ],
        input='#include "mace.h"\n',
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
