"""The libmace seam requires a native package and a framework-free public ABI."""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from mace_torch.deploy.bundle import export_lammps

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def metadata_reader(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    output = tmp_path_factory.mktemp("native-metadata") / "reader"
    subprocess.run([
        compiler, "-std=c++17", str(Path(__file__).with_name("native_metadata.cpp")),
        "-I", str(ROOT / "packages/mace-torch/native"), "-o", str(output),
    ], check=True, capture_output=True, text=True)
    return output


def test_shared_native_metadata_preserves_values_and_element_order(metadata_reader):
    result = subprocess.run([str(metadata_reader)], check=True, capture_output=True, text=True)
    assert result.stdout == "8 32 16 5 float64 cpu\nH:1 C:6 O:8 "


@pytest.mark.parametrize("key,value", [
    ("mace_native_version", "1.1"), ("mace_dtype", "float16"),
    ("mace_nodes", "2"), ("mace_edges", "0"), ("mace_nodes", "8x"),
    ("mace_comm_width", "-1"), ("mace_cutoff", "inf"),
    ("mace_cutoff", "nan"), ("mace_cutoff", "0"),
    ("mace_backend", "unknown"), ("mace_device", "mps"),
    ("mace_device", "cuda:abc"), ("mace_atomic_numbers", "1 1 8"),
    ("mace_atomic_numbers", "1 6 119"), ("mace_atomic_numbers", "1 x 8"),
    ("mace_atomic_numbers", "1 6 8 999999999999999999999"),
    ("mace_species", "H H O"), ("mace_species", "H O"),
    ("mace_evaluation", "unknown"), ("mace_evaluation", "periodic_system"),
])
def test_shared_native_metadata_refuses_invalid_fields(metadata_reader, key, value):
    result = subprocess.run([str(metadata_reader), key, value], capture_output=True, text=True)
    assert result.returncode == 1 and result.stderr


@pytest.mark.parametrize("cell,valid", [
    ("6 0 0 0.2 7 0 0.1 -0.1 8", True),
    ("6 0 0 0 7 0 0 0 0", False),
    ("6 0 0 0 7 0 0 0 -8", False),
    ("6 0 0 0 7 0 0 0 nan", False),
    ("6 0 0 0 7 0 0 0 8 9", False),
])
def test_periodic_native_metadata_validates_cell(metadata_reader, cell, valid):
    result = subprocess.run([
        str(metadata_reader), "mace_evaluation", "periodic_system",
        "mace_comm_width", "0", "mace_cell", cell,
    ], capture_output=True, text=True)
    assert (result.returncode == 0) == valid


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
    assert captured["mace_native_version"] == "1.0"
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
