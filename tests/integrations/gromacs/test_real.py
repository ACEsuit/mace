"""Unsupported models and inputs must fail before starting MD."""

import os
import subprocess
import pytest
from ase import Atoms
from tests.integrations.gromacs._harness import prepare

pytestmark = pytest.mark.bin_gromacs


@pytest.mark.parametrize(
    "setting,message",
    [
        ("nnpot-model-input4 = atom-numbers", "Duplicate MACE model input"),
        ("nnpot-model-input4 = atom-pairs", "requires exactly atom-positions"),
        ("nnpot-modelfile = legacy.model", "existing native.pt2"),
        ("nnpot-input-group = Subset", "requires the whole System group"),
        ("nnpot-embedding = electrostatic-model", "requires the whole System group"),
        ("pbc = no", "only full pbc or pbc=xy with walls is supported"),
    ],
)
def test_invalid_configuration_rejected(tmp_path, setting, message):
    package = tmp_path / "native.pt2"
    package.touch()
    prepare(
        tmp_path,
        Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[20] * 3, pbc=True),
        package,
    )
    key = setting.split("=")[0].strip()
    path = tmp_path / "run.mdp"
    path.write_text(
        "\n".join(
            setting if line.split("=")[0].strip() == key else line
            for line in path.read_text().splitlines()
        )
        + "\n"
    )
    if not any(
        line.split("=")[0].strip() == key for line in path.read_text().splitlines()
    ):
        with path.open("a") as file:
            file.write(setting + "\n")
    (tmp_path / "groups.ndx").write_text("[ System ]\n1 2\n[ Subset ]\n1\n")
    completed = subprocess.run(
        [
            os.environ["MACE_GROMACS_BINARY"],
            "grompp",
            "-f",
            "run.mdp",
            "-c",
            "input.g96",
            "-p",
            "system.top",
            "-n",
            "groups.ndx",
            "-o",
            "run.tpr",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert message in completed.stderr
