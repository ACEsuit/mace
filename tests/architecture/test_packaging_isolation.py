"""Each native distribution has one import root; one owns the CLI."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_packages_are_independent_and_only_torch_owns_scripts():
    scripts = {}
    names = set()
    for path in (ROOT / "packages").glob("*/pyproject.toml"):
        document = tomllib.loads(path.read_text())
        name = document["project"]["name"]
        assert name not in names
        names.add(name)
        for command, target in document["project"].get("scripts", {}).items():
            assert command not in scripts
            assert (command == "mace-jax" and name == "mace-jax") or (name == "mace-torch" and target.startswith("mace_torch.cli:"))
            scripts[command] = target
    assert "mace" in scripts and "mace_run_train" in scripts
    assert names == {"mace-core", "mace-torch", "mace-jax", "mace-backend-example"}
    assert not (ROOT / "mace").exists()
    assert not (ROOT / "setup.cfg").exists()


def test_workspace_has_one_native_toolchain():
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "build-system" not in document
    dev = document["dependency-groups"]["dev"]
    assert {"ruff==0.16.5", "ty==0.0.77", "prek==0.5.1"} <= set(x for x in dev if isinstance(x, str))
    assert not any(x.split("=")[0] in {"black", "isort", "pylint", "mypy"} for x in dev if isinstance(x, str))
