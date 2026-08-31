"""The launcher's legacy engine must change nothing at all.

`mace_run_train` is now a `mace-launcher` console script rather than a direct
pointer at `mace.cli.run_train:main`, so the whole value of the default engine
is that nobody can tell. These tests compare a run driven through the script
against the same run driven by calling the module directly, on both the flag
path and the config-file path.

The config-file case is separate on purpose: `--config` is a configargparse
config file, and it is the one argument where an extra pass over argv would
change which value wins rather than merely reorder something.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.helpers import REPO_ROOT, base_mace_params, run_mace_train

# Only the two byte-identity tests need the script on PATH. The ownership test
# reads installed metadata and must run everywhere, since a second distribution
# claiming a script is exactly the failure it exists to catch.
needs_console_script = pytest.mark.skipif(
    shutil.which("mace_run_train") is None,
    reason="mace-launcher is not installed, so the console script does not exist",
)

# The leading `%(asctime)s.%(msecs)03d` of every log line, plus anything that
# reports a duration. Two subprocesses can never match on these, whatever the
# dispatch does, so a comparison that keeps them tests the clock.
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} ")
DURATION = re.compile(r"\b\d+\.\d+s\b|\bin \d+\.\d+\b|\b\d+\.\d+ seconds\b")


def normalise(output: str) -> str:
    lines = []
    for line in output.splitlines():
        line = TIMESTAMP.sub("", line)
        line = DURATION.sub("<duration>", line)
        # Run directories differ between the two invocations by construction.
        line = re.sub(r"/[^\s'\"]*/pytest-[^\s'\"]*", "<tmp>", line)
        lines.append(line)
    return "\n".join(lines)


#: Both runs use one name, in separate directories. A `.model` is a zip whose
#: member names embed the run name, so two differently-named runs differ in the
#: zip header before any weight does, and the comparison could never pass.
RUN_NAME = "parity"


def tiny_params(work_dir: Path, fitting_configs) -> dict:
    """The smallest run that still exercises the whole CLI, seeded fixed."""
    from ase.io import write

    work_dir.mkdir(parents=True, exist_ok=True)
    name = RUN_NAME
    xyz = work_dir / f"{name}.xyz"
    write(str(xyz), fitting_configs)
    params = base_mace_params()
    params.update(
        {
            "name": name,
            "train_file": str(xyz),
            "valid_fraction": 0.5,
            "max_num_epochs": 1,
            "batch_size": 2,
            "valid_batch_size": 2,
            "seed": 42,
            "device": "cpu",
            "default_dtype": "float64",
            "hidden_irreps": "8x0e",
            "r_max": 3.0,
            "num_radial_basis": 4,
            "max_L": 0,
            "checkpoints_dir": str(work_dir / "ckpt"),
            "model_dir": str(work_dir / "model"),
            "log_dir": str(work_dir / "log"),
            "results_dir": str(work_dir / "results"),
        }
    )
    return params


def run_via_launcher(params: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    sys.path.insert(0, str(REPO_ROOT))
    env["PYTHONPATH"] = ":".join(sys.path)
    env["MACE_ENGINE"] = "legacy"
    cmd = ["mace_run_train"]
    for key, value in params.items():
        cmd.append(f"--{key}" if value is None else f"--{key}={value}")
    return subprocess.run(
        cmd, env=env, check=True, capture_output=True, text=True, cwd=tmp_path
    )


def model_bytes(params: dict) -> bytes:
    path = Path(params["model_dir"]) / f"{params['name']}.model"
    assert path.exists(), f"no model written at {path}"
    return path.read_bytes()


@pytest.mark.slow
@needs_console_script
def test_the_launcher_reproduces_a_direct_call_flag_for_flag(tmp_path, fitting_configs):
    direct_params = tiny_params(tmp_path / "direct", fitting_configs)
    launcher_params = tiny_params(tmp_path / "launcher", fitting_configs)

    direct = run_mace_train(direct_params, cwd=tmp_path, capture_output=True, text=True)
    launched = run_via_launcher(launcher_params, tmp_path)

    assert normalise(direct.stdout) == normalise(launched.stdout)

    assert model_bytes(direct_params) == model_bytes(launcher_params)


@pytest.mark.slow
@needs_console_script
def test_the_launcher_is_transparent_to_the_config_file_path(tmp_path, fitting_configs):
    """`--config` is configargparse's, and precedence is what an extra pass breaks."""
    direct_params = tiny_params(tmp_path / "cfg_direct", fitting_configs)
    launcher_params = tiny_params(tmp_path / "cfg_launcher", fitting_configs)

    direct_yaml = tmp_path / "direct.yaml"
    launcher_yaml = tmp_path / "launcher.yaml"
    direct_yaml.write_text(yaml.safe_dump(direct_params))
    launcher_yaml.write_text(yaml.safe_dump(launcher_params))

    direct = run_mace_train(
        {"config": str(direct_yaml)}, cwd=tmp_path, capture_output=True, text=True
    )
    launched = run_via_launcher({"config": str(launcher_yaml)}, tmp_path)

    assert normalise(direct.stdout) == normalise(launched.stdout)

    assert model_bytes(direct_params) == model_bytes(launcher_params)


def test_exactly_one_distribution_provides_each_console_script():
    """Two dists declaring one script name is undefined behaviour in pip."""
    from importlib.metadata import distributions

    # Deduplicated by distribution name: the same distribution is discovered
    # once per sys.path entry that contains it, and the tests put the repo on
    # the path more than once. What matters is how many distinct distributions
    # claim a script, never how many times one of them is seen.
    owners: dict[str, set[str]] = {}
    for distribution in distributions():
        name = distribution.metadata["Name"]
        for entry in distribution.entry_points:
            if entry.group == "console_scripts" and entry.name.startswith("mace_"):
                owners.setdefault(entry.name, set()).add(name)

    duplicated = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
    assert not duplicated, f"more than one distribution provides {duplicated}"

    launcher_pyproject = REPO_ROOT / "packages" / "mace-launcher" / "pyproject.toml"
    if launcher_pyproject.exists():
        import tomllib

        with launcher_pyproject.open("rb") as handle:
            declared = tomllib.load(handle)["project"]["scripts"]
        assert len(declared) == 12, f"expected twelve scripts, found {len(declared)}"
        # Installed or not, every declared script must be owned by the launcher
        # alone. json keeps the failure readable when it is not.
        wrong = {
            name: sorted(owners[name])
            for name in declared
            if name in owners and sorted(owners[name]) != ["mace-launcher"]
        }
        assert not wrong, f"scripts owned by something else: {json.dumps(wrong)}"
