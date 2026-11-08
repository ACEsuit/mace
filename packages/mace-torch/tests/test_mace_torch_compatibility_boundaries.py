"""Executable replacements and refusals for the frozen MERGE/DROP surfaces."""

import hashlib
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from ase import Atoms
from conftest import fp64_only
from mace_core.cli import main
from mace_core.config.legacy import LegacyFlagError
from mace_core.foundation_roster import roster_entry
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import parse
from mace_torch.deploy.loader import load_deployed
from mace_torch.train.checkpoint import latest_run_checkpoint, read_run_state
from safetensors.torch import load_file
from test_mace_torch_cli_commands import artifact as artifact
from test_mace_torch_cli_commands import migrate_config
from test_mace_torch_full_batch import configuration


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("flag", ["log_dir", "model_dir", "checkpoints_dir"])
def test_output_directory_alias_trains_and_resumes_the_same_run(tmp_path, flag):
    config = configuration(tmp_path, max_num_epochs=2).to_resolved_dict()
    config["runtime"].update(plot=False, name="location")
    config["precision"]["model"] = str(torch.get_default_dtype()).split(".")[-1]
    source = tmp_path / "run.json"
    source.write_text(json.dumps(config))
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    command = ["train", "--config", str(source)]
    assert main([*command, "--work_dir", str(reference)]) == 0
    assert main([*command, f"--{flag}", str(resumed), "--max_num_epochs", "1"]) == 0
    assert main([*command, f"--{flag}", str(resumed), "--restart_latest"]) == 0
    checkpoints = []
    for directory in (reference, resumed):
        path = latest_run_checkpoint(directory, "location")
        assert path is not None
        checkpoints.append(path)
    states = [read_run_state(path) for path in checkpoints]
    assert states[0] == states[1]
    weights = [load_file(str(path.with_suffix(".safetensors"))) for path in checkpoints]
    assert weights[0].keys() == weights[1].keys()
    for name, value in weights[0].items():
        torch.testing.assert_close(value, weights[1][name], atol=0, rtol=0)
    records = [json.loads(path.read_text()) for path in checkpoints]
    for name in ("optimizer", "scheduler", "ema"):
        assert records[0][name] == records[1][name]
    for directory in (reference, resumed):
        assert (directory / "location.json").is_file()
        assert (directory / "location.safetensors").is_file()
        assert (directory / "logs/location.log").is_file()
        assert (directory / "results/location.jsonl").is_file()
        assert not list(directory.rglob("*.model"))
        assert not list(directory.rglob("*.pt"))
        assert (
            load_deployed(directory / "location.json").config.runtime.work_dir
            == directory
        )


@fp64_only
@pytest.mark.parametrize("source", ["flags", "yaml"])
def test_directory_aliases_merge_only_when_their_roots_agree(tmp_path, source):
    import yaml

    root = tmp_path / "run"
    values = {
        key: str(root)
        for key in (
            "work_dir",
            "log_dir",
            "model_dir",
            "checkpoints_dir",
        )
    }

    def arguments():
        if source == "flags":
            return [
                word for key, value in values.items() for word in (f"--{key}", value)
            ]
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.safe_dump(values))
        return ["--config", str(migrate_config(path))]

    assert parse(arguments()).runtime.work_dir == root
    values["model_dir"] = str(tmp_path / "different")
    with pytest.raises(LegacyFlagError, match="one --work_dir"):
        parse(arguments())
    assert not root.exists() and not (tmp_path / "different").exists()


@fp64_only
@pytest.mark.parametrize("other", [{}, {"model_paths": "missing.json"}, {"models": []}])
def test_singular_model_path_is_refused_before_loading(other):
    with pytest.raises(TypeError, match=r"model_path.*model_paths"):
        MACECalculator(model_path="retired.model", **other)


@fp64_only
def test_singular_model_path_cannot_be_silently_ignored_with_a_valid_model(artifact):
    with pytest.raises(TypeError, match=r"model_path.*model_paths"):
        MACECalculator(model_paths=artifact, model_path=artifact)


@fp64_only
@pytest.mark.parametrize(
    "module", ["convert_e3nn_oeq", "convert_oeq_e3nn", "convert_e3nn_hybrid"]
)
@pytest.mark.parametrize(
    "option",
    [[], ["--output_model", "unused.model"], ["--device", "cuda"], ["--return_model"]],
)
def test_backend_converter_arguments_cannot_rewrite_a_canonical_artifact(
    artifact, module, option
):
    paths = [artifact.with_suffix(suffix) for suffix in (".json", ".safetensors")]
    before = [digest(path) for path in paths]
    with pytest.raises(ModuleNotFoundError) as missing:
        importlib.import_module(f"mace_torch.cli.{module}")
    assert missing.value.name == f"mace_torch.cli.{module}"
    with pytest.raises(SystemExit) as rejected:
        main(
            [
                "model",
                module.removeprefix("convert_").replace("_", "-"),
                str(artifact),
                *option,
            ]
        )
    assert rejected.value.code == 2
    assert [digest(path) for path in paths] == before
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = MACECalculator(model_paths=artifact)
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


@fp64_only
def test_lammps_export_requires_a_v1_checkpoint(tmp_path, capsys, monkeypatch):
    import mace_launcher

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ImportError):
        exec("from mace_torch.calculators import LAMMPS_MACE", {})
    with pytest.raises(SystemExit) as missing:
        main(["export", "lammps"])
    assert missing.value.code == 2
    monkeypatch.setattr(
        sys, "argv", ["mace_create_lammps_model", "--engine", "v1", "missing.model"]
    )
    with pytest.raises(SystemExit) as rejected:
        mace_launcher.create_lammps_model()
    assert rejected.value.code == 2
    assert "V1 .safetensors" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@fp64_only
def test_retired_foundation_has_a_replacement_and_no_singular_loader():
    entry = roster_entry("mace_anicc", "default")
    assert entry.family is None
    assert entry.replaced_by == "MACE-OFF23"
    assert entry.reason is not None
    assert "superseded" in entry.reason
    replacement = roster_entry("mace_off", "small")
    assert replacement.family == "scale_shift" and replacement.replaced_by is None
    with pytest.raises(ImportError):
        exec("from mace_torch.calculators import mace_anicc", {})


@fp64_only
def test_native_loading_and_evaluation_ignore_retired_numerics_and_pickle_switches(
    artifact, tmp_path
):
    # A fresh process keeps the live oracle's environment and imports out of
    # this check. The replacement performs real collation and derivatives.
    script = r"""
import importlib.abc, json, os, sys
from pathlib import Path
class NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mace', 'torch_geometric', 'e3nn'}:
            raise AssertionError('retired dependency reached: ' + fullname)
sys.meta_path.insert(0, NoLegacy())
before = os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('native inference reached pickle or TorchScript')
torch.load = torch.save = torch.jit.script = torch.jit.save = forbidden
from ase import Atoms
from mace_core.outputs import MACEOutput
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.loader import load_deployed
model = load_deployed(sys.argv[1])
calculator = MACECalculator(models=model)
atoms = Atoms('OH2', positions=[[0,0,0], [0.96,0,0], [-0.24,0.93,0]],
              cell=[5,5,5], pbc=True)
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces, stress = atoms.get_forces(), atoms.get_stress()
graph, _ = calculator._graph(atoms, padded=False)
assert isinstance(graph, dict)
assert isinstance(model.compute(graph), MACEOutput)
assert os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD') == before
assert not any(name == 'mace' or name.startswith('mace.') for name in sys.modules)
Path(sys.argv[2]).write_text(json.dumps({'energy':energy,'forces':forces.tolist(),'stress':stress.tolist()}))
"""
    outputs = []
    before = [
        digest(artifact.with_suffix(suffix)) for suffix in (".json", ".safetensors")
    ]
    for value in (None, "0", "1"):
        env = dict(os.environ)
        for key in ("MACE_USE_CUEQ_CG", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"):
            env.pop(key, None)
            if value is not None:
                env[key] = value
        output = tmp_path / f"environment-{value}.json"
        process = subprocess.run(
            [sys.executable, "-c", script, str(artifact), str(output)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1] == outputs[2]
    assert [
        digest(artifact.with_suffix(suffix)) for suffix in (".json", ".safetensors")
    ] == before
