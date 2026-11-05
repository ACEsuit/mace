"""Downloaded inputs are reusable across runs without moving training outputs."""

import json
from pathlib import Path

import pytest
from conftest import fp64_only
from mace_core.artifacts import download, resolve_artifact
from mace_core.cli import main
from mace_core.config.legacy import LegacyFlagError
from mace_torch.cli.run_train import parse
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.replay import fetch
from mace_torch.train.checkpoint import latest_run_checkpoint, read_run_state
from test_mace_torch_full_batch import configuration
from test_mace_torch_replay import Downloads


@fp64_only
@pytest.mark.parametrize("source", ["flags", "yaml"])
def test_explicit_download_directory_is_refused_before_creating_outputs(
    tmp_path, source
):
    import yaml

    work, cache = tmp_path / "run", tmp_path / "cache"
    values = {"work_dir": str(work), "downloads_dir": str(cache)}
    if source == "yaml":
        path = tmp_path / "old.yaml"
        path.write_text(yaml.safe_dump(values))
        arguments = ["--config", str(path)]
    else:
        arguments = [
            word for key, value in values.items() for word in (f"--{key}", value)
        ]
    with pytest.raises(LegacyFlagError, match=r"downloads_dir.*XDG cache") as error:
        parse(arguments)
    assert "XDG_CACHE_HOME" in str(error.value) and "--work_dir" in str(error.value)
    assert not cache.exists() and not work.exists()
    values["downloads_dir"] = None
    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump(values))
    assert parse(["--config", str(path)]).runtime.work_dir == work


@fp64_only
def test_download_cache_is_shared_by_runs_and_unchanged_by_resume(
    tmp_path, monkeypatch
):
    document = configuration(tmp_path, max_num_epochs=2).to_resolved_dict()
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    training_source = Path(document["data"]["heads"]["default"]["train_file"])
    server = Downloads(training_source.read_bytes())
    dataset = fetch("mp", opener=server)
    assert dataset.parent == cache_root / "mace"
    document["data"]["heads"]["default"]["train_file"] = str(dataset)
    document["runtime"].update(name="cached", plot=False)
    config = tmp_path / "run.json"
    config.write_text(json.dumps(document))
    first, second = tmp_path / "first", tmp_path / "second"
    command = ["train", "--config", str(config)]
    assert main([*command, "--work_dir", str(first), "--max_num_epochs", "1"]) == 0
    checkpoint = first / "cached.json"
    weights = checkpoint.with_suffix(".safetensors").read_bytes()
    artifact_server = Downloads(weights)

    def retrieve(url, destination):
        return download(url, destination, opener=artifact_server, progress=None)

    url = "https://example.org/cached.safetensors"
    model_artifact = resolve_artifact(url, fetch=retrieve)
    assert model_artifact.parent == cache_root / "mace"
    cached = {path.name: path.read_bytes() for path in dataset.parent.iterdir()}
    assert main([*command, "--work_dir", str(second)]) == 0
    assert main([*command, "--work_dir", str(first), "--restart_latest"]) == 0
    assert fetch("mp", opener=server) == dataset and len(server.calls) == 1
    assert resolve_artifact(url, fetch=retrieve) == model_artifact
    assert len(artifact_server.calls) == 1 and model_artifact.read_bytes() == weights
    assert cached == {path.name: path.read_bytes() for path in dataset.parent.iterdir()}
    for work in (first, second):
        loaded = load_deployed(work / "cached.json")
        assert loaded.config.runtime.work_dir == work
        state = read_run_state(latest_run_checkpoint(work, "cached"))
        assert state.epoch == 2
        assert not (work / "downloads").exists()
