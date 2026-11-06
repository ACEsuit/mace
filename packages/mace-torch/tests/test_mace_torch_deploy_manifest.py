"""The bundle stores the checkpoint unchanged and fixes the runtime choices."""

import json
import shutil

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.cli import main
from mace_torch.calculators import MACECalculator
from mace_torch.deploy import DeployManifest, load_deployed
from mace_torch.deploy.bundle import export_lammps
from test_mace_torch_calculator import water
from test_mace_torch_cli_commands import artifact as artifact


@fp64_only
def test_bundle_roundtrip_and_cli(artifact, tmp_path):
    assert (
        main(
            [
                "export",
                "lammps",
                str(artifact),
                "--no-halo-exchange",
                "--no-compile-regions",
            ]
        )
        == 0
    )
    bundle = artifact.with_suffix(".mace-deploy")
    assert {p.name for p in bundle.iterdir()} == {
        "deploy.json",
        "model.json",
        "model.safetensors",
    }
    for suffix in (".json", ".safetensors"):
        assert (
            artifact.with_suffix(suffix).read_bytes()
            == (bundle / ("model" + suffix)).read_bytes()
        )
    deployed = load_deployed(bundle)
    manifest = deployed.manifest
    assert manifest is not None
    assert not manifest.halo_exchange and not manifest.compile_regions
    assert manifest.backend is None and deployed.config.model.backend == "reference"
    assert (
        manifest.receptive_field
        == deployed.r_max * deployed.config.model.num_interactions
    )
    assert manifest.species == tuple(deployed.z_table.zs)
    assert deployed.select_head() == manifest.head
    for source in (str(bundle), bundle):
        atoms = water(periodic=True)
        atoms.calc = MACECalculator(artifact)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = MACECalculator(source)
        assert energy == atoms.get_potential_energy()
        np.testing.assert_array_equal(forces, atoms.get_forces())
    with pytest.raises(FileExistsError):
        export_lammps(artifact, bundle, halo_exchange=False)
    with pytest.raises(ValueError, match="head"):
        export_lammps(
            artifact, tmp_path / "bad-head", halo_exchange=True, head="missing"
        )
    assert not (tmp_path / "bad-head").exists()


@fp64_only
def test_manifest_validation_and_runtime_pins(artifact, tmp_path):
    bundle = export_lammps(
        artifact, halo_exchange=True, backend="reference", dtype="float32"
    )
    record = bundle / "deploy.json"
    raw = json.loads(record.read_bytes())
    deployed = load_deployed(bundle)
    assert deployed.manifest is not None and deployed.manifest.dtype == "float32"
    assert next(deployed.engine.parameters()).dtype == torch.float32
    with pytest.raises(ValueError, match="pins backend"):
        load_deployed(bundle, backend="cueq")
    for change, message in (
        ({"species": list(reversed(raw["species"]))}, "species"),
        ({"head": "absent"}, "head"),
        (
            {"num_interactions": 3, "receptive_field": 3 * raw["r_max"]},
            "num_interactions",
        ),
        ({"r_max": 0, "receptive_field": 0}, "r_max"),
        ({"receptive_field": -1}, "receptive_field"),
        ({"halo_exchange": "false"}, "halo_exchange"),
        ({"schema_version": "2.0"}, "schema_version"),
        ({"schema_version": "1.1"}, "schema_version"),
        ({"schema_version": "0.9"}, "schema_version"),
        ({"unknown": True}, "unknown"),
    ):
        record.write_text(json.dumps({**raw, **change}))
        with pytest.raises(ValueError, match=message):
            load_deployed(bundle)
    del raw["halo_exchange"]
    record.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="halo_exchange"):
        DeployManifest.read(record)
    source = tmp_path / "incomplete"
    shutil.copyfile(artifact, source.with_suffix(".json"))
    with pytest.raises(FileNotFoundError):
        export_lammps(source.with_suffix(".json"), halo_exchange=True)
    assert not source.with_suffix(".mace-deploy").exists()


@fp64_only
def test_multihead_bundle_uses_the_exported_head(tmp_path):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.graph.padding import FixedPadding
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.train import run_data_stage, run_model_stage, write_model
    from test_mace_torch_full_batch import configuration

    doc = configuration(tmp_path, max_num_epochs=1).to_resolved_dict()
    head = doc["data"]["heads"]["default"]
    doc["data"]["heads"] = {"pbe": head, "scan": head}
    config = ResolvedConfig.model_validate(doc)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    source = write_model(tmp_path / "multi", built.model, built.metadata)
    with pytest.raises(ValueError, match="head"):
        export_lammps(source, halo_exchange=False)
    pad = FixedPadding(nodes_budget=16, edges_budget=128, graphs_budget=2)
    bundle = export_lammps(source, halo_exchange=False, head="scan", pad_spec=pad)
    deployed = load_deployed(bundle)
    assert deployed.manifest is not None and deployed.manifest.pad_spec == pad
    assert deployed.heads == ("pbe", "scan") and deployed.select_head() == "scan"
    with pytest.raises(ValueError, match="head"):
        MACECalculator(bundle, head="pbe")
    atoms = water()
    original = MACECalculator(source, head="scan")
    original.calculate(atoms)
    packaged = MACECalculator(bundle)
    packaged.calculate(atoms)
    np.testing.assert_array_equal(
        packaged.results["forces"], original.results["forces"]
    )
    graph, _ = original._graph(atoms, padded=False)
    graph["head"].zero_()
    result = deployed.compute(graph)
    assert result.total_energy is not None
    assert result.total_energy.item() == original.results["energy"]
    assert graph["head"].item() == 0
