"""CLI results are checked after reading the artifacts with their public readers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import read, write
from ase.stress import full_3x3_to_voigt_6_stress
from mace_core.cli import main
from mace_core.config.model import GraphFeatureConfig
from mace_core.config.resolved import ResolvedConfig
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import parse, train
from mace_torch.deploy.loader import load_deployed
from mace_torch.train import run_data_stage, run_model_stage
from mace_torch.train.checkpoint import write_model
from test_mace_torch_calculator import water
from test_mace_torch_full_batch import CATALOGUE, configuration

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.helpers import REPO_ROOT, cli_command


def close(actual, expected):
    np.testing.assert_allclose(
        actual, expected, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol
    )


@pytest.fixture
def artifact(tmp_path):
    config = configuration(tmp_path, max_num_epochs=1)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "model": "plain",
                    "clebsch_gordan_basis": "full",
                    "scaling": "std",
                }
            )
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    write_model(tmp_path / "source", built.model, built.metadata)
    return tmp_path / "source.json"


def test_tree_help_imports_no_framework():
    code = "from mace_core.cli import main; import sys; main(['--help'])"
    checked = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr
    assert all(
        word in checked.stdout
        for word in ("eval", "model", "train", "data", "export", "plot", "md")
    )
    code = (
        "from mace_core.cli import build_parser, discover; import "
        "sys; build_parser(discover()); assert not ({'torch', 'jax', "
        "'mace'} & set(sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_and_legacy_configuration_precedence(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "name: old\nr_max: 4.0\nnum_channels: 8\nmax_num_epochs: 2\nplot: false\n"
    )
    config = parse(["--config", str(path), "--name", "new", "--r_max", "3.0"])
    assert config.runtime.name == "new"
    assert config.model.r_max == 3.0 and config.model.num_channels == 8
    assert config.training.max_num_epochs == 2 and not config.runtime.plot
    path.write_text(yaml.safe_dump(config.to_resolved_dict()))
    reread = parse(["--config", str(path), "--max_num_epochs", "4"])
    assert reread.training.max_num_epochs == 4
    assert reread.model == config.model


def test_unknown_config_and_options_are_errors(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("training: {learning_rtae: 0.1}")
    assert main(["train", "--config", str(path)]) == 2
    with pytest.raises(SystemExit):
        parse(["--runtime.seed", "3"])


def test_training_writes_citations_config_and_plot(tmp_path, capsys):
    config = configuration(tmp_path, max_num_epochs=1)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"plot": True, "plot_frequency": 1}
            )
        }
    )
    assert train(config) == 0
    deployed = load_deployed(tmp_path / "mace.json")
    assert deployed.metadata.citations
    assert "Higher Order" in capsys.readouterr().out
    original = (tmp_path / "results/mace.png").read_bytes()
    assert original.startswith(b"\x89PNG\r\n\x1a\n") and len(original) > 5000
    log = tmp_path / "results/mace.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["epoch"] is None
    assert {"epoch", "mode", "head", "loss", "rmse_energy"} <= records[0].keys()
    assert isinstance(records[0]["head"], str)
    assert {row["mode"] for row in records} == {"opt", "train", "valid"}
    replotted = tmp_path / "replotted.png"
    # This run has only epoch zero; the plotting CLI excludes its boundary.
    include_epoch_zero = ["--min_epoch", "-1"]
    renamed = tmp_path / "freely named training log.txt"
    renamed.write_bytes(log.read_bytes())
    for source in (log, log.parent, renamed):
        assert (
            main(
                [
                    "plot",
                    "training",
                    "--path",
                    str(source),
                    "--output",
                    str(replotted),
                    *include_epoch_zero,
                ]
            )
            == 0
        )
        assert replotted.read_bytes() == original
    # A diagnostic baseline must not become an optimizer epoch on the curve.
    without_initial = tmp_path / "epochs.jsonl"
    without_initial.write_text(
        "\n".join(json.dumps(row) for row in records if row["epoch"] is not None)
    )
    assert (
        main(
            [
                "plot",
                "training",
                "--path",
                str(without_initial),
                "--output",
                str(replotted),
                *include_epoch_zero,
            ]
        )
        == 0
    )
    assert replotted.read_bytes() == original
    assert (
        main(
            [
                "plot",
                "training",
                "--path",
                str(log),
                "--output_format",
                "pdf",
                *include_epoch_zero,
            ]
        )
        == 0
    )
    assert log.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        main(
            [
                "plot",
                "training",
                "--path",
                str(empty),
                "--output",
                str(empty / "empty.png"),
            ]
        )
        == 2
    )
    assert "no plotted values" in capsys.readouterr().err
    assert not (empty / "empty.png").exists()
    exported = tmp_path / "resolved.yaml"
    assert (
        main(
            [
                "model",
                "export-config",
                str(tmp_path / "mace.json"),
                "--output",
                str(exported),
            ]
        )
        == 0
    )
    assert (
        ResolvedConfig.load(exported).to_resolved_dict()
        == deployed.config.to_resolved_dict()
    )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_eval_columns_and_calculator_agree(artifact, tmp_path, batch_size):
    frames = [water(False), water(True).repeat((2, 1, 1)), water(False)[:2]]
    source, target = tmp_path / "structures.xyz", tmp_path / "evaluated.xyz"
    write(source, frames)
    assert (
        main(
            [
                "eval",
                "--configs",
                str(source),
                "--model",
                str(artifact),
                "--output",
                str(target),
                "--batch_size",
                str(batch_size),
                "--compute_stress",
                "--return_contributions",
                "--return_descriptors",
                "--return_node_energies",
                "--info_prefix",
                "TEST_",
            ]
        )
        == 0
    )
    evaluated = read(target, ":")
    calculator = MACECalculator(model_paths=artifact)
    for atoms, result in zip(frames, evaluated, strict=True):
        assert result.arrays["TEST_node_energies"].shape == (len(atoms),)
        assert result.arrays["TEST_forces"].shape == (len(atoms), 3)
        assert result.info["TEST_stress"].shape == (3, 3)
        assert not any(key.startswith("MACE_") for key in result.info)
        assert not any(key.startswith("MACE_") for key in result.arrays)
        atoms.calc = calculator
        close(result.info["TEST_energy"], atoms.get_potential_energy())
        close(result.arrays["TEST_forces"], atoms.get_forces())
        close(
            full_3x3_to_voigt_6_stress(result.info["TEST_stress"]), atoms.get_stress()
        )
        close(result.arrays["TEST_node_energies"].sum(), result.info["TEST_energy"])
        close(result.info["TEST_BO_contributions"].sum(), result.info["TEST_energy"])
        close(result.arrays["TEST_descriptors"], calculator.get_descriptors(atoms))


def test_eval_descriptor_options_and_optional_columns(artifact, tmp_path):
    frames = [water(False), water(True).repeat((2, 1, 1)), water(False)[1:]]
    source = tmp_path / "structures.xyz"
    write(source, frames)
    variants = {
        "default": [],
        "nodes": ["--return_descriptors"],
        "mean": ["--return_descriptors", "--descriptor_aggregation_method", "mean"],
        "elements": [
            "--return_descriptors",
            "--descriptor_aggregation_method",
            "per_element_mean",
        ],
        "one": ["--return_descriptors", "--descriptor_num_layers", "1"],
        "equivariant": [
            "--return_descriptors",
            "--descriptor_invariants_only",
            "false",
        ],
    }
    results = {}
    for name, flags in variants.items():
        target = tmp_path / f"{name}.xyz"
        assert (
            main(
                [
                    "eval",
                    "--configs",
                    str(source),
                    "--model",
                    str(artifact),
                    "--output",
                    str(target),
                    "--batch_size",
                    "2",
                    *flags,
                ]
            )
            == 0
        )
        frames_read = read(target, ":")
        assert isinstance(frames_read, list)
        results[name] = frames_read
    for index, atoms in enumerate(frames):
        default = results["default"][index]
        assert "MACE_stress" not in default.info
        assert "MACE_BO_contributions" not in default.info
        assert not {"MACE_descriptors", "MACE_node_energies"} & default.arrays.keys()
        nodes = results["nodes"][index].arrays["MACE_descriptors"]
        mean, elements = results["mean"][index], results["elements"][index]
        assert "MACE_descriptors" not in mean.arrays
        assert "MACE_descriptors" not in elements.arrays
        close(mean.info["MACE_descriptors"], nodes.mean(axis=0))
        expected = np.concatenate(
            [
                nodes[atoms.numbers == number].mean(axis=0)
                if (atoms.numbers == number).any()
                else np.zeros(nodes.shape[1])
                for number in (1, 8)
            ]
        )
        close(elements.info["MACE_descriptors"], expected)
        assert (
            results["one"][index].arrays["MACE_descriptors"].shape[1] < nodes.shape[1]
        )
        assert (
            results["equivariant"][index].arrays["MACE_descriptors"].shape[1]
            > nodes.shape[1]
        )


@pytest.mark.parametrize("requested_dtype", ["float32", "float64"])
def test_eval_from_launcher_converts_precision(artifact, tmp_path, requested_dtype):
    source, target = tmp_path / "source.xyz", tmp_path / "float32.xyz"
    atoms = water(True)
    write(source, [atoms])
    command = cli_command(REPO_ROOT / "mace/cli/eval_configs.py")
    finished = subprocess.run(
        [
            *command,
            "--engine",
            "v1",
            "--model",
            str(artifact),
            "--configs",
            str(source),
            "--output",
            str(target),
            "--default_dtype",
            requested_dtype,
            "--compute_stress",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, finished.stderr
    assert ("does not match model dtype" in finished.stderr) == (
        requested_dtype == "float32"
    )
    result = read(target)
    assert isinstance(result, Atoms)
    for dtype in ("float32", "float64"):
        atoms.calc = MACECalculator(model_paths=artifact, default_dtype=dtype)
        for actual, expected in (
            (result.info["MACE_energy"], atoms.get_potential_energy()),
            (result.arrays["MACE_forces"], atoms.get_forces()),
            (
                full_3x3_to_voigt_6_stress(result.info["MACE_stress"]),
                atoms.get_stress(),
            ),
        ):
            np.testing.assert_allclose(actual, expected, atol=FP32.atol, rtol=FP32.rtol)
        assert atoms.get_forces().dtype == np.dtype(dtype)


def test_eval_refuses_an_unknown_head_and_unsupported_contributions(tmp_path):
    config = configuration(tmp_path, max_num_epochs=1)
    assert config.model.model == "scale_shift"
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    artifact = write_model(tmp_path / "scaled", built.model, built.metadata)
    source, target = tmp_path / "source.xyz", tmp_path / "refused.xyz"
    write(source, [water(False)])
    for flags in (["--head", "absent"], ["--return_contributions"]):
        assert (
            main(
                [
                    "eval",
                    "--model",
                    str(artifact),
                    "--configs",
                    str(source),
                    "--output",
                    str(target),
                    *flags,
                ]
            )
            == 2
        )
        assert not target.exists()


def test_basis_conversion_preserves_function_and_records_gate(artifact, tmp_path):
    target = tmp_path / "reduced.json"
    assert (
        main(
            [
                "model",
                "convert",
                str(artifact),
                "--output",
                str(target),
                "--to-basis",
                "reduced",
            ]
        )
        == 0
    )
    source_calc, target_calc = (
        MACECalculator(model_paths=artifact),
        MACECalculator(model_paths=target),
    )
    for periodic in (False, True):
        atoms = water(periodic)
        atoms.calc = source_calc
        energy, forces, stress = (
            atoms.get_potential_energy(),
            atoms.get_forces(),
            atoms.get_stress(),
        )
        atoms.calc = target_calc
        close(atoms.get_potential_energy(), energy)
        close(atoms.get_forces(), forces)
        close(atoms.get_stress(), stress)
    record = load_deployed(target).metadata.conversion
    assert record is not None
    assert record.projection_verified and record.golden_gate_passed is None
    assert (
        main(
            [
                "model",
                "convert",
                str(target),
                "--output",
                str(tmp_path / "full"),
                "--to-basis",
                "full",
            ]
        )
        == 2
    )


@pytest.mark.parametrize("embedding", [False, True])
def test_multihead_selection_matches_selected_head(tmp_path, embedding, capsys):
    config = configuration(tmp_path, max_num_epochs=1)
    settings = config.data.heads["default"]
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"heads": {"first": settings, "second": settings}}
            ),
            "model": config.model.model_copy(
                update={
                    "graph_features": (
                        GraphFeatureConfig(
                            name="total_charge",
                            kind="categorical",
                            embedding_dim=2,
                            num_classes=2,
                        ),
                    )
                    if embedding
                    else (),
                    "readout": config.model.readout.model_copy(
                        update={"from_embedding": embedding}
                    ),
                }
            ),
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    write_model(tmp_path / "multi", built.model, built.metadata)
    command = ["model", "select-head", "--model", str(tmp_path / "multi.json")]
    assert main([*command, "--list_heads"]) == 0
    assert capsys.readouterr().out.strip() == "Heads: first, second"
    selected = tmp_path / "selected.json"
    assert (
        main(
            [
                "model",
                "select-head",
                "--model",
                str(tmp_path / "multi.json"),
                "--head",
                "second",
                "--output",
                str(selected),
                "--target_device",
                "cpu",
            ]
        )
        == 0
    )
    assert not (tmp_path / "multi-second.safetensors").exists()
    atoms = water(True)
    atoms.calc = MACECalculator(model_paths=tmp_path / "multi.json", head="second")
    expected, forces = atoms.get_potential_energy(), atoms.get_forces()
    atoms.calc = MACECalculator(model_paths=selected)
    close(atoms.get_potential_energy(), expected)
    close(atoms.get_forces(), forces)
    assert load_deployed(selected).heads == ("second",)
    assert next(load_deployed(selected).model.parameters()).device.type == "cpu"
    assert (
        main(["model", "select-head", str(tmp_path / "multi.json"), "--head", "second"])
        == 0
    )
    assert (tmp_path / "multi-second.safetensors").is_file()
    assert load_deployed(tmp_path / "multi-second.json").heads == ("second",)


def test_prepare_roundtrips_the_training_split(tmp_path):
    from mace_core.data.keys import KeySpecification
    from mace_core.data.registry import open_dataset

    configuration(tmp_path)
    destination = tmp_path / "prepared"
    assert (
        main(
            [
                "data",
                "prepare",
                "--train_file",
                str(tmp_path / "train.xyz"),
                "--output",
                str(destination),
                "--num_process",
                "1",
                "--compute_statistics",
                "--shard_size",
                "3",
            ]
        )
        == 0
    )
    train_set = open_dataset(
        str(destination / "train"), key_spec=KeySpecification.from_defaults()
    )
    valid_set = open_dataset(
        str(destination / "valid"), key_spec=KeySpecification.from_defaults()
    )
    assert len(train_set) + len(valid_set) == 15
    statistics = train_set.statistics()
    assert statistics is not None and statistics.atomic_numbers == [1, 8]
    assert train_set[len(train_set) - 1].properties["forces"].shape[1] == 3


def test_migration_config_has_the_same_semantics(tmp_path):
    from mace_torch.cli.migrate import run

    arguments = [
        "--name",
        "migrated",
        "--r_max",
        "4",
        "--hidden_irreps",
        "8x0e+8x1o",
        "--stage_two",
        "--start_stage_two",
        "2",
        "--max_num_epochs",
        "4",
    ]
    destination = tmp_path / "migrated.yaml"
    assert (
        run(
            argparse.Namespace(
                output=destination, command=["mace_run_train", *arguments]
            )
        )
        == 0
    )
    assert ResolvedConfig.load(destination) == parse(arguments)


def test_md_remains_retired_as_the_inventory_agrees(capsys):
    assert main(["md", "active-learning"]) == 2
    assert "ASE dynamics" in capsys.readouterr().err


def test_explicit_flags_keep_native_defaults_and_the_named_head(tmp_path):
    native = ResolvedConfig.from_dict({"data": {"heads": {"water": {}}}})
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump(native.to_resolved_dict()))
    config = parse(["--config", str(path), "--r_max", "5.0", "--train_file", "a.xyz"])
    assert config.model == native.model
    assert config.training == native.training
    assert tuple(config.data.heads) == ("water",)
    assert str(config.data.heads["water"].train_file) == "a.xyz"
    assert parse(["--r_max", "5.0"]).model == ResolvedConfig().model


def test_ambiguous_legacy_model_name_is_refused():
    with pytest.raises(ValueError, match="forced zero shift"):
        parse(["--model=MACE"])


def test_radial_width_override_keeps_the_configured_basis(tmp_path):
    path = tmp_path / "gaussian.yaml"
    path.write_text("model: {radial_basis: {kind: gaussian, num_basis: 8}}")
    config = parse(["--config", str(path), "--num_radial_basis", "6"])
    assert config.model.radial_basis.kind == "gaussian"
    assert config.model.radial_basis.num_basis == 6
    assert parse(["--num_radial_basis", "6"]).model.radial_basis.kind == "bessel"


@pytest.mark.parametrize(
    "preset,model",
    [("dipole", "dipole"), ("dipole_polar", "dielectric")],
)
def test_dipole_presets_train_from_flags_and_reload(tmp_path, preset, model):
    frames = [water(False) for _ in range(6)]
    for index, atoms in enumerate(frames):
        atoms.positions[1, 0] += 0.03 * index
        atoms.info["dipole"] = np.array([0.1 * index, 0.2, -0.1])
        atoms.info["polarizability"] = np.eye(3) * (0.2 + 0.01 * index)
        atoms.arrays["REF_charges"] = np.zeros(len(atoms))
    source = tmp_path / "labels.xyz"
    write(source, frames)
    assert (
        main(
            [
                "train",
                "--model",
                model,
                "--loss",
                preset,
                "--train_file",
                str(source),
                "--work_dir",
                str(tmp_path),
                "--name",
                "dipole",
                "--E0s",
                "isolated",
                "--num_radial_basis",
                "4",
                "--num_channels",
                "2",
                "--max_ell",
                "2",
                "--max_L",
                "2",
                "--correlation",
                "2",
                "--MLP_irreps",
                "4x0e+4x1o+4x2e",
                "--max_num_epochs",
                "1",
                "--batch_size",
                "2",
                "--valid_batch_size",
                "2",
                "--plot",
                "false",
            ]
        )
        == 0
    )
    artifact = tmp_path / "dipole.json"
    loaded = load_deployed(artifact)
    assert loaded.config.loss.kind.kind == "weighted"
    assert loaded.config.loss.weight("dipole") == (100.0 if preset == "dipole" else 1.0)
    calculator = MACECalculator(model_paths=artifact, charges_key="REF_charges")
    result = calculator.get_property("dipole", frames[0])
    assert result.shape == (3,) and np.isfinite(result).all()


def test_prepare_without_statistics_preserves_training_references(tmp_path):
    from mace_core.data import open_dataset
    from mace_core.data.keys import KeySpecification

    config = configuration(tmp_path, max_num_epochs=1)
    destination = tmp_path / "without_stats"
    assert (
        main(
            [
                "data",
                "prepare",
                "--train_file",
                str(config.data.heads["default"].train_file),
                "--output",
                str(destination),
                "--num_process",
                "1",
            ]
        )
        == 0
    )
    dataset = open_dataset(
        str(destination / "train"), key_spec=KeySpecification.from_defaults()
    )
    assert dataset.statistics() is None
    assert not (destination / "statistics.json").exists()
    document = config.to_resolved_dict()
    document["data"]["heads"]["default"]["train_file"] = str(destination / "train")
    document["data"]["heads"]["default"]["valid_file"] = str(destination / "valid")
    prepared = run_data_stage(ResolvedConfig.from_dict(document), CATALOGUE)
    original = run_data_stage(config, CATALOGUE)
    assert prepared.e0s == original.e0s


def test_migrated_selection_and_pseudolabel_commands_use_shared_stages(
    artifact, tmp_path
):
    from mace_torch.cli.migrate import run
    from mace_torch.finetune.stages import select_structures

    source = load_deployed(artifact)
    config_file = tmp_path / "selection.yaml"
    selected = tmp_path / "selected.xyz"
    assert (
        run(
            argparse.Namespace(
                output=config_file,
                command=[
                    "mace_finetuning_select",
                    "--configs_pt",
                    str(source.config.data.heads["default"].train_file),
                    "--model",
                    str(artifact),
                    "--subselect",
                    "random",
                    "--num_samples",
                    "3",
                    "--output",
                    str(selected),
                ],
            )
        )
        == 0
    )
    config = ResolvedConfig.load(config_file)
    assert (
        main(
            [
                "data",
                "select",
                "--config",
                str(config_file),
                "--head",
                "selected",
                "--output",
                str(selected),
            ]
        )
        == 0
    )
    expected_file = select_structures(config, "selected", tmp_path / "expected.xyz")
    assert selected.read_bytes() == expected_file.read_bytes()
    labels = tmp_path / "labels"
    assert (
        main(
            [
                "data",
                "pseudolabel",
                "--config",
                str(config_file),
                "--head",
                "selected",
                "--output",
                str(labels),
            ]
        )
        == 0
    )
    from mace_torch.finetune.pseudolabels import load_pseudolabel_artifact

    configurations, provenance = load_pseudolabel_artifact(labels / "selected")
    assert configurations and provenance is not None
    assert all(
        "energy" in item.properties and "forces" in item.properties
        for item in configurations
        if item.config_type != "IsolatedAtom"
    )


def test_head_and_reporting_flags_preserve_structured_context(tmp_path):
    from pathlib import Path

    config = parse(
        ["--heads", "{'water': {'train_file': 'water.xyz', 'E0s': 'average'}}"]
    )
    assert tuple(config.data.heads) == ("water",)
    assert config.data.heads["water"].train_file == Path("water.xyz")
    assert config.data.heads["water"].e0s.kind == "average"
    document = config.to_resolved_dict()
    document["runtime"]["wandb"] = {"enabled": True, "project": "kept"}
    path = tmp_path / "reporting.yaml"
    path.write_text(yaml.safe_dump(document))
    overridden = parse(
        ["--config", str(path), "--wandb_name", "new", "--E0s", "average"]
    )
    assert overridden.runtime.wandb.enabled
    assert overridden.runtime.wandb.project == "kept"
    assert overridden.runtime.wandb.name == "new"
    assert tuple(overridden.data.heads) == ("water",)


def test_replay_flags_create_an_ordinary_head_and_require_context():
    config = parse(
        [
            "--foundation_model",
            "foundation.json",
            "--train_file",
            "fit.xyz",
            "--multiheads_finetuning",
            "true",
            "--pt_train_file",
            "replay.xyz",
            "--subselect_pt",
            "random",
            "--num_samples_pt",
            "10",
        ]
    )
    selection = config.data.heads["pt_head"].subselect
    assert selection is not None and selection.num_samples == 10
    with pytest.raises(ValueError, match="replay flags require"):
        parse(["--num_samples_pt", "10"])


def test_stage_api_records_and_restores_its_precision(tmp_path):
    import torch
    from mace_core.config.precision import PrecisionConfig

    config = configuration(tmp_path, max_num_epochs=1)
    config = config.model_copy(update={"precision": PrecisionConfig(model="float32")})
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    assert next(built.model.parameters()).dtype == torch.float32
    path = write_model(tmp_path / "single_precision", built.model, built.metadata)
    assert next(load_deployed(path).model.parameters()).dtype == torch.float32
    explicit = run_model_stage(
        config, data, CATALOGUE, precision=PrecisionConfig(model="float64")
    )
    assert explicit.metadata.config.resolved["precision"]["model"] == "float64"
