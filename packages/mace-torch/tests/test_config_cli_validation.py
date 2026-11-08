"""File/CLI precedence and the configuration-only command."""

import argparse

import pytest
import yaml
from conftest import fp64_only
from mace_core.cli import main
from mace_core.cli.registry import DiscoveredCommand
from mace_core.config.loss import HuberLoss
from mace_core.config.resolved import ResolvedConfig
from mace_torch.cli.commands import VALIDATE_CONFIG
from mace_torch.cli.legacy_flags import add_arguments
from mace_torch.cli.run_train import parse


@fp64_only
def test_delta_override_keeps_the_files_loss_semantics(tmp_path):
    choice = {"kind": "huber", "delta": 0.02}
    loss = choice
    source = tmp_path / "run.yaml"
    source.write_text(yaml.safe_dump({"loss": {**loss, "weights": {"forces": 3}}}))
    config = parse(["--config", str(source), "--huber_delta", ".03"])
    assert isinstance(config.loss, HuberLoss)
    assert config.loss.delta == 0.03
    assert config.loss.weights == {"forces": 3}
    assert config.loss.use_structure_weights == {}
    assert config.model.observables == ("energy", "forces")


@fp64_only
def test_changing_loss_drops_old_parameters_and_preserves_weights(tmp_path):
    source = tmp_path / "run.yaml"
    source.write_text("loss:\n  kind: huber\n  delta: 0.02\n  weights: {forces: 3}\n")
    config = parse(["--config", str(source), "--loss", "weighted"])
    assert config.loss.kind == "weighted"
    assert "delta" not in config.loss.model_dump()
    assert config.loss.weights == {"forces": 3}


@fp64_only
def test_delta_without_compatible_loss_is_refused():
    with pytest.raises(ValueError, match="requires a huber or universal"):
        parse(["--huber_delta", ".03"])


@fp64_only
def test_validation_command_does_not_touch_the_dataset(tmp_path, monkeypatch, capsys):
    import mace_core.cli as cli

    monkeypatch.setattr(
        cli,
        "discover",
        lambda: {
            ("config", "validate"): DiscoveredCommand(
                ("config", "validate"), "mace-torch", VALIDATE_CONFIG
            ),
        },
    )
    source = tmp_path / "run.yaml"
    source.write_text(
        "data:\n  heads:\n    pbe: {train_file: missing.xyz}\n"
        "loss: {kind: huber, delta: 0.02}\n"
    )
    assert (
        main(["config", "validate", "--config", str(source), "--show", "minimal"]) == 0
    )
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["loss"] == {"kind": "huber", "delta": 0.02}
    loaded = ResolvedConfig.from_dict(printed)
    assert isinstance(loaded.loss, HuberLoss)
    assert loaded.loss.delta == 0.02
    assert not (tmp_path / "missing.xyz").exists()
    source.write_text("data: train.xyz\n")
    assert main(["config", "validate", "--config", str(source)]) == 2
    error = capsys.readouterr().err
    assert "data must be a configuration section" in error
    assert "Traceback" not in error


@fp64_only
def test_every_training_option_explains_its_purpose():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    assert all(action.help for action in parser._actions)
    help_text = parser.format_help()
    assert "Angstrom" in help_text
    assert "Base learning rate" in help_text


@fp64_only
@pytest.mark.parametrize(
    "document,location",
    [
        ({"training": {"batch_size": 0}}, "training.batch_size"),
        ({"model": {"observables": ["forcces"]}}, "model.observables"),
        ({"model": {"model": "scale_shifft"}}, "model.model"),
        ({"data": "train.xyz"}, "data"),
    ],
)
def test_standalone_training_reports_invalid_config_without_traceback(
    tmp_path, capsys, document, location
):
    from mace_torch.cli.run_train import main as train_main

    source = tmp_path / "invalid.yaml"
    source.write_text(yaml.safe_dump(document))
    assert train_main(["--config", str(source)]) == 2
    message = capsys.readouterr().err
    assert location in message
    assert "Traceback" not in message
    assert "input_value" not in message
    assert "errors.pydantic.dev" not in message


@fp64_only
def test_cli_refuses_weights_for_unrequested_outputs():
    with pytest.raises(ValueError, match="polarizability"):
        parse(["--model", "dipole", "--loss", "dipole", "--polarizability_weight", "3"])
