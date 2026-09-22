"""The training configuration: its sections and the rules between them.

The sections are only fields grouped by subject, and testing that a field holds
what was put in it says nothing. What is worth pinning is the part legacy does
not have: that an incoherent configuration is refused before anything is built,
and that the answer stops changing once it has been given.
"""

import json
from pathlib import Path

import pytest
import yaml
from mace_core.config.base import ConfigError, ConfigSection
from mace_core.config.e0s import E0sFromFoundation, E0sIsolatedAtoms
from mace_core.config.resolved import ENERGYLESS_MODELS, ResolvedConfig
from mace_core.config.runtime import WORK_DIR_LAYOUT
from mace_core.config.section import FrozenSection
from mace_core.config.training import (
    ConstantSchedule,
    LBFGSOptimizer,
    ScheduleFreeOptimizer,
)
from pydantic import BaseModel, ValidationError

SECTIONS = ("runtime", "data", "model", "loss", "training", "finetune")


def write(tmp_path: Path, suffix: str, document: dict) -> Path:
    path = tmp_path / f"config{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(document))
    else:
        path.write_text(yaml.safe_dump(document))
    return path


# ---------------------------------------------------------------------------
# The shape
# ---------------------------------------------------------------------------


def test_the_six_sections_are_the_declared_ones():
    assert tuple(ResolvedConfig.model_fields) == SECTIONS


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_a_config_loads_from_every_format(tmp_path, suffix):
    path = write(tmp_path, suffix, {"model": {"r_max": 4.5}})
    assert ResolvedConfig.load(path).model.r_max == 4.5


def test_an_override_reaches_every_section(tmp_path):
    config = ResolvedConfig.load(
        cli_overrides=[
            "--runtime.name=run",
            "--data.valid_fraction=0.2",
            "--model.r_max=3.0",
            "--loss.weights",
            '{"energy": 5.0}',
            "--training.lr=0.5",
            "--finetune.foundation_model=medium",
        ]
    )
    assert config.runtime.name == "run"
    assert config.data.valid_fraction == 0.2
    assert config.model.r_max == 3.0
    assert config.loss.weights == {"energy": 5.0}
    assert config.training.lr == 0.5
    assert config.finetune.foundation_model == "medium"


def test_precedence_runs_defaults_then_file_then_override(tmp_path):
    path = write(tmp_path, ".yaml", {"training": {"lr": 0.02, "batch_size": 32}})
    default = ResolvedConfig()
    from_file = ResolvedConfig.load(path)
    overridden = ResolvedConfig.load(path, ["--training.lr=0.5"])

    assert default.training.lr != 0.02
    assert from_file.training.lr == 0.02
    assert overridden.training.lr == 0.5
    # The file's other setting survives the override, rather than the override
    # replacing the section.
    assert overridden.training.batch_size == 32


def test_an_unknown_key_names_it_and_its_nearest_neighbour(tmp_path):
    path = write(tmp_path, ".yaml", {"training": {"learning_rate": 0.1}})
    with pytest.raises(ConfigError, match=r"training\.learning_rate"):
        ResolvedConfig.load(path)


def test_resolving_twice_is_a_fixed_point(tmp_path):
    """The property the whole schema has to keep, not just a section of it."""
    written = {
        "model": {"r_max": 4.0, "observables": ["energy", "forces", "stress"]},
        "data": {"heads": {"pbe": {"e0s": {"table": {"values": {1: -13.6}}}}}},
        "training": {"optimizer": {"schedulefree": {"warmup_steps": 100}}},
        "loss": {"kind": {"huber": {"delta": 0.05}}},
    }
    once = ResolvedConfig.load(write(tmp_path, ".yaml", written)).to_resolved_dict()
    twice = ResolvedConfig.model_validate(once).to_resolved_dict()
    assert once == twice


def _nested_models(model: type[BaseModel], seen=None):
    seen = seen if seen is not None else set()
    for field in model.model_fields.values():
        for candidate in (field.annotation, *getattr(field.annotation, "__args__", ())):
            known = isinstance(candidate, type) and issubclass(candidate, BaseModel)
            if known and candidate not in seen:
                seen.add(candidate)
                _nested_models(candidate, seen)
    return seen


def test_every_section_in_the_tree_is_frozen():
    """`frozen` on the root freezes the root's own fields and nothing deeper.

    So a section that inherits the plain base is mutable, and there is no way
    to see that from the root. Walked rather than listed: a section added later
    fails here instead of being the one writable corner of a resolved config.
    """
    mutable = sorted(
        model.__name__
        for model in _nested_models(ResolvedConfig)
        if issubclass(model, ConfigSection) and not model.model_config.get("frozen")
    )
    assert not mutable, mutable
    assert FrozenSection.model_config["frozen"] is True


def test_a_resolved_config_cannot_be_written_to_at_any_depth():
    config = ResolvedConfig()
    for target, name in [
        (config, "training"),
        (config.training, "lr"),
        (config.training.ema, "decay"),
        (config.training.optimizer, "beta"),
    ]:
        with pytest.raises(ValidationError):
            setattr(target, name, 1)


# ---------------------------------------------------------------------------
# What the sections own
# ---------------------------------------------------------------------------


def test_one_work_dir_replaces_six_directory_flags():
    config = ResolvedConfig.load(cli_overrides=["--runtime.work_dir=/runs/a"])
    assert set(WORK_DIR_LAYOUT) == {
        "logs",
        "models",
        "checkpoints",
        "results",
        "downloads",
    }
    for which in WORK_DIR_LAYOUT:
        assert config.runtime.directory(which).parent == Path("/runs/a")


def test_asking_for_a_directory_that_is_not_in_the_layout_names_the_ones_that_are():
    with pytest.raises(KeyError, match="run layout"):
        ResolvedConfig().runtime.directory("plots")


def test_a_declared_observable_with_no_weight_takes_one():
    """A declared property contributing nothing is asked for, never inherited."""
    config = ResolvedConfig.load(cli_overrides=["--loss.weights", '{"forces": 100.0}'])
    assert config.loss.weight("forces") == 100.0
    assert config.loss.weight("energy") == 1.0


def test_the_second_stage_only_overrides_the_weights_it_names():
    config = ResolvedConfig.load(
        cli_overrides=[
            "--loss.weights",
            '{"energy": 1.0, "forces": 100.0}',
            "--loss.stage_two_weights",
            '{"energy": 1000.0}',
        ]
    )
    assert config.loss.weight("energy", stage_two=True) == 1000.0
    assert config.loss.weight("forces", stage_two=True) == 100.0


def test_the_observable_list_is_not_inferred_from_the_model_name():
    """Legacy reads the class name and sets six booleans from it."""
    config = ResolvedConfig.load(
        cli_overrides=[
            "--model.model=AtomicDipolesMACE",
            "--model.observables",
            '["dipole"]',
        ]
    )
    assert config.model.model == "AtomicDipolesMACE"
    assert config.model.observables == ("dipole",)


def test_the_clebsch_gordan_basis_is_recorded_rather_than_inferred():
    """On the frozen tree it depends on what happens to be installed."""
    assert ResolvedConfig().model.clebsch_gordan_basis == "reduced"
    assert "clebsch_gordan_basis" in ResolvedConfig().to_resolved_dict()["model"]


def test_the_schedulefree_tuning_lives_with_its_optimizer():
    """Legacy has the three as top-level flags meaning nothing under Adam."""
    config = ResolvedConfig.load(
        cli_overrides=["--training.optimizer", '{"schedulefree": {"beta1": 0.8}}']
    )
    optimizer = config.training.optimizer
    assert isinstance(optimizer, ScheduleFreeOptimizer)
    assert optimizer.beta1 == 0.8
    with pytest.raises(ConfigError, match="beta1"):
        ResolvedConfig.load(
            cli_overrides=["--training.optimizer", '{"adam": {"beta1": 0.8}}']
        )


# ---------------------------------------------------------------------------
# The rules between sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["foundation", "estimated"])
def test_e0s_read_from_a_foundation_model_need_one_configured(kind):
    """Legacy asserts this for one of the two kinds and not the other."""
    heads = json.dumps({"pbe": {"e0s": {kind: {}}}})
    with pytest.raises(ValidationError, match="foundation_model is not set"):
        ResolvedConfig.load(cli_overrides=["--data.heads", heads])

    allowed = ResolvedConfig.load(
        cli_overrides=["--data.heads", heads, "--finetune.foundation_model=medium"]
    )
    assert allowed.data.heads["pbe"].e0s.kind == kind


@pytest.mark.parametrize("model", sorted(ENERGYLESS_MODELS))
def test_a_model_with_no_atomic_energies_refuses_e0s(model):
    """Legacy warns and rewrites the request to `average`."""
    with pytest.raises(ValidationError, match=r"no .*atomic-energy term"):
        ResolvedConfig.load(
            cli_overrides=[
                f"--model.model={model}",
                "--data.heads",
                json.dumps({"pbe": {"e0s": {"average": {}}}}),
            ]
        )


def test_a_model_with_no_atomic_energies_is_fine_with_no_e0s_at_all():
    """The rule is about asking for them, not about the default being there."""
    config = ResolvedConfig.load(
        cli_overrides=[
            "--model.model=AtomicDipolesMACE",
            "--data.heads",
            json.dumps({"pbe": {"train_file": "train.xyz"}}),
        ]
    )
    assert config.data.heads["pbe"].e0s == E0sIsolatedAtoms()


def test_pseudolabels_cannot_be_generated_and_read_at_once():
    with pytest.raises(ValidationError, match="Set one"):
        ResolvedConfig.load(
            cli_overrides=[
                "--finetune.pseudolabels.enabled=true",
                "--finetune.pseudolabels.labels_from=/runs/a/labels.xyz",
            ]
        )


def test_lbfgs_refuses_an_ema():
    """Legacy builds the EMA around the optimizer it then replaces."""
    with pytest.raises(ValidationError, match="once per epoch"):
        ResolvedConfig.load(
            cli_overrides=[
                "--training.optimizer",
                '{"lbfgs": {}}',
                "--training.scheduler.kind",
                '{"constant": {}}',
                "--training.ema.enabled=true",
            ]
        )


def test_lbfgs_refuses_a_plateau_schedule():
    with pytest.raises(ValidationError, match="plateau"):
        ResolvedConfig.load(cli_overrides=["--training.optimizer", '{"lbfgs": {}}'])


def test_lbfgs_is_accepted_with_a_schedule_that_does_not_watch_steps():
    config = ResolvedConfig.load(
        cli_overrides=[
            "--training.optimizer",
            '{"lbfgs": {}}',
            "--training.scheduler.kind",
            '{"constant": {}}',
        ]
    )
    assert isinstance(config.training.optimizer, LBFGSOptimizer)
    assert isinstance(config.training.scheduler.kind, ConstantSchedule)


def test_a_second_stage_running_lbfgs_is_checked_too():
    """The rule is about a stage, so it has to look at both of them."""
    with pytest.raises(ValidationError, match=r"stage_two\.optimizer runs lbfgs"):
        ResolvedConfig.load(
            cli_overrides=[
                "--training.ema.enabled=true",
                "--training.scheduler.kind",
                '{"constant": {}}',
                "--training.stage_two",
                '{"enabled": true, "start_epoch": 100, "optimizer": {"lbfgs": {}}}',
            ]
        )


def test_a_second_stage_that_never_starts_is_refused():
    with pytest.raises(ValidationError, match="start"):
        ResolvedConfig.load(cli_overrides=["--training.stage_two.enabled=true"])


def test_a_second_stage_inherits_its_optimizer_by_naming_it():
    """`None` cannot say whether it means none of them or not written."""
    config = ResolvedConfig.load(
        cli_overrides=["--training.stage_two", '{"enabled": true, "start_epoch": 100}']
    )
    assert config.training.stage_two.optimizer.kind == "inherit"


def test_a_coherent_finetuning_configuration_is_accepted():
    """The guard against a suite that only ever asserts refusals."""
    config = ResolvedConfig.load(
        cli_overrides=[
            "--finetune.foundation_model=medium",
            "--data.heads",
            json.dumps({"pbe": {"e0s": {"foundation": {"head": "mp"}}}}),
            "--training.stage_two",
            '{"enabled": true, "start_epoch": 1200}',
        ]
    )
    assert config.data.heads["pbe"].e0s == E0sFromFoundation(head="mp")
    assert config.training.stage_two.start_epoch == 1200
