"""The ``mace`` program: commands from entry points, one file and explicit flags.

The commands here are declared by stand-in distributions, which is exactly how
the torch and jax packages declare theirs: nothing in ``mace_core.cli`` names a
package, so a stand-in exercises the same path.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from mace_core import cli
from mace_core.cli import Command, ConfigFlag, discover, main, set_value
from mace_core.cli import registry as registry_module
from mace_core.config import ConfigError, ConfigSection, ReforgeBaseConfig


class Runtime(ConfigSection):
    name: str = "mace"
    seed: int = 0


class Training(ConfigSection):
    lr: float = 0.01
    batch_size: int = 16


class Schema(ReforgeBaseConfig):
    runtime: Runtime = Runtime()
    training: Training = Training()


RAN: list[Any] = []


def _record(arguments) -> int:
    RAN.append(arguments)
    return 0


FIT = Command(
    help="Fit something.",
    run=_record,
    schema=Schema,
    flags=(
        ConfigFlag("--seed", "runtime.seed", "Random seed.", type=int),
        ConfigFlag("--lr", "training.lr", "Learning rate.", type=float),
    ),
)

SHOW = Command(
    help="Show something.",
    run=_record,
    arguments=lambda parser: parser.add_argument("target"),
)


@dataclass
class FakeEntryPoint:
    name: str
    loaded: Any
    distribution: str

    @property
    def value(self) -> str:
        return f"{self.distribution}.commands:{self.name}"

    @property
    def dist(self):
        return SimpleNamespace(name=self.distribution)

    def load(self):
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


@pytest.fixture
def installed(monkeypatch):
    """Replace what is installed with the entry points a test lists."""
    RAN.clear()

    def install(*entries: FakeEntryPoint) -> None:
        monkeypatch.setattr(
            registry_module,
            "entry_points",
            lambda group: list(entries) if group == cli.ENTRY_POINT_GROUP else [],
        )

    return install


def test_commands_nest_by_the_words_of_their_names(installed, capsys):
    installed(
        FakeEntryPoint("fit", FIT, "engine-a"),
        FakeEntryPoint("model.show", SHOW, "engine-b"),
    )
    assert main(["model", "show", "x.model"]) == 0
    assert RAN[-1].target == "x.model"
    with pytest.raises(SystemExit):
        main(["--help"])
    listing = capsys.readouterr().out
    assert "fit" in listing and "Fit something." in listing and "model" in listing


def test_a_group_or_no_command_prints_help_and_fails(installed, capsys):
    installed(FakeEntryPoint("model.show", SHOW, "engine-b"))
    assert main([]) == 2
    assert main(["model"]) == 2
    assert "show" in capsys.readouterr().err


def test_the_file_is_read_and_a_flag_writes_one_value_into_it(installed, tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("training: {lr: 0.5, batch_size: 4}\nruntime: {name: run}\n")
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    assert main(["fit", "--config", str(path), "--lr", "0.25"]) == 0
    configuration = RAN[-1].configuration
    assert isinstance(configuration, Schema)
    assert configuration.training.lr == 0.25
    assert configuration.training.batch_size == 4
    assert configuration.runtime.name == "run"
    assert configuration.runtime.seed == 0
    assert RAN[-1].config == path


def test_without_a_file_the_flags_set_values_over_the_defaults(installed):
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    assert main(["fit", "--seed", "7"]) == 0
    assert RAN[-1].configuration.runtime.seed == 7
    assert RAN[-1].configuration.training == Training()


def test_there_is_no_general_override_syntax(installed, capsys):
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    with pytest.raises(SystemExit) as caught:
        main(["fit", "--training.batch_size", "8"])
    assert caught.value.code == 2
    assert "--training.batch_size" in capsys.readouterr().err
    assert not RAN


def test_an_unknown_key_in_the_file_is_reported_without_a_traceback(
    installed, tmp_path, capsys
):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"training": {"learning_rate": 0.1}}))
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    assert main(["fit", "--config", str(path)]) == 2
    error = capsys.readouterr().err
    assert error.startswith("mace fit:")
    assert "training.learning_rate" in error
    assert not RAN


def test_a_file_that_does_not_parse_is_reported(installed, tmp_path, capsys):
    path = tmp_path / "run.toml"
    path.write_text("training = [\n")
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    assert main(["fit", "--config", str(path)]) == 2
    assert "cannot parse config file" in capsys.readouterr().err


def test_a_flag_value_of_the_wrong_type_is_argparse_s_error(installed):
    installed(FakeEntryPoint("fit", FIT, "engine-a"))
    with pytest.raises(SystemExit) as caught:
        main(["fit", "--seed", "seven"])
    assert caught.value.code == 2


def test_two_packages_declaring_one_command_are_refused(installed, capsys):
    installed(
        FakeEntryPoint("fit", FIT, "engine-a"),
        FakeEntryPoint("fit", FIT, "engine-b"),
    )
    assert main(["fit"]) == 2
    error = capsys.readouterr().err
    assert "engine-a" in error and "engine-b" in error


def test_a_command_that_is_also_a_group_is_refused(installed):
    installed(
        FakeEntryPoint("model", SHOW, "engine-a"),
        FakeEntryPoint("model.show", SHOW, "engine-b"),
    )
    with pytest.raises(cli.CommandConflictError, match="group of commands"):
        discover()


def test_a_command_that_does_not_import_is_listed_and_says_why(installed, capsys):
    installed(
        FakeEntryPoint("fit", ImportError("no module named torch"), "engine-a"),
        FakeEntryPoint("show", "not a command", "engine-b"),
    )
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "not available" in capsys.readouterr().out
    assert main(["fit"]) == 2
    error = capsys.readouterr().err
    assert "engine-a" in error and "no module named torch" in error
    assert main(["show"]) == 2
    assert "not a Command" in capsys.readouterr().err


def test_set_value_creates_sections_and_leaves_neighbours():
    document: dict[str, Any] = {"training": {"lr": 0.1, "batch_size": 4}}
    set_value(document, "training.lr", 0.2)
    set_value(document, "runtime.name", "run")
    assert document == {
        "training": {"lr": 0.2, "batch_size": 4},
        "runtime": {"name": "run"},
    }


def test_set_value_refuses_to_write_through_a_value():
    with pytest.raises(ConfigError, match="training holds float"):
        set_value({"training": 0.1}, "training.lr", 0.2)


def test_a_flag_can_place_its_value_itself(installed):
    def into_runtime_name(document, value):
        set_value(document, "runtime.name", value.upper())

    placed = Command(
        help="Fit.",
        run=_record,
        schema=Schema,
        flags=(ConfigFlag("--tag", "runtime.name", "Tag.", write=into_runtime_name),),
    )
    installed(FakeEntryPoint("fit", placed, "engine-a"))
    assert main(["fit", "--tag", "run"]) == 0
    assert RAN[-1].configuration.runtime.name == "RUN"


PROBE = """
import json, sys
from mace_core.cli import main
try:
    main(["--help"])
except SystemExit:
    pass
print(json.dumps(sorted({m.split(".", 1)[0] for m in sys.modules})))
"""


def test_the_help_imports_no_framework():
    """In a fresh interpreter, with whatever is installed declaring commands.
    A command's module is imported to list it, so it must defer its framework
    to when it runs."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    )
    reached = set(json.loads(result.stdout.splitlines()[-1]))
    assert not reached & {"torch", "jax", "jaxlib", "e3nn", "mace"}
