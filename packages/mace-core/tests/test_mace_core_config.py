"""`ReforgeBaseConfig`: file formats, `from_dict`, unknown keys, and the two
exports' fixed point."""

import inspect
import json
import math
import re
import subprocess
import sys
from collections.abc import Set as AbstractSet
from typing import Annotated, Any

import pytest
import yaml
from mace_core.config import (
    ConfigError,
    ConfigSection,
    ReforgeBaseConfig,
    read_config_file,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

#: A warning the test did not ask for is a failure.
pytestmark = pytest.mark.filterwarnings("error")

# ---------------------------------------------------------------------------
# The demo schema: two levels of nesting, a list, an optional, a Literal.


class RadialSection(ConfigSection):
    num_bessel: int = 8
    cutoff: float = 5.0


class ModelSection(ConfigSection):
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"
    radial: RadialSection = RadialSection()


class DataSection(ConfigSection):
    train_file: str | None = None
    valid_fraction: float = 0.1
    energy_key: str = "REF_energy"
    heads: list[str] = Field(default_factory=lambda: ["default"])


class StageTwoSection(ConfigSection):
    start_epoch: int = 100
    energy_weight: float = 1000.0


class DemoConfig(ReforgeBaseConfig):
    name: str = "mace"
    seed: int = 123
    default_dtype: str = "float64"
    model: ModelSection = ModelSection()
    data: DataSection = DataSection()
    #: A section left at its defaults unless a file or the CLI writes into it.
    stage_two: StageTwoSection = StageTwoSection()


#: One config, as a dict. Each format test writes it out and loads it back.
FILE_VALUES = {
    "name": "water",
    "seed": 7,
    "model": {"num_interactions": 4, "radial": {"cutoff": 4.5}},
    "data": {"train_file": "train.xyz", "heads": ["pbe", "r2scan"]},
}


def to_toml(values, prefix=""):
    """Enough TOML for a None-free config: scalars and lists share JSON's
    literal syntax, nested dicts become `[a.b]` tables after the scalars."""
    lines = [
        f"{k} = {json.dumps(v)}" for k, v in values.items() if not isinstance(v, dict)
    ]
    for key, value in values.items():
        if isinstance(value, dict):
            lines += [f"\n[{prefix}{key}]", to_toml(value, f"{prefix}{key}.")]
    return "\n".join(lines)


def dump(values, extension):
    if extension == ".toml":
        return to_toml(values)
    if extension == ".json":
        return json.dumps(values)
    return yaml.safe_dump(values)


def write_config(tmp_path, extension, values=FILE_VALUES, name="config"):
    path = tmp_path / f"{name}{extension}"
    path.write_text(dump(values, extension), encoding="utf-8")
    return path


def error_locations(excinfo):
    """The dotted location of every error a `ValidationError` carries."""
    return [".".join(map(str, error["loc"])) for error in excinfo.value.errors()]


# ---------------------------------------------------------------------------
# File loading


@pytest.mark.parametrize("extension", [".toml", ".yaml", ".yml", ".json"])
def test_same_config_loads_identically_from_every_format(tmp_path, extension):
    config = DemoConfig.load(write_config(tmp_path, extension))
    assert config == DemoConfig.model_validate(FILE_VALUES)
    # The file set two fields at depth two; the sibling kept its default.
    assert config.model.radial.cutoff == 4.5
    assert config.model.radial.num_bessel == 8


def test_read_config_file_returns_the_parsed_dict(tmp_path):
    assert read_config_file(write_config(tmp_path, ".toml")) == FILE_VALUES


def test_extension_is_matched_in_any_case(tmp_path):
    path = tmp_path / "CONFIG.YAML"
    path.write_text("seed: 5\n", encoding="utf-8")
    assert DemoConfig.load(path).seed == 5


def test_unknown_extension_is_an_error(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("seed = 1", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown extension '\.ini'"):
        DemoConfig.load(path)


def test_empty_file_is_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert DemoConfig.load(path) == DemoConfig()


def test_comment_only_file_is_all_defaults(tmp_path):
    path = tmp_path / "comments.yaml"
    path.write_text("# nothing set yet\n", encoding="utf-8")
    assert DemoConfig.load(path) == DemoConfig()


def test_file_must_be_a_mapping_at_the_top(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping of keys to values at the top"):
        DemoConfig.load(path)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"cannot read config file .*nope\.yaml"):
        DemoConfig.load(tmp_path / "nope.yaml")


def test_unreadable_file_is_a_config_error(tmp_path):
    path = tmp_path / "latin.yaml"
    path.write_bytes(b"name: caf\xe9\n")
    with pytest.raises(ConfigError, match=r"cannot read config file .*latin\.yaml"):
        DemoConfig.load(path)


@pytest.mark.parametrize(
    ("extension", "text"),
    [(".toml", "seed = \n"), (".yaml", "seed: [1\n"), (".json", "{")],
)
def test_malformed_file_is_a_config_error(tmp_path, extension, text):
    path = tmp_path / f"broken{extension}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=r"cannot parse config file .*broken"):
        DemoConfig.load(path)


def test_a_yaml_anchor_that_contains_itself_is_a_config_error(tmp_path):
    # Under a section it would be pydantic's error; under a free dict it would
    # load and then fail to export, so the file is refused up front.
    class Free(ReforgeBaseConfig):
        extra: dict[str, Any] = Field(default_factory=dict)

    path = tmp_path / "loop.yaml"
    path.write_text("extra: &loop {b: *loop}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"loop\.yaml contains a value that refers"):
        Free.load(path)
    path.write_text("extra: {a: &shared {x: 1}, b: *shared}\n", encoding="utf-8")
    assert Free.load(path).extra == {"a": {"x": 1}, "b": {"x": 1}}  # sharing is fine


# ---------------------------------------------------------------------------
# Precedence: defaults < the file. Nothing else feeds a config.


def test_file_overrides_defaults(tmp_path):
    config = DemoConfig.load(write_config(tmp_path, ".yaml"))
    assert config.model.num_interactions == 4  # from the file
    assert config.default_dtype == "float64"  # untouched default


def test_values_are_handed_to_pydantic_as_given(tmp_path):
    values = {"seed": "9", "data": {"train_file": None, "heads": ["a", "b"]}}
    config = DemoConfig.load(write_config(tmp_path, ".json", values))
    assert config.seed == 9  # pydantic's lax coercion, not the loader's
    assert config.data.train_file is None
    assert config.data.heads == ["a", "b"]


def test_value_of_the_wrong_type_is_a_validation_error(tmp_path):
    with pytest.raises(ValidationError, match="seed"):
        DemoConfig.load(write_config(tmp_path, ".yaml", {"seed": "seven"}))


# ---------------------------------------------------------------------------
# `from_dict` is `load` for a caller that edits the parsed file first (a
# command line writing its flags); it validates the same way.


def test_from_dict_validates_a_parsed_document(tmp_path):
    document = read_config_file(write_config(tmp_path, ".toml"))
    document["seed"] = 9
    document.setdefault("stage_two", {})["start_epoch"] = 50
    config = DemoConfig.from_dict(document)
    assert config.seed == 9
    assert config.model.num_interactions == 4  # the file's values survive
    assert config.stage_two == StageTwoSection(start_epoch=50)
    assert DemoConfig.load(write_config(tmp_path, ".toml")) == DemoConfig.from_dict(
        read_config_file(write_config(tmp_path, ".toml"))
    )


def test_from_dict_does_not_write_into_the_document():
    document = {"model": {"radial": {"cutoff": 6.0}}}
    DemoConfig.from_dict(document)
    assert document == {"model": {"radial": {"cutoff": 6.0}}}


# ---------------------------------------------------------------------------
# Unknown keys are pydantic's error, at their dotted location; every error of
# one load is reported together.


def test_unknown_top_level_key_in_file(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("sead: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        DemoConfig.load(path)
    assert error_locations(excinfo) == ["sead"]
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_unknown_nested_key_in_file_names_the_dotted_path(tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"model": {"radial": {"cutof": 4.0}}}), encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        DemoConfig.load(path)
    assert error_locations(excinfo) == ["model.radial.cutof"]
    assert "model.radial.cutof" in str(excinfo.value)


def test_every_error_is_reported_at_once(tmp_path):
    path = tmp_path / "typos.yaml"
    path.write_text(
        "sead: 1\nseed: seven\nmodel:\n  num_interaction: 3\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError) as excinfo:
        DemoConfig.load(path)
    assert set(error_locations(excinfo)) == {"sead", "seed", "model.num_interaction"}


def test_unknown_key_in_a_document_is_reported_at_its_path():
    documents = {
        "nmae": {"nmae": 1},
        "model.num_interaction": {"model": {"num_interaction": 1}},
        "stage_two.start": {"stage_two": {"start": 1}},
    }
    for dotted_path, document in documents.items():
        with pytest.raises(ValidationError) as excinfo:
            DemoConfig.from_dict(document)
        assert error_locations(excinfo) == [dotted_path]


def test_every_bad_list_item_is_reported(tmp_path):
    class Layers(ReforgeBaseConfig):
        layers: list[RadialSection] = Field(default_factory=list)

    path = write_config(tmp_path, ".json", {"layers": [{"cutof": 1}, {"nb": 2}]})
    with pytest.raises(ValidationError) as excinfo:
        Layers.load(path)
    assert error_locations(excinfo) == ["layers.0.cutof", "layers.1.nb"]


def test_direct_construction_rejects_unknown_keys_too():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DemoConfig(model={"num_interaction": 3})


# ---------------------------------------------------------------------------
# Resolved export


def test_resolved_dict_has_every_default_in_declaration_order(tmp_path):
    # The file lists keys in the reverse of the schema's order.
    path = tmp_path / "reversed.yaml"
    path.write_text("seed: 1\nname: x\n", encoding="utf-8")
    resolved = DemoConfig.load(path).to_resolved_dict()
    assert list(resolved) == [
        "name",
        "seed",
        "default_dtype",
        "model",
        "data",
        "stage_two",
    ]
    assert resolved["stage_two"] == {"start_epoch": 100, "energy_weight": 1000.0}
    assert list(resolved["model"]) == ["num_interactions", "hidden_irreps", "radial"]
    assert resolved["model"]["radial"] == {"num_bessel": 8, "cutoff": 5.0}
    assert resolved["data"]["train_file"] is None


def assert_fixed_point(tmp_path, first, extension):
    written = tmp_path / f"resolved{extension}"
    written.write_text(dump(first, extension), encoding="utf-8")
    second = DemoConfig.load(written).to_resolved_dict()
    assert second == first
    assert json.dumps(second) == json.dumps(first)  # order included


@pytest.mark.parametrize("extension", [".yaml", ".json"])
def test_file_to_resolved_to_file_to_resolved_is_a_fixed_point(tmp_path, extension):
    values = {**FILE_VALUES, "data": {"train_file": None, "heads": ["pbe"]}}
    first = DemoConfig.load(write_config(tmp_path, ".yaml", values)).to_resolved_dict()
    assert first["data"]["train_file"] is None  # a None is part of what has to survive
    assert_fixed_point(tmp_path, first, extension)


def test_fixed_point_holds_through_toml_when_nothing_is_none(tmp_path):
    # TOML has no null, so the file sets the optional file name; the resolved
    # dict then goes through all three formats.
    values = {**FILE_VALUES, "stage_two": {"start_epoch": 50}}
    first = DemoConfig.load(write_config(tmp_path, ".yaml", values)).to_resolved_dict()
    assert "null" not in json.dumps(first)
    for extension in (".toml", ".yaml", ".json"):
        assert_fixed_point(tmp_path, first, extension)


def test_inf_and_nan_survive_the_exports_in_every_format(tmp_path):
    # pydantic's JSON mode writes them as null by default, which would put a
    # different value into the model metadata. Each format spells them its own
    # way (JSON constants, YAML .inf/.nan, TOML inf/nan); nan != nan, so the
    # round trip is compared as JSON text.
    values = {
        **FILE_VALUES,  # the file sets the optional file name, so no null
        "model": {"radial": {"cutoff": math.inf}},
        "stage_two": {"energy_weight": math.nan},
    }
    config = DemoConfig.load(write_config(tmp_path, ".yaml", values))
    resolved = config.to_resolved_dict()
    assert resolved["model"]["radial"]["cutoff"] == math.inf
    assert math.isnan(resolved["stage_two"]["energy_weight"])
    assert math.isnan(config.to_user_dict()["stage_two"]["energy_weight"])
    text = json.dumps(resolved)
    assert "null" not in text and "Infinity" in text and "NaN" in text
    for extension, body in [
        (".json", text),
        (".yaml", yaml.safe_dump(resolved)),
        (".toml", "[model.radial]\ncutoff = inf\n[stage_two]\nenergy_weight = nan\n"),
    ]:
        path = tmp_path / f"special{extension}"
        path.write_text(body, encoding="utf-8")
        second = DemoConfig.load(path).to_resolved_dict()
        assert second["model"]["radial"]["cutoff"] == math.inf, extension
        assert math.isnan(second["stage_two"]["energy_weight"]), extension
    assert (
        json.dumps(DemoConfig.load(tmp_path / "special.json").to_resolved_dict())
        == text
    )


class LenientSection(BaseModel):
    cutoff: float = 5.0


def test_field_shapes_the_contract_cannot_keep_are_rejected_at_class_definition():
    # Each shape would break a guarantee: set order varies with the hash
    # seed; aliases, excluded and computed fields do not validate back; a
    # lenient section would swallow typos.
    shapes = {
        r"tags is typed as a set.*Use a list": ("tags", list[set[str]]),
        r"names is typed as a set.*Use a list": ("names", AbstractSet[str]),
        r"num has an alias": ("num", Annotated[int, Field(alias="n")]),
        r"vnum has an alias": ("vnum", Annotated[int, Field(validation_alias="n")]),
        r"snum has an alias": ("snum", Annotated[int, Field(serialization_alias="n")]),
        r"hidden is excluded from dumps": (
            "hidden",
            Annotated[int, Field(exclude=True)],
        ),
        r"radial holds LenientSection, which is not a ConfigSection": (
            "radial",
            LenientSection | None,
        ),
    }
    for message, (name, annotation) in shapes.items():
        with pytest.raises(TypeError, match=message):
            type("Bad", (ConfigSection,), {"__annotations__": {name: annotation}})

    with pytest.raises(TypeError, match=r"double is a computed field"):

        class Computed(ConfigSection):
            seed: int = 1

            @computed_field
            def double(self) -> int:
                return 2 * self.seed


def test_a_section_cannot_reopen_extra():
    with pytest.raises(TypeError, match=r"Loose sets extra='allow'; a section keeps"):

        class Loose(ConfigSection):
            model_config = ConfigDict(extra="allow")
            seed: int = 1


# A class that names a class defined below it is incomplete at definition:
# pydantic keeps the name, so the check cannot see through the field. The
# first `load` resolves the name and checks the class then.


class Forward(ConfigSection):
    later: "Later" = Field(default_factory=lambda: Later())


class Later(ConfigSection):
    x: int = 1


class ForwardConfig(ReforgeBaseConfig):
    forward: Forward = Field(default_factory=Forward)


class Leaking(ConfigSection):
    plain: "PlainLater" = Field(default_factory=lambda: PlainLater())


class PlainLater(BaseModel):  # not a ConfigSection: it would swallow a typo
    a: int = 1


class LeakingConfig(ReforgeBaseConfig):
    leaking: Leaking = Field(default_factory=Leaking)


def test_a_forward_reference_is_resolved_and_checked_on_load(tmp_path):
    empty = write_config(tmp_path, ".yaml", {})
    assert ForwardConfig.load(empty).forward.later == Later()
    config = ForwardConfig.from_dict({"forward": {"later": {"x": 2}}})
    assert config.forward.later == Later(x=2)
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"forward": {"later": {"x": 2, "typo": 1}}}))
    with pytest.raises(ValidationError) as excinfo:
        ForwardConfig.load(path)
    assert error_locations(excinfo) == ["forward.later.typo"]


def test_a_lenient_section_behind_a_forward_reference_is_rejected(tmp_path):
    with pytest.raises(
        TypeError, match=r"Leaking.plain holds PlainLater, which is not a ConfigSection"
    ):
        LeakingConfig.load(write_config(tmp_path, ".yaml", {}))


def test_user_dict_holds_only_what_was_set(tmp_path):
    config = DemoConfig.load(write_config(tmp_path, ".json"))
    assert config.to_user_dict() == FILE_VALUES
    assert config.to_user_dict() is not FILE_VALUES


# ---------------------------------------------------------------------------
# Nothing but the file feeds a config.


def test_environment_variables_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("NAME", "from-the-environment")
    monkeypatch.setenv("SEED", "99")
    config = DemoConfig.load(write_config(tmp_path, ".yaml", {}))
    assert config.name == "mace"
    assert config.seed == 123


@pytest.mark.parametrize("key", ["SEED", "Seed", "_env_file", "_cli_parse_args"])
def test_root_keys_are_validated_like_any_section(tmp_path, key):
    # Neither case variants nor BaseSettings-style private constructor
    # options are special at the top level: unknown is unknown.
    path = tmp_path / "root.json"
    path.write_text(json.dumps({key: 1}), encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        DemoConfig.load(path)
    assert error_locations(excinfo) == [key]


def test_config_module_imports_neither_torch_nor_jax():
    """In a fresh interpreter, so another test's imports cannot mask a leak."""
    code = (
        "import sys, mace_core.config; "
        "leaked = {'torch', 'jax', 'e3nn'} & set(sys.modules); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_base_does_not_import_the_command_line_module():
    # The package init re-exports both, so `sys.modules` cannot tell; the
    # source can: `cli` imports `ConfigError` from `base`, never the reverse.
    import mace_core.config.base as base_module

    source = inspect.getsource(base_module)
    assert not re.search(r"^\s*(from|import) .*\bcli\b", source, re.MULTILINE)
