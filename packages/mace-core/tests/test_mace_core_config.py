"""`ReforgeBaseConfig`: file formats, precedence, dotted overrides, unknown keys,
overrides without effect, and the resolved export's fixed point."""

import json
import re
import subprocess
import sys
import warnings
from typing import Annotated, Any

import pytest
import yaml
from mace_core.config import (
    ConfigError,
    ConfigSection,
    ConfigWarning,
    ReforgeBaseConfig,
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


# ---------------------------------------------------------------------------
# File loading


@pytest.mark.parametrize("extension", [".toml", ".yaml", ".yml", ".json"])
def test_same_config_loads_identically_from_every_format(tmp_path, extension):
    config = DemoConfig.load(write_config(tmp_path, extension))
    assert config == DemoConfig.model_validate(FILE_VALUES)
    # The file set two fields at depth two; the sibling kept its default.
    assert config.model.radial.cutoff == 4.5
    assert config.model.radial.num_bessel == 8


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


def test_file_must_be_a_table_at_the_top(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="table of keys at the top level"):
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


# ---------------------------------------------------------------------------
# Precedence: defaults < files in order < CLI. The legacy behaviour this pins is
# tests/unit/test_arg_parser.py::test_cli_flag_overrides_yaml_config.


def test_no_inputs_gives_the_defaults():
    config = DemoConfig.load()
    assert config == DemoConfig()
    assert config.model.num_interactions == 2
    assert DemoConfig.load(None) == DemoConfig()  # an optional path, unset


def test_file_overrides_defaults(tmp_path):
    config = DemoConfig.load(write_config(tmp_path, ".yaml"))
    assert config.model.num_interactions == 4  # from the file
    assert config.default_dtype == "float64"  # untouched default


def test_cli_overrides_file_which_overrides_defaults(tmp_path):
    config = DemoConfig.load(
        write_config(tmp_path, ".toml"), ["--model.num_interactions", "3"]
    )
    assert config.model.num_interactions == 3  # CLI beats the file's 4
    assert config.model.radial.cutoff == 4.5  # the file's other values survive
    assert config.seed == 7
    assert config.model.radial.num_bessel == 8  # defaults fill the rest
    assert config.default_dtype == "float64"


def test_files_apply_in_order_before_the_overrides(tmp_path):
    first = write_config(tmp_path, ".yaml", name="defaults")
    second = write_config(
        tmp_path, ".toml", {"seed": 8, "model": {"radial": {"num_bessel": 6}}}, "site"
    )
    config = DemoConfig.load([first, second], ["--model.num_interactions", "3"])
    assert config.seed == 8  # the second file beats the first
    assert config.name == "water"  # the first file's other values survive
    assert config.model.radial == RadialSection(num_bessel=6, cutoff=4.5)
    assert config.model.num_interactions == 3  # the CLI beats both
    assert DemoConfig.load([]) == DemoConfig()


# ---------------------------------------------------------------------------
# Dotted CLI overrides


def test_dotted_override_reaches_a_two_level_nested_field():
    config = DemoConfig.load(cli_overrides=["--model.radial.cutoff", "6.0"])
    assert config.model.radial.cutoff == 6.0
    assert config.model.radial.num_bessel == 8


def test_dotted_override_reaches_a_section_left_at_its_defaults():
    config = DemoConfig.load(cli_overrides=["--stage_two.start_epoch", "50"])
    assert config.stage_two == StageTwoSection(start_epoch=50)
    assert DemoConfig.load().stage_two == StageTwoSection()


def test_override_forms_and_types():
    config = DemoConfig.load(
        cli_overrides=[
            "--seed=9",
            "--data.train_file",
            "null",
            "--data.heads",
            '["a", "b"]',
        ]
    )
    assert config.seed == 9
    assert config.data.train_file is None
    assert config.data.heads == ["a", "b"]


def test_value_of_the_wrong_type_is_a_validation_error(tmp_path):
    with pytest.raises(ValidationError, match="seed"):
        DemoConfig.load(cli_overrides=["--seed", "seven"])
    with pytest.raises(ValidationError, match="seed"):
        DemoConfig.load(write_config(tmp_path, ".yaml", {"seed": "seven"}))


def test_override_missing_its_value_is_a_config_error():
    with pytest.raises(ConfigError, match="override --seed is missing its value"):
        DemoConfig.load(cli_overrides=["--seed"])


def test_override_that_is_not_valid_json_is_a_config_error():
    with pytest.raises(ConfigError, match="override --model is not valid JSON"):
        DemoConfig.load(cli_overrides=["--model", "{oops"])


def test_value_starting_with_dashes_works_in_both_forms():
    assert DemoConfig.load(cli_overrides=["--name=--odd"]).name == "--odd"
    assert DemoConfig.load(cli_overrides=["--name", "--odd"]).name == "--odd"


@pytest.mark.parametrize("token", ["--", "--=5"])
def test_bare_dashes_are_an_unknown_option_not_a_key(token):
    with pytest.raises(ConfigError, match=rf"unknown config option '{token}'"):
        DemoConfig.load(cli_overrides=[token, "--seed", "5"])


def test_overrides_given_as_one_string_are_refused():
    with pytest.raises(TypeError, match="cli_overrides is a string"):
        DemoConfig.load(cli_overrides="--seed 5")


def test_dict_valued_field_takes_json_and_dotted_paths_into_its_entries():
    class Sources(ReforgeBaseConfig):
        by_name: dict[str, RadialSection] = Field(default_factory=dict)

    config = Sources.load(cli_overrides=["--by_name", '{"pbe": {"cutoff": 4.0}}'])
    assert config.by_name == {"pbe": RadialSection(cutoff=4.0)}
    dotted = Sources.load(cli_overrides=["--by_name.pbe.cutoff", "4.0"])
    assert dotted.by_name == {"pbe": RadialSection(cutoff=4.0)}
    # Inside an entry, the neighbour is still found: the key passes through.
    with pytest.raises(
        ConfigError,
        match=r"'by_name\.pbe\.cutof'; did you mean 'by_name\.pbe\.cutoff'\?",
    ):
        Sources.load(cli_overrides=["--by_name", '{"pbe": {"cutof": 4.0}}'])


def test_collections_behind_none_or_annotated_keep_their_dotted_paths(tmp_path):
    class Collections(ReforgeBaseConfig):
        counts: dict[str, int] | None = None
        documented: dict[str, Annotated[RadialSection, Field(description="d")]] = Field(
            default_factory=dict
        )
        pair: tuple[int, RadialSection] | None = None

    assert Collections.load(cli_overrides=["--counts.x", "1"]).counts == {"x": 1}
    with pytest.raises(
        ConfigError,
        match=r"'documented\.a\.cutof'; did you mean 'documented\.a\.cutoff'",
    ):
        Collections.load(cli_overrides=["--documented", '{"a": {"cutof": 4.0}}'])
    path = write_config(tmp_path, ".json", {"pair": [1, {"cutof": 4.0}]})
    with pytest.raises(
        ConfigError, match=r"'pair\.1\.cutof'; did you mean 'pair\.1\.cutoff'"
    ):
        Collections.load(path)


def test_dict_override_merges_entries_but_list_override_replaces(tmp_path):
    class Sources(ReforgeBaseConfig):
        by_name: dict[str, RadialSection] = Field(default_factory=dict)
        heads: list[str] = Field(default_factory=list)

    path = write_config(
        tmp_path, ".yaml", {"by_name": {"pbe": {"cutoff": 4.0}}, "heads": ["a", "b"]}
    )
    config = Sources.load(
        path, ["--by_name", '{"r2scan": {"cutoff": 6.0}}', "--heads", '["c"]']
    )
    assert set(config.by_name) == {"pbe", "r2scan"}
    assert config.heads == ["c"]


def test_overrides_apply_in_order_on_top_of_the_file(tmp_path):
    # A dotted value merges into what the file set in the section; a dotted
    # value followed by the whole section keeps both.
    config = DemoConfig.load(
        write_config(tmp_path, ".yaml", {"stage_two": {"energy_weight": 5.0}}),
        cli_overrides=[
            "--stage_two.start_epoch",
            "5",
            "--model.radial.cutoff",
            "4",
            "--model",
            '{"num_interactions": 3}',
        ],
    )
    assert config.stage_two == StageTwoSection(start_epoch=5, energy_weight=5.0)
    assert (config.model.num_interactions, config.model.radial.cutoff) == (3, 4.0)


def test_a_yaml_anchor_does_not_share_an_override(tmp_path):
    class Two(ReforgeBaseConfig):
        a: dict[str, Any] = Field(default_factory=dict)
        b: dict[str, Any] = Field(default_factory=dict)

    path = tmp_path / "anchors.yaml"
    path.write_text("a: &empty {}\nb: *empty\n", encoding="utf-8")
    config = Two.load(path, ["--a.x", "1"])
    assert (config.a, config.b) == ({"x": "1"}, {})


# ---------------------------------------------------------------------------
# An override that had no effect on the config that runs is a warning; a file
# never warns.


class Extras(ReforgeBaseConfig):
    seed: int = 1
    extra: dict[str, Any] = Field(default_factory=dict)


WITHOUT_EFFECT = {
    "a later override at the same path": (
        ["--seed", "1", "--seed", "2"],
        ["--seed 1 is overridden: seed is 2"],
    ),
    "a later override above it": (
        ["--extra.a.b", "2", "--extra.a", "5"],
        ["--extra.a.b 2 is overridden: extra.a is 5"],
    ),
    "a later json override above it": (
        ["--extra.a.b", "2", "--extra.a", "[1]"],
        ["--extra.a.b 2 is overridden: extra.a is [1]"],
    ),
    "part of a json override replaced": (
        ["--extra", '{"a": {"b": 1}}', "--extra.a.b", "2"],
        ['--extra {"a": {"b": 1}} is overridden: extra.a.b is 2'],
    ),
    "a later override below a scalar of it": (
        ["--extra.a", "5", "--extra.a.b", "2"],
        ["--extra.a 5 is overridden: extra.a.b is 2"],
    ),
    "the same override twice: the first is overridden": (
        ["--seed", "2", "--seed", "2"],
        ["--seed 2 is overridden: seed is 2"],
    ),
    "the same json override twice: the first is overridden": (
        ["--extra", '{"a": 1}', "--extra", '{"a": 1}'],
        ['--extra {"a": 1} is overridden: extra.a is 1'],
    ),
    "the last of three at one path is named, once": (
        ["--seed", "1", "--seed", "1", "--seed", "3"],
        ["--seed 1 is overridden: seed is 3"],
    ),
    "a plain key named kind is a key, not a selection": (
        ["--extra.kind", "a", "--extra.kind", "b"],
        ["--extra.kind a is overridden: extra.kind is b"],
    ),
    "an empty mapping above a later key is a merge": (
        ["--extra.a.b", "2", "--extra.a", "{}"],
        [],
    ),
    "an empty mapping below a later key is a merge": (
        ["--extra.a", "{}", "--extra.a.b", "2"],
        [],
    ),
}


@pytest.mark.parametrize("row", WITHOUT_EFFECT, ids=WITHOUT_EFFECT)
def test_an_override_without_effect_warns(row):
    tokens, expected = WITHOUT_EFFECT[row]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Extras.load(cli_overrides=tokens)
    assert [str(w.message) for w in caught] == expected
    assert all(issubclass(w.category, ConfigWarning) for w in caught)


def test_a_file_value_the_cli_replaces_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        config = DemoConfig.load(write_config(tmp_path, ".yaml"), ["--seed", "9"])
    assert config.seed == 9


# ---------------------------------------------------------------------------
# Unknown keys name the key and its nearest neighbour, in files and on the CLI.


def test_unknown_top_level_key_in_file(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("sead: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"'sead'; did you mean 'seed'\?"):
        DemoConfig.load(path)


def test_unknown_nested_key_in_file_names_the_dotted_path(tmp_path):
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"model": {"radial": {"cutof": 4.0}}}), encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match=r"'model\.radial\.cutof'; did you mean 'model\.radial\.cutoff'\?",
    ):
        DemoConfig.load(path)


def test_every_unknown_key_is_reported_at_once(tmp_path):
    path = tmp_path / "typos.yaml"
    path.write_text("sead: 1\nmodel:\n  num_interaction: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        DemoConfig.load(path)
    assert "'sead'" in str(excinfo.value)
    assert "'model.num_interaction'" in str(excinfo.value)


def test_unknown_keys_are_reported_under_their_file_or_override(tmp_path):
    first = write_config(tmp_path, ".yaml", {"sead": 1}, "first")
    second = write_config(
        tmp_path, ".json", {"model": {"num_interaction": 3}}, "second"
    )
    with pytest.raises(ConfigError) as excinfo:
        DemoConfig.load([first, second], ["--nmae", "x"])
    assert str(excinfo.value).splitlines() == [
        f"{first}: unknown config key 'sead'; did you mean 'seed'?",
        f"{second}: unknown config key 'model.num_interaction'; "
        "did you mean 'model.num_interactions'?",
        "--nmae x: unknown config key 'nmae'; did you mean 'name'?",
    ]


def test_unknown_key_without_a_close_neighbour_still_names_it(tmp_path):
    path = tmp_path / "far.yaml"
    path.write_text("zzzzzz: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown config key 'zzzzzz'$"):
        DemoConfig.load(path)


def test_unknown_dotted_override_names_the_neighbour():
    with pytest.raises(
        ConfigError,
        match=r"'model\.num_interaction'; did you mean 'model\.num_interactions'\?",
    ):
        DemoConfig.load(cli_overrides=["--model.num_interaction", "3"])


def test_unknown_key_inside_a_nested_section():
    with pytest.raises(
        ConfigError,
        match=r"'stage_two\.start'; did you mean 'stage_two\.start_epoch'\?",
    ):
        DemoConfig.load(cli_overrides=["--stage_two.start", "50"])


def test_every_bad_list_item_is_reported(tmp_path):
    class Layers(ReforgeBaseConfig):
        layers: list[RadialSection] = Field(default_factory=list)

    path = write_config(tmp_path, ".json", {"layers": [{"cutof": 1}, {"nb": 2}]})
    with pytest.raises(ConfigError) as excinfo:
        Layers.load(path)
    assert re.findall(r"unknown config key '([^']*)'", str(excinfo.value)) == [
        "layers.0.cutof",
        "layers.1.nb",
    ]


def test_help_flag_is_an_error_not_an_exit():
    with pytest.raises(ConfigError, match=r"unknown config option '-h'"):
        DemoConfig.load(cli_overrides=["-h"])


def test_abbreviated_option_is_unknown_not_expanded():
    with pytest.raises(ConfigError, match=r"key 'se'; did you mean 'seed'"):
        DemoConfig.load(cli_overrides=["--se", "3"])


def test_empty_inline_value_does_not_hide_the_next_option():
    with pytest.raises(ConfigError, match=r"'sead'; did you mean 'seed'\?"):
        DemoConfig.load(cli_overrides=["--name=", "--sead", "5"])


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
    first = DemoConfig.load(
        write_config(tmp_path, ".toml"),
        ["--model.num_interactions", "3", "--data.train_file", "null"],
    ).to_resolved_dict()
    assert first["data"]["train_file"] is None  # a None is part of what has to survive
    assert_fixed_point(tmp_path, first, extension)


def test_fixed_point_holds_through_toml_when_nothing_is_none(tmp_path):
    # TOML has no null, so the file sets the optional file name; the resolved
    # dict then goes through all three formats.
    first = DemoConfig.load(
        write_config(tmp_path, ".yaml"), ["--stage_two.start_epoch", "50"]
    ).to_resolved_dict()
    assert "null" not in json.dumps(first)
    for extension in (".toml", ".yaml", ".json"):
        assert_fixed_point(tmp_path, first, extension)


class LenientSection(BaseModel):
    cutoff: float = 5.0


def test_field_shapes_the_contract_cannot_keep_are_rejected_at_class_definition():
    # Each shape would break a guarantee: set order varies with the hash
    # seed; aliases, excluded and computed fields do not validate back; a
    # union of sections would let a value pick its section; a lenient section
    # would swallow typos; a section is never optional.
    shapes = {
        r"tags is typed as a set.*Use a list": ("tags", list[set[str]]),
        r"num has an alias": ("num", Annotated[int, Field(alias="n")]),
        r"vnum has an alias": ("vnum", Annotated[int, Field(validation_alias="n")]),
        r"snum has an alias": ("snum", Annotated[int, Field(serialization_alias="n")]),
        r"hidden is excluded from dumps": (
            "hidden",
            Annotated[int, Field(exclude=True)],
        ),
        r"either is a union of sections": ("either", RadialSection | StageTwoSection),
        r"radial holds LenientSection, which is not a ConfigSection": (
            "radial",
            LenientSection | None,
        ),
        r"radial mixes its kinds with int": ("radial", RadialSection | int),
        r"stage admits None": ("stage", StageTwoSection | None),
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


def test_a_forward_reference_is_checked_and_walked_once_it_resolves(tmp_path):
    assert not Forward.__pydantic_complete__
    assert ForwardConfig.load().forward.later == Later()
    config = ForwardConfig.load(cli_overrides=["--forward.later.x", "2"])
    assert config.forward.later == Later(x=2)
    path = tmp_path / "typo.json"
    path.write_text(json.dumps({"forward": {"later": {"x": 2, "typo": 1}}}))
    with pytest.raises(ConfigError, match=r"unknown config key 'forward.later.typo'"):
        ForwardConfig.load(path)


def test_a_lenient_section_behind_a_forward_reference_is_rejected():
    with pytest.raises(
        TypeError, match=r"Leaking.plain holds PlainLater, which is not a ConfigSection"
    ):
        LeakingConfig.load()


def test_user_dict_holds_only_what_was_set(tmp_path):
    config = DemoConfig.load(
        write_config(tmp_path, ".json"), ["--model.num_interactions", "3"]
    )
    assert config.to_user_dict() == {
        "name": "water",
        "seed": 7,
        "model": {"num_interactions": 3, "radial": {"cutoff": 4.5}},
        "data": {"train_file": "train.xyz", "heads": ["pbe", "r2scan"]},
    }


# ---------------------------------------------------------------------------
# Nothing but the files and the CLI feed a config.


def test_environment_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("NAME", "from-the-environment")
    monkeypatch.setenv("SEED", "99")
    config = DemoConfig.load()
    assert config.name == "mace"
    assert config.seed == 123


@pytest.mark.parametrize("key", ["SEED", "Seed", "_env_file", "_cli_parse_args"])
def test_root_keys_are_validated_like_any_section(tmp_path, key):
    # Neither case variants nor BaseSettings-style private constructor
    # options are special at the top level: unknown is unknown.
    path = tmp_path / "root.json"
    path.write_text(json.dumps({key: 1}), encoding="utf-8")
    with pytest.raises(ConfigError, match=f"unknown config key '{key}'"):
        DemoConfig.load(path)


def test_config_module_imports_neither_torch_nor_jax():
    """In a fresh interpreter, so another test's imports cannot mask a leak."""
    code = (
        "import sys, mace_core.config; "
        "leaked = {'torch', 'jax', 'e3nn'} & set(sys.modules); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
