"""`mace_core.config.cli`: the `--a.b value` grammar to a mapping of dotted
paths, and that mapping written into a parsed file before one validation. The
config base knows nothing of either; a command line composes them with
`read_config_file` and `from_dict`."""

import json
from typing import Annotated, Any, Literal

import pytest
from mace_core.config import (
    ConfigError,
    ConfigSection,
    ReforgeBaseConfig,
    apply_overrides,
    parse_overrides,
    read_config_file,
)
from pydantic import Field, ValidationError

#: A warning the test did not ask for is a failure.
pytestmark = pytest.mark.filterwarnings("error")

# ---------------------------------------------------------------------------
# The demo schema: two levels of nesting, a list, an optional, a free dict, a
# kinds field.


class RadialSection(ConfigSection):
    num_bessel: int = 8
    cutoff: float = 5.0


class ModelSection(ConfigSection):
    num_interactions: int = 2
    radial: RadialSection = RadialSection()


class DataSection(ConfigSection):
    train_file: str | None = None
    heads: list[str] = Field(default_factory=lambda: ["default"])


class StageTwoSection(ConfigSection):
    start_epoch: int = 100
    energy_weight: float = 1000.0


class Weighted(ConfigSection):
    kind: Literal["weighted"] = "weighted"
    stress_weight: float = 0.0


class Huber(ConfigSection):
    kind: Literal["huber"] = "huber"
    delta: float = 0.01


class DemoConfig(ReforgeBaseConfig):
    name: str = "mace"
    seed: int = 123
    model: ModelSection = ModelSection()
    data: DataSection = DataSection()
    stage_two: StageTwoSection = StageTwoSection()
    loss: Annotated[Weighted | Huber, Field(discriminator="kind")] = Weighted()
    extra: dict[str, Any] = Field(default_factory=dict)


FILE_VALUES = {
    "name": "water",
    "seed": 7,
    "model": {"num_interactions": 4, "radial": {"cutoff": 4.5}},
    "data": {"train_file": "train.xyz", "heads": ["pbe", "r2scan"]},
}

HEADS_JSON = '["a", "b"]'


def write_config(tmp_path, values=FILE_VALUES):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def load(tmp_path, argv, values=FILE_VALUES, root=DemoConfig):
    """What a command line does with its config file and the tokens after it."""
    document = read_config_file(write_config(tmp_path, values))
    return root.from_dict(apply_overrides(document, parse_overrides(argv)))


def error_locations(excinfo):
    return [".".join(map(str, error["loc"])) for error in excinfo.value.errors()]


# ---------------------------------------------------------------------------
# parse_overrides: tokens to a mapping of dotted path to value.


def test_both_forms_and_the_value_types():
    argv = ["--seed=9", "--data.train_file", "null", "--data.heads", HEADS_JSON]
    assert parse_overrides(argv) == {
        "seed": "9",  # a string: pydantic coerces it, the grammar does not
        "data.train_file": None,
        "data.heads": ["a", "b"],
    }
    assert parse_overrides(["--model", json.dumps({"num_interactions": 3})]) == {
        "model": {"num_interactions": 3}
    }
    assert parse_overrides([]) == {}


def test_paths_keep_their_order_and_a_repeated_path_moves_to_where_it_is_last():
    overrides = parse_overrides(["--b", "1", "--a", "2", "--b", "3"])
    assert list(overrides.items()) == [("a", "2"), ("b", "3")]
    # So a later write under a parent the same command line replaced survives.
    argv = ["--m.r.c", "4", "--m.r", "null", "--m.r.c", "5"]
    assert apply_overrides({}, parse_overrides(argv)) == {"m": {"r": {"c": "5"}}}


def test_an_empty_inline_value_does_not_hide_the_next_option():
    assert parse_overrides(["--name=", "--seed", "5"]) == {"name": "", "seed": "5"}


@pytest.mark.parametrize("token", ["--a..b", "--a.", "--.a"])
def test_an_empty_key_in_a_path_is_a_config_error(token):
    with pytest.raises(ConfigError, match=rf"override {token} has an empty key"):
        parse_overrides([token, "1"])


def test_a_value_starting_with_dashes_works_in_both_forms():
    assert parse_overrides(["--name=--odd"]) == {"name": "--odd"}
    assert parse_overrides(["--name", "--odd"]) == {"name": "--odd"}


def test_a_path_missing_its_value_is_a_config_error():
    with pytest.raises(ConfigError, match="override --seed is missing its value"):
        parse_overrides(["--seed"])


def test_a_value_that_is_not_valid_json_is_a_config_error():
    with pytest.raises(ConfigError, match="override --model is not valid JSON"):
        parse_overrides(["--model", "{oops"])


@pytest.mark.parametrize("token", ["--", "--=5", "seed=5", "-s"])
def test_a_token_that_is_not_a_dotted_option_is_a_config_error(token):
    with pytest.raises(ConfigError, match=rf"unknown config option '{token}'"):
        parse_overrides([token, "--seed", "5"])


def test_tokens_given_as_one_string_are_refused():
    with pytest.raises(TypeError, match="tokens is a string"):
        parse_overrides("--seed 5")


# ---------------------------------------------------------------------------
# apply_overrides: a dotted path is written into a copy of the parsed file.


def test_a_top_level_key_is_written():
    assert apply_overrides({}, {"seed": 9}) == {"seed": 9}
    assert apply_overrides({"name": "x"}, {"seed": 9}) == {"name": "x", "seed": 9}


def test_a_nested_path_writes_into_the_section_the_file_set():
    document = {"model": {"num_interactions": 4, "radial": {"cutoff": 4.5}}}
    assert apply_overrides(document, {"model.radial.cutoff": 6.0}) == {
        "model": {"num_interactions": 4, "radial": {"cutoff": 6.0}}
    }


def test_the_sections_a_path_passes_through_are_created():
    assert apply_overrides({}, {"model.radial.cutoff": 6.0, "stage_two.x": 1}) == {
        "model": {"radial": {"cutoff": 6.0}},
        "stage_two": {"x": 1},
    }


@pytest.mark.parametrize("parent", [5, [1, 2], None], ids=["scalar", "list", "null"])
def test_a_parent_that_is_not_a_mapping_is_replaced(parent):
    assert apply_overrides({"a": parent}, {"a.b": 2}) == {"a": {"b": 2}}


def test_a_mapping_value_merges_into_a_mapping_but_a_list_replaces():
    document = {"by_name": {"pbe": {"cutoff": 4.0}}, "heads": ["a", "b"]}
    merged = apply_overrides(
        document, {"by_name": {"r2scan": {"cutoff": 6.0}}, "heads": ["c"]}
    )
    assert merged == {
        "by_name": {"pbe": {"cutoff": 4.0}, "r2scan": {"cutoff": 6.0}},
        "heads": ["c"],
    }
    # Deeper too, and a key of the JSON value is one key even with a dot in it.
    assert apply_overrides(
        {"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"c": 3}, "x.y": 4}}
    ) == {"a": {"b": {"c": 3, "d": 2}, "x.y": 4}}
    # A mapping over a scalar, or a scalar over a mapping, replaces.
    assert apply_overrides({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert apply_overrides({"a": {"b": 2}}, {"a": 1}) == {"a": 1}


def test_paths_apply_in_order_on_top_of_the_file():
    # A dotted value followed by the whole section keeps both.
    document = {"stage_two": {"energy_weight": 5.0}}
    argv = ["--stage_two.start_epoch", "5", "--model.radial.cutoff", "4"]
    argv += ["--model", json.dumps({"num_interactions": 3})]
    assert apply_overrides(document, parse_overrides(argv)) == {
        "stage_two": {"energy_weight": 5.0, "start_epoch": "5"},
        "model": {"radial": {"cutoff": "4"}, "num_interactions": 3},
    }


def test_neither_the_document_nor_the_overrides_are_written_into():
    document = {"extra": {"a": {"x": {}}}}
    overrides = {"extra.a": {"x": {"y": 1}}, "extra.a.x.z": [1]}
    copy = apply_overrides(document, overrides)
    assert copy == {"extra": {"a": {"x": {"y": 1, "z": [1]}}}}
    assert document == {"extra": {"a": {"x": {}}}}
    assert overrides == {"extra.a": {"x": {"y": 1}}, "extra.a.x.z": [1]}
    copy["extra"]["a"]["x"]["z"].append(2)  # the value was copied too
    assert overrides["extra.a.x.z"] == [1]


def test_a_yaml_anchor_does_not_share_an_override(tmp_path):
    path = tmp_path / "anchors.yaml"
    path.write_text("a: &empty {}\nb: *empty\n", encoding="utf-8")
    document = read_config_file(path)
    assert document["a"] is document["b"]  # what the parser hands over
    assert apply_overrides(document, {"a.x": 1}) == {"a": {"x": 1}, "b": {}}


def test_a_value_that_contains_itself_is_a_config_error():
    loop: dict[str, Any] = {}
    loop["b"] = loop
    with pytest.raises(ConfigError, match="refers to itself"):
        apply_overrides({"a": loop}, {})
    with pytest.raises(ConfigError, match="refers to itself"):
        apply_overrides({}, {"a": loop})


# ---------------------------------------------------------------------------
# Composed with the base: the override beats the file, which beats the
# defaults, and every schema error is pydantic's at the path the override named.


def test_an_override_beats_the_file_which_beats_the_defaults(tmp_path):
    config = load(tmp_path, ["--model.num_interactions", "3"])
    assert config.model.num_interactions == 3  # the override beats the file's 4
    assert config.model.radial.cutoff == 4.5  # the file's other values survive
    assert config.model.radial.num_bessel == 8  # defaults fill the rest
    assert load(tmp_path, []) == DemoConfig.from_dict(FILE_VALUES)


def test_values_are_handed_to_pydantic_as_given(tmp_path):
    argv = ["--seed=9", "--data.train_file", "null", "--data.heads", HEADS_JSON]
    config = load(tmp_path, argv, {})
    assert config.seed == 9  # pydantic's lax coercion
    assert config.data.train_file is None
    assert config.data.heads == ["a", "b"]
    assert load(tmp_path, ["--stage_two.start_epoch", "50"], {}).stage_two == (
        StageTwoSection(start_epoch=50)
    )


def test_a_value_of_the_wrong_type_is_a_validation_error(tmp_path):
    with pytest.raises(ValidationError, match="seed"):
        load(tmp_path, ["--seed", "seven"])


@pytest.mark.parametrize(
    "dotted_path", ["nmae", "model.num_interaction", "stage_two.start"]
)
def test_an_unknown_key_in_an_override_is_reported_at_its_path(tmp_path, dotted_path):
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, [f"--{dotted_path}", "1"], {})
    assert error_locations(excinfo) == [dotted_path]
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_an_override_writes_into_a_kinds_field(tmp_path):
    config = load(tmp_path, ["--loss.delta", "2"], {"loss": {"kind": "huber"}})
    assert config.loss == Huber(kind="huber", delta=2.0)
    config = load(tmp_path, ["--loss", json.dumps({"kind": "huber", "delta": 2})], {})
    assert config.loss == Huber(kind="huber", delta=2.0)
    assert config.to_user_dict() == {"loss": {"kind": "huber", "delta": 2.0}}


def test_an_override_that_changes_the_kind_leaves_the_old_settings_to_pydantic(
    tmp_path,
):
    # The file's `delta` stays in the dict and is an unknown key of the new kind.
    huber = {"loss": {"kind": "huber", "delta": 0.5}}
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, ["--loss.kind", "weighted"], huber)
    assert error_locations(excinfo) == ["loss.weighted.delta"]
