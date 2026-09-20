"""A field of several kinds of section (a discriminated union) under the file
and dotted-override contract. A config writes the kind as the key the section
sits under (`loss: {huber: {delta: 0.1}}`, `--loss.huber.delta 0.1`) or as a
bare name for the kind with its defaults; code sees the union. The file and
every override merge in order, the kind written last wins, the others are
dropped with a warning, two kinds in one place is an error."""

import json
import re
import warnings
from typing import Annotated, Any, Literal

import pytest
from mace_core.config import (
    ConfigError,
    ConfigSection,
    ConfigWarning,
    ReforgeBaseConfig,
)
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# The schema: a loss of three kinds, one of which holds a field of two kinds;
# the same choice with "none of them" as a fourth kind and the default, and
# inside the values of a dict and the items of a list.


class SubX(ConfigSection):
    kind: Literal["x"] = "x"
    a: float = 1.0


class SubY(ConfigSection):
    kind: Literal["y"] = "y"
    b: float = 2.0


class Weighted(ConfigSection):
    kind: Literal["weighted"] = "weighted"
    stress_weight: float = 0.0


class Huber(ConfigSection):
    kind: Literal["huber"] = "huber"
    delta: float = 0.01
    sub: Annotated[SubX | SubY, Field(discriminator="kind")] = SubX()


class Universal(ConfigSection):
    kind: Literal["universal"] = "universal"
    huber_delta: float = 0.01


class Plain(ConfigSection):
    p: int = 0


Choice = Annotated[Weighted | Huber | Universal, Field(discriminator="kind")]


class NoChoice(ConfigSection):
    """None of the other kinds is a kind too."""

    kind: Literal["none"] = "none"


OptChoice = Annotated[
    Weighted | Huber | Universal | NoChoice, Field(discriminator="kind")
]


class HeadSection(ConfigSection):
    loss: Choice = Weighted()


class LossConfig(ReforgeBaseConfig):
    energy_weight: float = 1.0
    choice: Choice = Weighted()
    opt: OptChoice = NoChoice()
    heads: dict[str, Plain] = Field(default_factory=dict)
    per_head: dict[str, HeadSection] = Field(default_factory=dict)
    layers: list[HeadSection] = Field(default_factory=list)


class HuberRequired(ConfigSection):
    kind: Literal["huber"] = "huber"
    path: str
    delta: float = 0.01


class RequiredConfig(ReforgeBaseConfig):
    choice: Annotated[Weighted | HuberRequired, Field(discriminator="kind")] = (
        Weighted()
    )


# ---------------------------------------------------------------------------
# Expectations. A row is (file values, command line, expectation); the
# expectation is a predicate on the loaded config, optionally with the
# warnings the load must emit, or an error spec.

KINDS = "weighted, huber, universal"
OPT_KINDS = "weighted, huber, universal, none"


class Raises:
    def __init__(self, error_type, *fragments):
        self.error_type = error_type
        self.fragments = fragments


class Warns:
    """A predicate plus the exact warning messages, in order."""

    def __init__(self, predicate, *messages):
        self.predicate = predicate
        self.messages = messages


def error(*fragments):
    """A `ConfigError` whose message holds every fragment."""
    return Raises(ConfigError, *fragments)


def ignored(loser, source, winner_source, winner):
    return f"{loser} from {source} is ignored: {winner_source} selects {winner}"


def huber(delta=0.01, sub=SubX, **sub_fields):
    return lambda c: (
        isinstance(c.choice, Huber)
        and c.choice.delta == delta
        and isinstance(c.choice.sub, sub)
        and all(getattr(c.choice.sub, k) == v for k, v in sub_fields.items())
    )


def weighted(stress_weight=0.0):
    return lambda c: (
        isinstance(c.choice, Weighted) and c.choice.stress_weight == stress_weight
    )


def universal(huber_delta=0.01):
    return lambda c: (
        isinstance(c.choice, Universal) and c.choice.huber_delta == huber_delta
    )


def opt(kind, **fields):
    return lambda c: (
        isinstance(c.opt, kind)
        and all(getattr(c.opt, k) == v for k, v in fields.items())
    )


def per_head(name, kind, **fields):
    return lambda c: (
        isinstance(c.per_head[name].loss, kind)
        and all(getattr(c.per_head[name].loss, k) == v for k, v in fields.items())
    )


def layer(index, kind, **fields):
    return lambda c: (
        isinstance(c.layers[index].loss, kind)
        and all(getattr(c.layers[index].loss, k) == v for k, v in fields.items())
    )


HUBER_FILE = {"choice": {"huber": {"delta": 0.5}}}

SWITCHING = {
    "file kind, no cli": (HUBER_FILE, "", huber(0.5)),
    "cli name switches kind, file section dropped with a warning": (
        HUBER_FILE,
        "--choice weighted",
        Warns(
            weighted(),
            ignored(
                "choice.huber",
                "the config file",
                "--choice weighted",
                "choice.weighted",
            ),
        ),
    ),
    "cli dotted key switches kind": (
        HUBER_FILE,
        "--choice.weighted.stress_weight 5",
        Warns(
            weighted(5.0),
            ignored(
                "choice.huber",
                "the config file",
                "--choice.weighted.stress_weight 5",
                "choice.weighted",
            ),
        ),
    ),
    "cli json switches kind": (
        HUBER_FILE,
        '--choice {"universal": {"huber_delta": 3}}',
        Warns(
            universal(3.0),
            ignored(
                "choice.huber",
                "the config file",
                '--choice {"universal": {"huber_delta": 3}}',
                "choice.universal",
            ),
        ),
    ),
    "cli name of the file's kind keeps the file's keys": (
        HUBER_FILE,
        "--choice huber",
        huber(0.5),
    ),
    "cli key of the file's kind merges": (
        HUBER_FILE,
        "--choice.huber.sub y",
        huber(0.5, SubY),
    ),
    "switch away and back keeps the file's keys": (
        HUBER_FILE,
        "--choice weighted --choice.huber.sub y",
        Warns(
            huber(0.5, SubY),
            ignored(
                "choice.weighted",
                "--choice weighted",
                "--choice.huber.sub y",
                "choice.huber",
            ),
        ),
    ),
    "two kinds on the cli, the later wins": (
        {},
        "--choice.huber.delta 2 --choice.weighted.stress_weight 5",
        Warns(
            weighted(5.0),
            ignored(
                "choice.huber",
                "--choice.huber.delta 2",
                "--choice.weighted.stress_weight 5",
                "choice.weighted",
            ),
        ),
    ),
    "no file, default kind": ({}, "", weighted()),
    "no file, cli key of another kind": ({}, "--choice.huber.delta 2", huber(2.0)),
    "empty section is the default kind": ({"choice": {}}, "", weighted()),
    "empty section with cli key": (
        {"choice": {}},
        "--choice.huber.delta 2",
        huber(2.0),
    ),
    "bare name in the file": ({"choice": "huber"}, "", huber()),
    "bare name in the file, cli key of that kind": (
        {"choice": "huber"},
        "--choice.huber.delta 2",
        huber(2.0),
    ),
    "bare name in the file, cli other kind": (
        {"choice": "huber"},
        "--choice.weighted.stress_weight 5",
        Warns(
            weighted(5.0),
            ignored(
                "choice.huber",
                "the config file",
                "--choice.weighted.stress_weight 5",
                "choice.weighted",
            ),
        ),
    ),
    "= form": (HUBER_FILE, "--choice.huber.delta=2", huber(2.0)),
    "three kinds in a row": (
        HUBER_FILE,
        "--choice weighted --choice universal",
        Warns(
            universal(),
            ignored(
                "choice.huber",
                "the config file",
                "--choice universal",
                "choice.universal",
            ),
            ignored(
                "choice.weighted",
                "--choice weighted",
                "--choice universal",
                "choice.universal",
            ),
        ),
    ),
}

ERRORS = {
    "two kinds in the file": (
        {"choice": {"huber": {}, "weighted": {}}},
        "",
        error(
            "choice is given as several kinds (huber, weighted) in the config "
            "file; keep one"
        ),
    ),
    "two kinds in one json override": (
        {},
        '--choice {"huber": {}, "weighted": {}}',
        error("choice is given as several kinds (huber, weighted) in", "keep one"),
    ),
    "unknown kind in the file": (
        {"choice": {"hubr": {}}},
        "",
        error(
            "unknown config key 'choice.hubr'; did you mean 'choice.huber'?; "
            f"the kinds of choice are {KINDS}"
        ),
    ),
    "unknown bare name in the file": (
        {"choice": "hubr"},
        "",
        error("unknown config key 'choice.hubr'; did you mean 'choice.huber'?"),
    ),
    "unknown kind on the cli": (
        {},
        "--choice hubr",
        error("unknown config key 'choice.hubr'; did you mean 'choice.huber'?"),
    ),
    "unknown dotted kind on the cli": (
        {},
        "--choice.hubr.delta 2",
        error(
            "unknown config key 'choice.hubr.delta'; did you mean 'choice.huber.delta'?"
        ),
    ),
    "key of another kind on the cli is unknown": (
        {},
        "--choice.delta 2",
        error("unknown config key 'choice.delta'; did you mean 'choice.huber.delta'?"),
    ),
    "key of another kind in the file": (
        {"choice": {"weighted": {"delta": 2}}},
        "",
        error(
            "unknown config key 'choice.weighted.delta'; did you mean "
            "'choice.weighted.stress_weight'?"
        ),
    ),
    "the tag is not a key, on the cli": (
        {},
        "--choice.huber.kind huber",
        error("unknown config key 'choice.huber.kind'"),
    ),
    "the tag is not a key, in the file": (
        {"choice": {"huber": {"kind": "weighted"}}},
        "",
        error("choice.huber.kind is not a key; the kind is given by the key 'huber'"),
    ),
    "the tagged form is refused with the key form as the fix": (
        {"choice": {"kind": "huber", "delta": 2}},
        "",
        error(
            "choice.kind is not a key; write the kind as the key the section "
            'sits under, choice: {"huber": {...}}'
        ),
    ),
    "the tagged form on the cli": (
        {},
        '--choice {"kind": "huber"}',
        error("choice.kind is not a key"),
    ),
    "a list at a kinds field": (
        {"choice": [1]},
        "",
        error(
            "choice must be the name of a kind or a mapping under one, one of "
            f"{KINDS}; got [1]"
        ),
    ),
    "a number at a kinds field": (
        {"choice": 3},
        "",
        error(
            "choice must be the name of a kind or a mapping under one, one of "
            f"{KINDS}; got 3"
        ),
    ),
    "a scalar under a kind": (
        {"choice": {"huber": 3}},
        "",
        error("choice.huber must be a mapping of the kind's keys; got 3"),
    ),
    "null at a kinds field that does not admit it": (
        {"choice": None},
        "",
        error(f"choice does not take null; write a kind, one of {KINDS}"),
    ),
    "a dropped section is not validated": (
        {"choice": {"huber": {"nonsense": 1}}},
        "--choice weighted",
        Warns(
            weighted(),
            ignored(
                "choice.huber",
                "the config file",
                "--choice weighted",
                "choice.weighted",
            ),
        ),
    ),
}

NULL = {
    "null override then a key starts the section afresh": (
        HUBER_FILE,
        "--choice null --choice.huber.sub y",
        huber(0.01, SubY),
    ),
    "null under a kind, on the cli": (
        HUBER_FILE,
        "--choice.huber null",
        error(
            "choice.huber does not take null; set the keys wanted under it, "
            "or write another kind"
        ),
    ),
    "null under an unknown kind is an unknown kind": (
        {"choice": {"hubr": None}},
        "",
        error("unknown config key 'choice.hubr'; did you mean 'choice.huber'?"),
    ),
    "null under a kind, in the file": (
        {"choice": {"huber": {"delta": 0.5}, "weighted": None}},
        "",
        error("choice.weighted does not take null"),
    ),
}

NONE_KIND = {
    "none kind, absent": ({}, "", opt(NoChoice)),
    "none kind, by name": ({}, "--opt none", opt(NoChoice)),
    "none kind, cli key": ({}, "--opt.huber.delta 2", opt(Huber, delta=2.0)),
    "none kind, file null": (
        {"opt": None},
        "",
        error(f"opt does not take null; write a kind, one of {OPT_KINDS}"),
    ),
    "none kind, cli null after the file": (
        {"opt": {"huber": {}}},
        "--opt null",
        error(f"opt does not take null; write a kind, one of {OPT_KINDS}"),
    ),
    "none kind, back to none with a warning": (
        {"opt": {"huber": {}}},
        "--opt none",
        Warns(
            opt(NoChoice),
            ignored("opt.huber", "the config file", "--opt none", "opt.none"),
        ),
    ),
}

NESTED = {
    "nested kind by name": (HUBER_FILE, "--choice.huber.sub y", huber(0.5, SubY)),
    "nested kind by key": (
        HUBER_FILE,
        "--choice.huber.sub.y.b 7",
        huber(0.5, SubY, b=7.0),
    ),
    "nested kind in the file, cli key of it": (
        {"choice": {"huber": {"sub": {"y": {"b": 3}}}}},
        "--choice.huber.sub.y.b 7",
        huber(0.01, SubY, b=7.0),
    ),
    "nested switch warns": (
        {"choice": {"huber": {"sub": {"y": {"b": 3}}}}},
        "--choice.huber.sub x",
        Warns(
            huber(0.01, SubX),
            ignored(
                "choice.huber.sub.y",
                "the config file",
                "--choice.huber.sub x",
                "choice.huber.sub.x",
            ),
        ),
    ),
    "nested empty section is its default kind": (
        {"choice": {"huber": {"sub": {}}}},
        "",
        huber(0.01, SubX),
    ),
    "outer switch drops the nested section silently": (
        {"choice": {"huber": {"sub": {"y": {"b": 3}}}}},
        "--choice weighted",
        Warns(
            weighted(),
            ignored(
                "choice.huber",
                "the config file",
                "--choice weighted",
                "choice.weighted",
            ),
        ),
    ),
    "unknown nested kind": (
        {"choice": {"huber": {"sub": {"z": {}}}}},
        "",
        error(
            "unknown config key 'choice.huber.sub.z'",
            "the kinds of choice.huber.sub are x, y",
        ),
    ),
    "unknown key under a nested kind": (
        {"choice": {"huber": {"sub": {"y": {"a": 1}}}}},
        "",
        error(
            "unknown config key 'choice.huber.sub.y.a'; did you mean "
            "'choice.huber.sub.y.b'?"
        ),
    ),
}

COLLECTIONS = {
    "kind inside a dict value, by name": (
        {"per_head": {"h": {"loss": "huber"}}},
        "",
        per_head("h", Huber),
    ),
    "kind inside a dict value, by key": (
        {"per_head": {"h": {"loss": {"huber": {"delta": 2}}}}},
        "",
        per_head("h", Huber, delta=2.0),
    ),
    "kind inside a dict value, json merges the entry": (
        {"per_head": {"h": {"loss": {"huber": {"delta": 2}}}}},
        '--per_head {"h": {"loss": "huber"}}',
        per_head("h", Huber, delta=2.0),
    ),
    "kind inside a dict value, json switches with a warning": (
        {"per_head": {"h": {"loss": {"huber": {"delta": 2}}}}},
        '--per_head {"h": {"loss": "weighted"}}',
        Warns(
            per_head("h", Weighted),
            ignored(
                "per_head.h.loss.huber",
                "the config file",
                '--per_head {"h": {"loss": "weighted"}}',
                "per_head.h.loss.weighted",
            ),
        ),
    ),
    "kind inside a dict value, empty is the default": (
        {"per_head": {"h": {"loss": {}}}},
        "",
        per_head("h", Weighted),
    ),
    "unknown key inside a dict value names the kind": (
        {"per_head": {"h": {"loss": {"huber": {"stress_weight": 1}}}}},
        "",
        error(
            "unknown config key 'per_head.h.loss.huber.stress_weight'; did you mean "
            "'per_head.h.loss.huber.delta'?"
        ),
    ),
    "kind inside a list item": (
        {"layers": [{"loss": "huber"}, {"loss": {"universal": {"huber_delta": 3}}}]},
        "",
        lambda c: layer(0, Huber)(c) and layer(1, Universal, huber_delta=3.0)(c),
    ),
    "list replaced whole by json": (
        {"layers": [{"loss": "huber"}]},
        '--layers [{"loss": {"weighted": {"stress_weight": 5}}}]',
        lambda c: len(c.layers) == 1 and layer(0, Weighted, stress_weight=5.0)(c),
    ),
    "unknown key inside a list item": (
        {"layers": [{"loss": {"huber": {"stress_weight": 1}}}]},
        "",
        error(
            "unknown config key 'layers.0.loss.huber.stress_weight'; did you mean "
            "'layers.0.loss.huber.delta'?"
        ),
    ),
    "list item cannot be addressed by a dotted key": (
        {},
        "--layers.0.loss huber",
        error("unknown config key 'layers.0.loss'"),
    ),
}

OTHER = {
    "keys of other fields are untouched by a switch": (
        {"energy_weight": 3.0, **HUBER_FILE},
        "--choice weighted",
        Warns(
            lambda c: c.energy_weight == 3.0 and weighted()(c),
            ignored(
                "choice.huber",
                "the config file",
                "--choice weighted",
                "choice.weighted",
            ),
        ),
    ),
    "a required key of the kind supplied by the cli": (
        {"choice": {"huber": {}}},
        "--choice.huber.path p",
        lambda c: isinstance(c.choice, HuberRequired) and c.choice.path == "p",
    ),
    "a required key of the kind missing": (
        {"choice": "huber"},
        "",
        Raises(ValidationError, "choice.huber.path", "Field required"),
    ),
}

ROWS = {**SWITCHING, **ERRORS, **NULL, **NONE_KIND, **NESTED, **COLLECTIONS}


def split_cli(cli):
    """`--a.b 1 --c {"d": 2}` as argv: a JSON value keeps its spaces."""
    argv = []
    for chunk in filter(None, re.split(r"\s+(?=--)", cli)):
        argv.extend(chunk.split(" ", 1))
    return argv


def load(tmp_path, root, file_values, cli):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(file_values))
    return root.load(path, cli_overrides=split_cli(cli))


def check(tmp_path, root, file_values, cli, expected):
    if isinstance(expected, Raises):
        with pytest.raises(expected.error_type) as info, warnings.catch_warnings():
            warnings.simplefilter("ignore", ConfigWarning)
            load(tmp_path, root, file_values, cli)
        for fragment in expected.fragments:
            assert fragment in str(info.value), str(info.value)
        return
    predicate, messages = expected, ()
    if isinstance(expected, Warns):
        predicate, messages = expected.predicate, expected.messages
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = load(tmp_path, root, file_values, cli)
    assert [str(w.message) for w in caught] == list(messages)
    assert all(issubclass(w.category, ConfigWarning) for w in caught)
    assert predicate(config), config


@pytest.mark.parametrize("row", ROWS, ids=ROWS)
def test_kind_selection(tmp_path, row):
    check(tmp_path, LossConfig, *ROWS[row])


@pytest.mark.parametrize("row", OTHER, ids=OTHER)
def test_kind_selection_on_other_roots(tmp_path, row):
    file_values, cli, expected = OTHER[row]
    root = RequiredConfig if "required" in row else LossConfig
    check(tmp_path, root, file_values, cli, expected)


@pytest.mark.parametrize(
    ("extension", "text", "shown"),
    [
        (".toml", "choice = 2020-01-01\n", '"2020-01-01"'),
        (".yaml", "choice:\n  huber: 2020-01-01\n", '"2020-01-01"'),
        (".yaml", "choice: !!set {huber: null}\n", "\"{'huber'}\""),
    ],
)
def test_a_value_json_cannot_show_is_still_a_config_error(
    tmp_path, extension, text, shown
):
    path = tmp_path / f"config{extension}"
    path.write_text(text)
    with pytest.raises(ConfigError, match=rf"got {shown}$"):
        LossConfig.load(path)


@pytest.mark.parametrize("extension", [".toml", ".yaml"])
def test_kinds_load_from_every_file_format(tmp_path, extension):
    text = {
        ".toml": "[choice.huber]\ndelta = 0.5\n[choice.huber.sub.y]\nb = 3\n"
        "[opt]\nuniversal = {}\n",
        ".yaml": "choice:\n  huber:\n    delta: 0.5\n    sub:\n      y:\n"
        "        b: 3\nopt: universal\n",
    }[extension]
    path = tmp_path / f"config{extension}"
    path.write_text(text)
    config = LossConfig.load(path)
    assert huber(0.5, SubY, b=3.0)(config) and isinstance(config.opt, Universal)


# ---------------------------------------------------------------------------
# The dicts: kind as key, a fixed point, and only what was set.


@pytest.mark.parametrize(
    "cli",
    [
        "",
        "--choice huber",
        "--choice.huber.sub.y.b 7 --opt universal",
        '--per_head {"h": {"loss": "huber"}} --layers [{"loss": {"universal": {}}}]',
    ],
)
def test_resolved_dict_is_a_fixed_point_and_writes_the_kind_as_key(tmp_path, cli):
    config = load(tmp_path, LossConfig, {}, cli)
    resolved = config.to_resolved_dict()
    kind = type(config.choice).model_fields["kind"].default
    assert list(resolved["choice"]) == [kind]
    assert "kind" not in resolved["choice"][kind]
    reloaded = load(tmp_path, LossConfig, resolved, "")
    assert reloaded.to_resolved_dict() == resolved
    assert reloaded == config


def test_resolved_dict_of_the_defaults():
    assert LossConfig().to_resolved_dict() == {
        "energy_weight": 1.0,
        "choice": {"weighted": {"stress_weight": 0.0}},
        "opt": {"none": {}},
        "heads": {},
        "per_head": {},
        "layers": [],
    }


def test_user_dict_holds_the_kind_and_only_what_was_set(tmp_path):
    config = load(tmp_path, LossConfig, {"choice": "huber"}, "--choice.huber.delta 2")
    assert config.to_user_dict() == {"choice": {"huber": {"delta": 2.0}}}
    # A kind chosen by default is what ran, so it is set.
    assert load(tmp_path, LossConfig, {"choice": {}}, "").to_user_dict() == {
        "choice": {"weighted": {}}
    }
    # Built in code, the tag is an unset default; the kind is still the key.
    assert LossConfig(choice=Huber(delta=2)).to_user_dict() == {
        "choice": {"huber": {"delta": 2.0}}
    }


def test_json_schema_is_produced_in_both_modes():
    for mode in ("validation", "serialization"):
        schema = LossConfig.model_json_schema(mode=mode)
        assert set(schema["properties"]) == set(LossConfig.model_fields), mode


def test_code_sees_the_union_and_may_construct_it_either_way():
    by_key = {"choice": {"huber": {"delta": 2}}}
    by_tag = {"choice": {"kind": "huber", "delta": 2}}
    assert LossConfig(choice=Huber(delta=2)).choice == Huber(delta=2)
    assert LossConfig.model_validate(by_key).choice == Huber(delta=2)
    assert LossConfig.model_validate(by_tag).choice == Huber(delta=2)


# ---------------------------------------------------------------------------
# Schema rules.


class NotASection(BaseModel):
    kind: Literal["plain"] = "plain"


def test_union_shapes_the_contract_cannot_keep_are_rejected_at_class_definition():
    class IntTag(ConfigSection):
        kind: Literal[1] = 1

    class TwoTags(ConfigSection):
        kind: Literal["a", "b"] = "a"

    class NamedLikeTheTag(ConfigSection):
        kind: Literal["kind"] = "kind"

    shapes = {
        r"choice defaults to None, which is not a kind; default to a variant": (
            Choice,
            None,
        ),
        r"choice admits None, which is not a kind; default to a variant, or for none": (
            Choice | None,
            Weighted(),
        ),
        r"choice has variant NamedLikeTheTag whose kind is named 'kind' like the tag": (
            Annotated[Weighted | NamedLikeTheTag, Field(discriminator="kind")],
            Weighted(),
        ),
        r"choice is a union of sections without a discriminator": (
            Weighted | Huber,
            Weighted(),
        ),
        r"choice is a union of sections inside a dict, list or tuple": (
            list[Choice],
            [],
        ),
        r"choice holds NotASection, which is not a ConfigSection": (
            Annotated[Weighted | NotASection, Field(discriminator="kind")],
            Weighted(),
        ),
        r"choice has variant IntTag whose kind must be a Literal of exactly one": (
            Annotated[Weighted | IntTag, Field(discriminator="kind")],
            Weighted(),
        ),
        r"choice has variant TwoTags whose kind must be a Literal of exactly one": (
            Annotated[Weighted | TwoTags, Field(discriminator="kind")],
            Weighted(),
        ),
        r"choice has a default that is not one of its variants; write e.g. Weighted": (
            Choice,
            Plain(),
        ),
        r"choice has a default_factory; write the default as an instance": (
            Weighted | Huber,
            Field(discriminator="kind", default_factory=Weighted),
        ),
        r"choice mixes its kinds with dict; a kinds field holds its variants only": (
            Choice | dict[str, int],
            Weighted(),
        ),
    }
    for message, (annotation, default) in shapes.items():
        with pytest.raises(TypeError, match=message):
            type(
                "Bad",
                (ConfigSection,),
                {"__annotations__": {"choice": annotation}, "choice": default},
            )


def test_a_required_kinds_field_is_accepted():
    factory_calls = []

    class Required(ReforgeBaseConfig):
        choice: Choice
        anything: Any = None
        made: int = Field(default_factory=lambda: factory_calls.append(1) or 1)

    assert factory_calls == [], "the schema check must not run default factories"
    with pytest.raises(ConfigError, match="choice needs a kind; one of"):
        Required.load(cli_overrides=["--choice", "{}"])
    assert isinstance(Required.load(cli_overrides=["--choice", "huber"]).choice, Huber)


def test_a_kind_named_like_a_class_still_reads_as_a_kind_in_error_paths():
    class Root(ConfigSection):
        kind: Literal["Root"] = "Root"
        q: int = 0

    class Config(ReforgeBaseConfig):
        choice: Annotated[Weighted | Root, Field(discriminator="kind")] = Weighted()

    with pytest.raises(
        ConfigError, match=r"'choice\.Root\.qq'; did you mean 'choice\.Root\.q'"
    ):
        Config.load(cli_overrides=["--choice", '{"Root": {"qq": 1}}'])


def test_a_collection_key_named_like_the_element_class_stays_in_error_paths():
    with pytest.raises(
        ConfigError, match=r"'heads\.Plain\.q'; did you mean 'heads\.Plain\.p'"
    ):
        LossConfig.load(cli_overrides=["--heads", '{"Plain": {"q": 1}}'])


def test_warning_can_be_turned_into_an_error(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        with pytest.raises(ConfigWarning, match=r"choice\.huber from the config file"):
            load(tmp_path, LossConfig, HUBER_FILE, "--choice weighted")
