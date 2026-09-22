"""A field of several kinds of section (a discriminated union) under the file
and dotted-override contract. A config writes each kind's settings under the
kind's name (`loss: {huber: {delta: 0.1}}`, `--loss.huber.delta 0.1`); the
settings of every kind are kept across files and overrides. Which kind runs is
selected by `kind: huber`, `--loss.kind huber` or the bare name `--loss huber`;
a single kind key selects itself; the last selection wins. An override that
changed nothing about the config that runs warns, a file never warns; two kinds
with no selection is an error. `kind` is the tag only under a kinds field: a
plain section field of the same class keeps it as a key. Code sees the union."""

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

#: A warning the test did not ask for is a failure.
pytestmark = pytest.mark.filterwarnings("error")

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


def no_effect(source, field, kind, running):
    return f"{source}: {field}.{kind} has no effect, {field} runs {running}"


def overridden(source, field, running):
    return f"{source} is overridden: {field} runs {running}"


def needs_kind(field, *kinds):
    """The kinds error up to the sources; the first kind is the example."""
    choices = " or ".join(f"kind: {kind}" for kind in kinds)
    return (
        f"{field} needs a kind; write {choices} in a file, or pass --{field} {kinds[0]}"
    )


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
    "cli name selects; the file's settings of the other kind stay unused": (
        HUBER_FILE,
        "--choice weighted",
        weighted(),
    ),
    "cli key of a second kind without a selection is an error": (
        HUBER_FILE,
        "--choice.weighted.stress_weight 5",
        error(
            needs_kind("choice", "huber", "weighted"),
            "; huber from ",
            "config.json, weighted from --choice.weighted.stress_weight 5",
        ),
    ),
    "cli json of a second kind without a selection is an error": (
        HUBER_FILE,
        '--choice {"universal": {"huber_delta": 3}}',
        error(
            needs_kind("choice", "huber", "universal"),
            'config.json, universal from --choice {"universal": {"huber_delta": 3}}',
        ),
    ),
    "the tag alone selects on the cli": ({}, '--choice {"kind": "huber"}', huber()),
    "the tag beside the settings selects in the file": (
        {"choice": {"kind": "weighted", "huber": {"delta": 0.5}, "weighted": {}}},
        "",
        weighted(),
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
    "a selection, then a key of another kind: the key warns": (
        HUBER_FILE,
        "--choice weighted --choice.huber.sub y",
        Warns(
            weighted(),
            no_effect("--choice.huber.sub y", "choice", "huber", "weighted"),
        ),
    ),
    "keys of two kinds on the cli without a selection is an error": (
        {},
        "--choice.huber.delta 2 --choice.weighted.stress_weight 5",
        error(
            needs_kind("choice", "huber", "weighted")
            + "; huber from --choice.huber.delta 2, "
            "weighted from --choice.weighted.stress_weight 5"
        ),
    ),
    "keys of two kinds on the cli, then a selection": (
        {},
        "--choice.huber.delta 2 --choice.weighted.stress_weight 5 --choice huber",
        Warns(
            huber(2.0),
            no_effect(
                "--choice.weighted.stress_weight 5", "choice", "weighted", "huber"
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
    "bare name in the file, cli key of another kind warns": (
        {"choice": "huber"},
        "--choice.weighted.stress_weight 5",
        Warns(
            huber(),
            no_effect(
                "--choice.weighted.stress_weight 5", "choice", "weighted", "huber"
            ),
        ),
    ),
    "= form": (HUBER_FILE, "--choice.huber.delta=2", huber(2.0)),
    "three selections in a row: the last runs, the lost cli one warns": (
        HUBER_FILE,
        "--choice weighted --choice universal",
        Warns(universal(), overridden("--choice weighted", "choice", "universal")),
    ),
}

ERRORS = {
    "two kinds in the file": (
        {"choice": {"huber": {}, "weighted": {}}},
        "",
        error(
            needs_kind("choice", "huber", "weighted"),
            "; huber from ",
            "config.json, weighted from ",
        ),
    ),
    "two kinds in one json override": (
        {},
        '--choice {"huber": {}, "weighted": {}}',
        error(
            needs_kind("choice", "huber", "weighted")
            + '; huber from --choice {"huber": {}, "weighted": {}}, '
            'weighted from --choice {"huber": {}, "weighted": {}}'
        ),
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
            "unknown config key 'choice.hubr'; did you mean 'choice.huber'?; "
            f"the kinds of choice are {KINDS}"
        ),
    ),
    "a setting beside the kinds is unknown and the kinds are listed": (
        {},
        "--choice.delta 2",
        error(f"unknown config key 'choice.delta'; the kinds of choice are {KINDS}"),
    ),
    "key of another kind in the file": (
        {"choice": {"weighted": {"delta": 2}}},
        "",
        error("unknown config key 'choice.weighted.delta'"),
    ),
    "the tag is not a key, on the cli": (
        {},
        "--choice.huber.kind huber",
        error(
            "unknown config key 'choice.huber.kind'; the key huber already names "
            "the kind"
        ),
    ),
    "the tag is not a key, in the file": (
        {"choice": {"huber": {"kind": "weighted"}}},
        "",
        error(
            "unknown config key 'choice.huber.kind'; the key huber already names "
            "the kind"
        ),
    ),
    "the flat form fails on the setting beside the tag": (
        {"choice": {"kind": "huber", "delta": 2}},
        "",
        error(f"unknown config key 'choice.delta'; the kinds of choice are {KINDS}"),
    ),
    "a selection that is not a kind": (
        {},
        "--choice.kind hubr",
        error(
            "unknown config key 'choice.hubr'; did you mean 'choice.huber'?; "
            f"the kinds of choice are {KINDS}"
        ),
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
        error("choice.huber must be a mapping of its keys; got 3"),
    ),
    "null at a kinds field": (
        {"choice": None},
        "",
        error(
            "choice must be the name of a kind or a mapping under one, one of "
            f"{KINDS}; got null"
        ),
    ),
    "settings of a kind that does not run are still checked": (
        {"choice": {"huber": {"nonsense": 1}}},
        "--choice weighted",
        error("unknown config key 'choice.huber.nonsense'"),
    ),
}

NULL = {
    "null at a kinds field is an error even when a key follows": (
        HUBER_FILE,
        "--choice null --choice.huber.sub y",
        error(
            "choice must be the name of a kind or a mapping under one, one of "
            f"{KINDS}; got null"
        ),
    ),
    "null under a kind, on the cli": (
        HUBER_FILE,
        "--choice.huber null",
        error("choice.huber must be a mapping of its keys; got null"),
    ),
    "null under an unknown kind is an unknown kind": (
        {"choice": {"hubr": None}},
        "",
        error("unknown config key 'choice.hubr'; did you mean 'choice.huber'?"),
    ),
    "null under a kind, in the file": (
        {"choice": {"huber": {"delta": 0.5}, "weighted": None}},
        "",
        error("choice.weighted must be a mapping of its keys; got null"),
    ),
}

NONE_KIND = {
    "none kind, absent": ({}, "", opt(NoChoice)),
    "none kind, by name": ({}, "--opt none", opt(NoChoice)),
    "none kind, cli key": ({}, "--opt.huber.delta 2", opt(Huber, delta=2.0)),
    "none kind, file null": (
        {"opt": None},
        "",
        error(
            "opt must be the name of a kind or a mapping under one, one of "
            f"{OPT_KINDS}; got null"
        ),
    ),
    "none kind, cli null after the file": (
        {"opt": {"huber": {}}},
        "--opt null",
        error(
            "opt must be the name of a kind or a mapping under one, one of "
            f"{OPT_KINDS}; got null"
        ),
    ),
    "none kind, back to none": ({"opt": {"huber": {}}}, "--opt none", opt(NoChoice)),
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
    "nested selection runs; the file's settings of the other kind stay unused": (
        {"choice": {"huber": {"sub": {"y": {"b": 3}}}}},
        "--choice.huber.sub x",
        huber(0.01, SubX),
    ),
    "nested key of another kind on the cli warns": (
        {"choice": {"huber": {"sub": {"kind": "y", "y": {"b": 3}}}}},
        "--choice.huber.sub.x.a 5",
        Warns(
            huber(0.01, SubY, b=3.0),
            no_effect("--choice.huber.sub.x.a 5", "choice.huber.sub", "x", "y"),
        ),
    ),
    "nested empty section is its default kind": (
        {"choice": {"huber": {"sub": {}}}},
        "",
        huber(0.01, SubX),
    ),
    "outer selection: the nested settings stay unused": (
        {"choice": {"huber": {"sub": {"y": {"b": 3}}}}},
        "--choice weighted",
        weighted(),
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
        error("unknown config key 'choice.huber.sub.y.a'"),
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
    "kind inside a dict value, json selects another kind": (
        {"per_head": {"h": {"loss": {"huber": {"delta": 2}}}}},
        '--per_head {"h": {"loss": "weighted"}}',
        per_head("h", Weighted),
    ),
    "kind inside a dict value, dotted key of another kind warns": (
        {"per_head": {"h": {"loss": {"kind": "huber", "huber": {"delta": 2}}}}},
        "--per_head.h.loss.weighted.stress_weight 5",
        Warns(
            per_head("h", Huber, delta=2.0),
            no_effect(
                "--per_head.h.loss.weighted.stress_weight 5",
                "per_head.h.loss",
                "weighted",
                "huber",
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
        error("unknown config key 'per_head.h.loss.huber.stress_weight'"),
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
        error("unknown config key 'layers.0.loss.huber.stress_weight'"),
    ),
    "unknown kind inside a list item": (
        {"layers": [{"loss": {"hubr": {}}}]},
        "",
        error(
            "unknown config key 'layers.0.loss.hubr'; did you mean "
            f"'layers.0.loss.huber'?; the kinds of layers.0.loss are {KINDS}"
        ),
    ),
    "list item cannot be addressed by a dotted key": (
        {},
        "--layers.0.loss huber",
        error("unknown config key 'layers.0'; layers is written whole"),
    ),
}

WARNINGS = {
    "a selection lost to a later one warns, the settings stay": (
        {},
        "--choice huber --choice.huber.delta 2 --choice weighted",
        Warns(
            weighted(),
            overridden("--choice huber", "choice", "weighted"),
            no_effect("--choice.huber.delta 2", "choice", "huber", "weighted"),
        ),
    ),
    "a selection by the tag lost to a later one warns": (
        {},
        "--choice.kind huber --choice weighted",
        Warns(weighted(), overridden("--choice.kind huber", "choice", "weighted")),
    ),
    "a json override that selects one kind and tunes another warns once": (
        {},
        '--choice {"kind": "weighted", "huber": {"delta": 1, "sub": "y"}}',
        Warns(
            weighted(),
            '--choice {"kind": "weighted", "huber": {"delta": 1, "sub": "y"}}: '
            "choice.huber has no effect, choice runs weighted",
        ),
    ),
    "the same selection twice: the first is overridden": (
        {},
        "--choice huber --choice huber",
        Warns(huber(), overridden("--choice huber", "choice", "huber")),
    ),
    "a setting of the running kind is silent": (
        {"choice": "huber"},
        "--choice.huber.delta 3",
        huber(3.0),
    ),
    "an empty mapping before a setting is silent": (
        {},
        "--choice {} --choice.huber.delta 1",
        huber(1.0),
    ),
    "a file tuning several kinds never warns": (
        {"choice": {"kind": "huber", "huber": {"delta": 0.5}, "weighted": {}}},
        "--energy_weight 2",
        lambda c: c.energy_weight == 2.0 and huber(0.5)(c),
    ),
}

OTHER = {
    "keys of other fields are untouched by a selection": (
        {"energy_weight": 3.0, **HUBER_FILE},
        "--choice weighted",
        lambda c: c.energy_weight == 3.0 and weighted()(c),
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

ROWS = {
    **SWITCHING,
    **ERRORS,
    **NULL,
    **NONE_KIND,
    **NESTED,
    **COLLECTIONS,
    **WARNINGS,
}


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


def test_the_empty_mapping_keeps_the_default_instance_with_its_settings(tmp_path):
    class Tuned(ReforgeBaseConfig):
        choice: Choice = Weighted(stress_weight=2.0)
        made: Choice = Field(default_factory=lambda: Huber(delta=9.0, sub=SubY(b=4.0)))

    config = load(tmp_path, Tuned, {"choice": {}, "made": {}}, "")
    assert config == Tuned()
    assert config.to_user_dict() == {
        "choice": {"weighted": {"stress_weight": 2.0}},
        "made": {"huber": {"delta": 9.0, "sub": {"y": {"b": 4.0}}}},
    }
    assert load(tmp_path, Tuned, config.to_user_dict(), "") == config
    # A kind key with no settings runs the class defaults, not the instance's.
    assert load(tmp_path, Tuned, {"choice": {"weighted": {}}}, "").choice == Weighted()
    tuned = load(tmp_path, Tuned, {"choice": {}}, "--choice.weighted.stress_weight 3")
    assert tuned.choice == Weighted(stress_weight=3.0)


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


def test_a_variant_class_as_a_plain_field_keeps_kind_as_a_key(tmp_path):
    class Reuse(ReforgeBaseConfig):
        direct: Huber = Huber()

    resolved = {"direct": {"kind": "huber", "delta": 0.01, "sub": {"x": {"a": 1.0}}}}
    assert Reuse().to_resolved_dict() == resolved
    assert load(tmp_path, Reuse, resolved, "").to_resolved_dict() == resolved
    config = load(tmp_path, Reuse, {}, "--direct.kind huber --direct.sub y")
    assert config.direct == Huber(sub=SubY())
    with pytest.raises(ValidationError, match=r"direct\.kind"):
        load(tmp_path, Reuse, {}, "--direct.kind weighted")


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

    class Colliding(ConfigSection):
        """In the flat form `{kind: c, weighted: 9}`, the field would read as
        the settings of the sibling kind."""

        kind: Literal["c"] = "c"
        weighted: float = 3.0

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
        r"choice has variant Colliding with a field named like the kind weighted; ": (
            Annotated[Weighted | Colliding, Field(discriminator="kind")],
            Weighted(),
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
    with pytest.raises(ConfigError) as excinfo:
        Required.load(cli_overrides=["--choice", "{}"])
    # Nothing wrote a kind, so no source is named.
    assert str(excinfo.value) == (
        "choice needs a kind; write kind: weighted or kind: huber or kind: universal "
        "in a file, or pass --choice weighted"
    )
    assert isinstance(Required.load(cli_overrides=["--choice", "huber"]).choice, Huber)


def test_a_default_factory_that_returns_no_variant_is_reported(tmp_path):
    class Broken(ReforgeBaseConfig):
        # The wrong result is the point; ty sees only the factory type.
        choice: Choice = Field(default_factory=lambda: None)  # ty: ignore[invalid-assignment]

    with pytest.raises(
        TypeError, match=r"Broken\.choice has a default factory whose result is not"
    ):
        load(tmp_path, Broken, {"choice": {}}, "")


def test_several_kinds_inside_a_list_item_names_only_the_file_fix(tmp_path):
    file_values = {"layers": [{"loss": {"huber": {}, "weighted": {}}}]}
    with pytest.raises(ConfigError) as excinfo:
        load(tmp_path, LossConfig, file_values, "")
    path = tmp_path / "config.json"
    assert str(excinfo.value) == (
        "layers.0.loss needs a kind; write kind: huber or kind: weighted in a file; "
        f"huber from {path}, weighted from {path}"
    )


def test_several_files_each_writing_one_kind_are_named(tmp_path):
    defaults = tmp_path / "defaults.json"
    defaults.write_text(json.dumps({"choice": {"huber": {"delta": 2}}}))
    user = tmp_path / "user.json"
    user.write_text(json.dumps({"choice": {"weighted": {}}}))
    with pytest.raises(ConfigError) as excinfo:
        LossConfig.load([defaults, user])
    assert str(excinfo.value) == (
        "choice needs a kind; write kind: huber or kind: weighted in a file, or "
        f"pass --choice huber; huber from {defaults}, weighted from {user}"
    )


def test_the_tag_inside_a_kind_is_an_error_under_model_validate():
    # Not a silent switch to the other kind: the tag is not a key under a kind.
    message = r"choice\.huber\.kind is not a key; huber already names the kind"
    with pytest.raises(ValidationError, match=message):
        LossConfig.model_validate({"choice": {"huber": {"kind": "weighted"}}})
    with pytest.raises(ValidationError, match=message):
        LossConfig.model_validate(
            {"choice": {"kind": "huber", "huber": {"kind": "huber", "delta": 2}}}
        )


def test_a_subclass_of_a_variant_exports_under_the_variant_kind(tmp_path):
    class SubHuber(Huber):
        extra_knob: int = 5

    config = LossConfig(choice=SubHuber(delta=3.0))
    resolved = config.to_resolved_dict()
    assert resolved["choice"] == {"huber": {"delta": 3.0, "sub": {"x": {"a": 1.0}}}}
    assert config.to_user_dict() == {"choice": {"huber": {"delta": 3.0}}}
    reloaded = load(tmp_path, LossConfig, resolved, "")
    assert reloaded.choice == Huber(delta=3.0)
    assert reloaded.to_resolved_dict() == resolved


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
        ConfigError, match=r"'heads\.Plain\.pp'; did you mean 'heads\.Plain\.p'"
    ):
        LossConfig.load(cli_overrides=["--heads", '{"Plain": {"pp": 1}}'])


def test_warning_can_be_turned_into_an_error(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConfigWarning)
        with pytest.raises(
            ConfigWarning, match=r"choice\.weighted has no effect, choice runs huber"
        ):
            load(
                tmp_path,
                LossConfig,
                {"choice": "huber"},
                "--choice.weighted.stress_weight 5",
            )
