"""A field of several kinds of section (a discriminated union on `kind`) is an
ordinary pydantic feature under the one-file contract: a config
writes it in the tagged form (`loss: {kind: huber, delta: 0.1}`), pydantic
picks the variant by the tag and reports every error under it
(`loss.huber.delta`), and both exports write the tag back so the dict loads to
the same config. Nothing in the loader knows about kinds."""

import json
from typing import Annotated, Literal

import pytest
from mace_core.config import ConfigSection, ReforgeBaseConfig
from pydantic import Field, ValidationError

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


def huber(delta=0.01, sub=SubX, **sub_fields):
    return lambda c: (
        isinstance(c.choice, Huber)
        and c.choice.delta == delta
        and isinstance(c.choice.sub, sub)
        and all(getattr(c.choice.sub, k) == v for k, v in sub_fields.items())
    )


def load(tmp_path, root, file_values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(file_values))
    return root.load(path)


def error_locations(excinfo):
    """The dotted location of every error a `ValidationError` carries."""
    return [".".join(map(str, error["loc"])) for error in excinfo.value.errors()]


# ---------------------------------------------------------------------------
# Loading: the tagged form, and pydantic's errors under the tag.


@pytest.mark.parametrize("extension", [".toml", ".yaml", ".json"])
def test_kinds_load_from_every_file_format(tmp_path, extension):
    text = {
        ".toml": '[choice]\nkind = "huber"\ndelta = 0.5\n[choice.sub]\nkind = "y"\n'
        'b = 3\n[opt]\nkind = "universal"\n',
        ".yaml": "choice:\n  kind: huber\n  delta: 0.5\n  sub:\n    kind: y\n"
        "    b: 3\nopt:\n  kind: universal\n",
        ".json": json.dumps(
            {
                "choice": {"kind": "huber", "delta": 0.5, "sub": {"kind": "y", "b": 3}},
                "opt": {"kind": "universal"},
            }
        ),
    }[extension]
    path = tmp_path / f"config{extension}"
    path.write_text(text)
    config = LossConfig.load(path)
    assert huber(0.5, SubY, b=3.0)(config) and isinstance(config.opt, Universal)


@pytest.mark.parametrize("value", [{"delta": 0.5}, {}], ids=["settings", "empty"])
def test_a_kinds_field_without_its_tag_is_a_validation_error(tmp_path, value):
    # Even though the field has a default: pydantic needs the tag to pick.
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, LossConfig, {"choice": value})
    assert error_locations(excinfo) == ["choice"]
    assert excinfo.value.errors()[0]["type"] == "union_tag_not_found"


def test_a_wrong_tag_is_a_validation_error_listing_the_kinds(tmp_path):
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, LossConfig, {"choice": {"kind": "hubr"}})
    assert error_locations(excinfo) == ["choice"]
    (error,) = excinfo.value.errors()
    assert error["type"] == "union_tag_invalid"
    assert all(kind in error["msg"] for kind in ("weighted", "huber", "universal"))


@pytest.mark.parametrize(
    ("file_values", "location"),
    [
        (
            {"choice": {"kind": "huber", "stress_weight": 1}},
            "choice.huber.stress_weight",
        ),
        (
            {"choice": {"kind": "huber", "sub": {"kind": "y", "a": 1}}},
            "choice.huber.sub.y.a",
        ),
        (
            {"per_head": {"h": {"loss": {"kind": "huber", "stress_weight": 1}}}},
            "per_head.h.loss.huber.stress_weight",
        ),
        (
            {"layers": [{"loss": {"kind": "huber", "stress_weight": 1}}]},
            "layers.0.loss.huber.stress_weight",
        ),
    ],
    ids=["top", "nested", "dict value", "list item"],
)
def test_unknown_keys_under_a_kind_are_reported_under_the_tag(
    tmp_path, file_values, location
):
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, LossConfig, file_values)
    assert error_locations(excinfo) == [location]
    assert excinfo.value.errors()[0]["type"] == "extra_forbidden"


def test_a_required_key_of_the_kind_is_reported_under_the_tag(tmp_path):
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, RequiredConfig, {"choice": {"kind": "huber"}})
    assert error_locations(excinfo) == ["choice.huber.path"]
    assert excinfo.value.errors()[0]["type"] == "missing"
    config = load(tmp_path, RequiredConfig, {"choice": {"kind": "huber", "path": "p"}})
    assert isinstance(config.choice, HuberRequired) and config.choice.path == "p"


# ---------------------------------------------------------------------------
# The dicts: the tag is written back, a fixed point, and only what was set.


@pytest.mark.parametrize(
    "file_values",
    [
        {},
        {"choice": {"kind": "huber"}},
        {
            "choice": {"kind": "huber", "sub": {"kind": "y", "b": 7}},
            "opt": {"kind": "universal"},
        },
        {
            "per_head": {"h": {"loss": {"kind": "huber"}}},
            "layers": [{"loss": {"kind": "universal"}}],
        },
    ],
    ids=["defaults", "huber", "nested", "collections"],
)
def test_resolved_dict_is_a_fixed_point_and_writes_the_tag(tmp_path, file_values):
    config = load(tmp_path, LossConfig, file_values)
    resolved = config.to_resolved_dict()
    assert resolved["choice"]["kind"] == config.choice.kind
    reloaded = load(tmp_path, LossConfig, resolved)
    assert reloaded.to_resolved_dict() == resolved
    assert reloaded == config


def test_resolved_dict_of_the_defaults():
    assert LossConfig().to_resolved_dict() == {
        "energy_weight": 1.0,
        "choice": {"kind": "weighted", "stress_weight": 0.0},
        "opt": {"kind": "none"},
        "per_head": {},
        "layers": [],
    }


def test_user_dict_holds_the_tag_and_only_what_was_set(tmp_path):
    config = load(tmp_path, LossConfig, {"choice": {"kind": "huber", "delta": 2.0}})
    assert config.to_user_dict() == {"choice": {"kind": "huber", "delta": 2.0}}
    assert load(tmp_path, LossConfig, config.to_user_dict()) == config
    # The documented caveat: built in code without its tag, a variant exports
    # without `kind` and the dict does not load back; pass the tag, or use
    # `to_resolved_dict`.
    assert LossConfig(choice=Huber(delta=2)).to_user_dict() == {
        "choice": {"delta": 2.0}
    }
    tagged = LossConfig(choice=Huber(kind="huber", delta=2))
    assert tagged.to_user_dict() == {"choice": {"kind": "huber", "delta": 2.0}}
    assert LossConfig(choice=Huber(delta=2)).to_resolved_dict()["choice"]["kind"] == (
        "huber"
    )


def test_json_schema_is_produced_in_both_modes():
    for mode in ("validation", "serialization"):
        schema = LossConfig.model_json_schema(mode=mode)
        assert set(schema["properties"]) == set(LossConfig.model_fields), mode


def test_a_variant_class_as_a_plain_field_keeps_kind_as_a_key(tmp_path):
    class Reuse(ReforgeBaseConfig):
        direct: Huber = Huber()

    resolved = {
        "direct": {"kind": "huber", "delta": 0.01, "sub": {"kind": "x", "a": 1.0}}
    }
    assert Reuse().to_resolved_dict() == resolved
    assert load(tmp_path, Reuse, resolved).to_resolved_dict() == resolved
    config = load(tmp_path, Reuse, {"direct": {"kind": "huber", "sub": {"kind": "y"}}})
    assert config.direct == Huber(sub=SubY())
    with pytest.raises(ValidationError, match=r"direct\.kind"):
        load(tmp_path, Reuse, {"direct": {"kind": "weighted"}})
