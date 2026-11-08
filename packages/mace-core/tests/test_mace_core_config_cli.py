"""Overrides update documents before one schema validation."""

import pytest
from mace_core.config import (
    ConfigError,
    apply_overrides,
    read_config_file,
)
from mace_core_demo import FILE_VALUES, DemoConfig, Huber, error_locations, write_config
from pydantic import ValidationError

#: A warning the test did not ask for is a failure.
pytestmark = pytest.mark.filterwarnings("error")


def load(tmp_path, overrides, values=FILE_VALUES, root=DemoConfig):
    document = read_config_file(write_config(tmp_path, ".json", values))
    return root.from_dict(apply_overrides(document, overrides))


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
def test_a_parent_that_is_not_a_mapping_is_rejected(parent):
    with pytest.raises(ConfigError, match=r"cannot set a\.b"):
        apply_overrides({"a": parent}, {"a.b": 2})


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
    overrides = {
        "stage_two.start_epoch": "5",
        "model.radial.cutoff": "4",
        "model": {"num_interactions": 3},
    }
    assert apply_overrides(document, overrides) == {
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


# ---------------------------------------------------------------------------
# Composed with the base: the override beats the file, which beats the
# defaults, and every schema error is pydantic's at the path the override named.


def test_an_override_beats_the_file_which_beats_the_defaults(tmp_path):
    config = load(tmp_path, {"model.num_interactions": 3})
    assert config.model.num_interactions == 3  # the override beats the file's 4
    assert config.model.radial.cutoff == 4.5  # the file's other values survive
    assert config.model.radial.num_bessel == 8  # defaults fill the rest
    assert load(tmp_path, {}) == DemoConfig.from_dict(FILE_VALUES)
    # A section the file left alone is created on the way and validated like
    # any other section.
    assert load(tmp_path, {"stage_two.start_epoch": 50}).stage_two.start_epoch == 50
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, {"stage_two.start": 50})
    assert error_locations(excinfo) == ["stage_two.start"]


def test_an_override_writes_into_a_kinds_field(tmp_path):
    config = load(tmp_path, {"loss.delta": 2}, {"loss": {"kind": "huber"}})
    assert config.loss == Huber(kind="huber", delta=2.0)
    config = load(tmp_path, {"loss": {"kind": "huber", "delta": 2}}, {})
    assert config.loss == Huber(kind="huber", delta=2.0)
    assert config.to_user_dict() == {"loss": {"kind": "huber", "delta": 2.0}}


def test_an_override_that_changes_the_kind_leaves_the_old_settings_to_pydantic(
    tmp_path,
):
    # The file's `delta` stays in the dict and is an unknown key of the new kind.
    huber = {"loss": {"kind": "huber", "delta": 0.5}}
    with pytest.raises(ValidationError) as excinfo:
        load(tmp_path, {"loss.kind": "weighted"}, huber)
    assert error_locations(excinfo) == ["loss.weighted.delta"]
