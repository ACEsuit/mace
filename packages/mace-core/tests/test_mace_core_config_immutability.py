"""Validated configurations stay valid across container access and copies."""

import copy
import pickle

import pytest
from mace_core.config.base import BaseConfig
from mace_core.config.model import ModelConfig
from mace_core.config.resolved import ResolvedConfig
from pydantic import Field, ValidationError


class ContainerConfig(BaseConfig):
    entries: dict[str, list[int]] = Field(default_factory=lambda: {"sizes": [2, 4]})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config.entries.__setitem__("sizes", []),
        lambda config: config.entries.__delitem__("sizes"),
        lambda config: config.entries.update(other=[]),
        lambda config: config.entries.setdefault("other", []),
        lambda config: config.entries.pop("sizes"),
        lambda config: config.entries.popitem(),
        lambda config: config.entries.clear(),
        lambda config: config.entries.__ior__({"other": []}),
        lambda config: config.entries["sizes"].__setitem__(0, -1),
        lambda config: config.entries["sizes"].__delitem__(0),
        lambda config: config.entries["sizes"].append(8),
        lambda config: config.entries["sizes"].extend([8]),
        lambda config: config.entries["sizes"].insert(0, 8),
        lambda config: config.entries["sizes"].pop(),
        lambda config: config.entries["sizes"].remove(2),
        lambda config: config.entries["sizes"].reverse(),
        lambda config: config.entries["sizes"].sort(),
        lambda config: config.entries["sizes"].clear(),
        lambda config: config.entries["sizes"].__iadd__([8]),
        lambda config: config.entries["sizes"].__imul__(2),
    ],
)
def test_nested_container_mutations_are_refused(mutate):
    config = ContainerConfig()
    with pytest.raises(TypeError, match="immutable"):
        mutate(config)
    assert config.entries == {"sizes": [2, 4]}


def test_input_and_export_are_independent_editable_values():
    document = {"entries": {"sizes": [1, 3]}}
    config = ContainerConfig.from_dict(document)
    document["entries"]["sizes"].append(5)
    exported = config.to_resolved_dict()
    exported["entries"]["sizes"].append(7)
    assert config.entries == {"sizes": [1, 3]}
    assert ContainerConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda c: pickle.loads(pickle.dumps(c))]
)
def test_worker_serialization_and_copy_preserve_immutability(clone):
    config = clone(ContainerConfig())
    with pytest.raises(TypeError, match="immutable"):
        config.entries["sizes"].append(8)


@pytest.mark.parametrize("deep", [False, True])
def test_copy_validates_updates_and_unknown_keys(deep):
    config = ModelConfig(num_channels=4)
    for update in ({"num_channels": -1}, {"num_chanels": 8}):
        with pytest.raises(ValidationError):
            config.model_copy(update=update, deep=deep)
    changed = config.model_copy(update={"num_channels": 8}, deep=deep)
    assert changed.num_channels == 8 and config.num_channels == 4
    assert changed.model_fields_set == {"num_channels"}


def test_root_copy_keeps_nested_explicit_field_provenance():
    config = ResolvedConfig.model_validate(
        {"model": {"num_channels": 4}, "runtime": {"seed": 23}}
    )
    changed = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"seed": 24})}
    )
    assert changed.to_user_dict() == {
        "model": {"num_channels": 4},
        "runtime": {"seed": 24},
    }
    with pytest.raises(ValidationError):
        config.model_copy(update={"model": {"num_channels": -1}})


def test_copy_validates_cross_section_constraints():
    config = ResolvedConfig()
    with pytest.raises(ValidationError):
        config.model_copy(
            update={
                "finetune": {"foundation_model": None},
                "data": {"heads": {"default": {"e0s": {"kind": "foundation"}}}},
            }
        )
