"""The demo schema the config tests share: two levels of nesting, a list, an
optional, a free dict, a kinds field; one config as a dict; and the helpers
that write it out in each format and read a `ValidationError`'s locations."""

import json
from typing import Annotated, Any, Literal

import yaml
from mace_core.config import BaseConfig, ConfigSection
from pydantic import Field


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


class Weighted(ConfigSection):
    kind: Literal["weighted"] = "weighted"
    stress_weight: float = 0.0


class Huber(ConfigSection):
    kind: Literal["huber"] = "huber"
    delta: float = 0.01


class DemoConfig(BaseConfig):
    name: str = "mace"
    seed: int = 123
    default_dtype: str = "float64"
    model: ModelSection = ModelSection()
    data: DataSection = DataSection()
    #: A section left at its defaults unless a file or the CLI writes into it.
    stage_two: StageTwoSection = StageTwoSection()
    loss: Annotated[Weighted | Huber, Field(discriminator="kind")] = Weighted()
    extra: dict[str, Any] = Field(default_factory=dict)


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


def write_config(tmp_path, extension=".json", values=FILE_VALUES, name="config"):
    path = tmp_path / f"{name}{extension}"
    path.write_text(dump(values, extension), encoding="utf-8")
    return path


def error_locations(excinfo):
    """The dotted location of every error a `ValidationError` carries."""
    return [".".join(map(str, error["loc"])) for error in excinfo.value.errors()]
