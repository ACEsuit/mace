"""The deployment contract stored next to an unchanged V1 checkpoint pair."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from mace_core.graph.padding import PaddingPolicy
from pydantic import ConfigDict, Field, StrictBool, TypeAdapter
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class DeployManifest:
    """Options for the MD runtime. Distances and cutoffs are in Å."""

    VERSION: ClassVar[str] = "1.1"
    schema_version: str
    target: Literal["lammps-mliap", "lammps-native", "ase", "openmm"]
    head: Annotated[str, Field(min_length=1)]
    dtype: Literal["float32", "float64"]
    halo_exchange: StrictBool
    backend: Annotated[str, Field(min_length=1)] | None = None
    compile_regions: StrictBool = True
    pad_spec: PaddingPolicy | None = None
    species: tuple[Annotated[int, Field(strict=True, gt=0)], ...] = ()
    r_max: Annotated[float, Field(ge=0)] = 0.0
    num_interactions: Annotated[int, Field(strict=True, ge=0)] = 0
    receptive_field: Annotated[float, Field(ge=0)] = 0.0

    def __post_init__(self):
        if self.schema_version != self.VERSION:
            raise ValueError(
                f"unsupported deploy schema_version {self.schema_version!r}; "
                f"expected {self.VERSION!r}"
            )
        if len(set(self.species)) != len(self.species):
            raise ValueError("deploy species must be unique and in element-table order")
        if self.receptive_field != self.num_interactions * self.r_max:
            raise ValueError(
                "deploy receptive_field must equal num_interactions * r_max"
            )

    @classmethod
    def read(cls, path: Path) -> DeployManifest:
        return TypeAdapter(cls).validate_json(path.read_bytes())

    def write(self, path: Path) -> None:
        path.write_bytes(TypeAdapter(DeployManifest).dump_json(self, indent=2) + b"\n")
