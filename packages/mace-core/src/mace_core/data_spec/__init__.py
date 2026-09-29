"""The on-disk formats a prepared dataset is written in, as schemas.

A backend reads a format and a writer writes it; the format itself is stated
here once, so the two cannot drift apart.
"""

from mace_core.data_spec.shard_format import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SHARD_FORMAT,
    read_configuration,
    read_manifest,
    write_configuration,
    write_shards,
)

__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "SHARD_FORMAT",
    "read_configuration",
    "read_manifest",
    "write_configuration",
    "write_shards",
]
