"""The data layer's boundary objects and the pure functions over them.

A format backend yields :class:`Configuration` objects and graph construction
consumes them; nothing in between passes a dictionary of tensors around. The
key mapping that turns a file's own names into the stack's convention names is
resolved here, at the parse, and goes no further.

:class:`AtomicNumberTable` and :class:`DefaultKeys` are re-exported from
:mod:`mace_core.elements`, where they live because a model needs the element
table whether or not it ever reads a file.
"""

from mace_core.data.configuration import (
    DEFAULT_CONFIG_TYPE,
    DEFAULT_HEAD,
    Configuration,
)
from mace_core.data.keys import (
    ARRAYS_CONVENTION_NAMES,
    INFO_CONVENTION_NAMES,
    EmbeddingFeatureSpec,
    KeySpecification,
)
from mace_core.data.splitting import group_by_config_type, random_train_valid_split
from mace_core.data.xyz import (
    ISOLATED_ATOM_CONFIG_TYPE,
    ParsedConfigurations,
    configuration_from_atoms,
    read_configurations,
)
from mace_core.elements import (
    AtomicNumberTable,
    DefaultKeys,
    atomic_number_table_from_zs,
)

__all__ = [
    "ARRAYS_CONVENTION_NAMES",
    "DEFAULT_CONFIG_TYPE",
    "DEFAULT_HEAD",
    "INFO_CONVENTION_NAMES",
    "ISOLATED_ATOM_CONFIG_TYPE",
    "AtomicNumberTable",
    "Configuration",
    "DefaultKeys",
    "EmbeddingFeatureSpec",
    "KeySpecification",
    "ParsedConfigurations",
    "atomic_number_table_from_zs",
    "configuration_from_atoms",
    "group_by_config_type",
    "random_train_valid_split",
    "read_configurations",
]
