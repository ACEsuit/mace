"""The flat graph dict a model is handed: its schema, its checks, and batching.

``mace_core.graph`` owns the specification every framework binds:
:data:`GRAPH_SCHEMA` and :func:`validate_graph`, the numpy reference
:func:`collate`, and :func:`graph_sample`, the step from a parsed
:class:`~mace_core.data.configuration.Configuration` to one structure's sample.
Framework packages tensorize; none redefines the schema.

No framework is imported here.
"""

from mace_core.graph.build import graph_sample
from mace_core.graph.collate import COLLATED_KEYS, collate, derived_cell_quantities
from mace_core.graph.schema import (
    FLOAT,
    GRAPH_SCHEMA,
    HOST_KEYS,
    INPUT_KEYS,
    LABEL_KEYS,
    NOT_KEYS,
    FieldSpec,
    PadValue,
    Producer,
    SymDim,
    presence_key,
)
from mace_core.graph.validation import (
    GraphInfo,
    GraphValidationError,
    GraphView,
    validate_graph,
)

__all__ = [
    "COLLATED_KEYS",
    "FLOAT",
    "GRAPH_SCHEMA",
    "HOST_KEYS",
    "INPUT_KEYS",
    "LABEL_KEYS",
    "NOT_KEYS",
    "FieldSpec",
    "GraphInfo",
    "GraphValidationError",
    "GraphView",
    "PadValue",
    "Producer",
    "SymDim",
    "collate",
    "derived_cell_quantities",
    "graph_sample",
    "presence_key",
    "validate_graph",
]
