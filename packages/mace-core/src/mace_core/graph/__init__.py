"""The flat graph dict a model is handed: its core, its layout, its checks, batching.

``mace_core.graph`` owns the specification every framework binds: the
structural core :data:`GRAPH_CORE`, the per-run :class:`GraphLayout`, the
:class:`GraphInfo` kept beside a batch, the checks split by cost, the numpy
reference :func:`collate`, and the two steps from a
parsed :class:`~mace_core.data.configuration.Configuration` to a sample,
:func:`structure_graph` and :func:`graph_sample`. Framework packages tensorize;
none redefines the core or the layout.

No framework is imported here.
"""

from mace_core.graph.build import SampleError, graph_sample, structure_graph
from mace_core.graph.collate import COLLATED_KEYS, collate, collate_targets
from mace_core.graph.info import GraphInfo, PaddingInfo
from mace_core.graph.layout import (
    DEFAULT_INPUTS,
    DEFAULT_TARGETS,
    GraphLayout,
    default_layout,
    feature_field,
    input_field,
    target_field,
)
from mace_core.graph.schema import (
    ANY_TRAILING,
    EDGE_KEYS,
    FLOAT,
    GRAPH_CORE,
    NOT_KEYS,
    FieldSpec,
    Producer,
    SymDim,
    presence_key,
    weight_key,
)
from mace_core.graph.validation import (
    FLOAT_DTYPES,
    GraphValidationError,
    GraphView,
    check_batch_invariants,
    check_layout,
    dtype_name,
)

__all__ = [
    "ANY_TRAILING",
    "COLLATED_KEYS",
    "DEFAULT_INPUTS",
    "DEFAULT_TARGETS",
    "EDGE_KEYS",
    "FLOAT",
    "FLOAT_DTYPES",
    "GRAPH_CORE",
    "NOT_KEYS",
    "FieldSpec",
    "GraphInfo",
    "GraphLayout",
    "GraphValidationError",
    "GraphView",
    "PaddingInfo",
    "Producer",
    "SampleError",
    "SymDim",
    "check_batch_invariants",
    "check_layout",
    "collate",
    "collate_targets",
    "default_layout",
    "dtype_name",
    "feature_field",
    "graph_sample",
    "input_field",
    "presence_key",
    "structure_graph",
    "target_field",
    "weight_key",
]
