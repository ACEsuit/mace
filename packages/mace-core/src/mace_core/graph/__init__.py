"""The flat graph dict a model is handed: its core, its layout, its checks, batching.

``mace_core.graph`` owns the specification every framework binds: the
structural core :data:`GRAPH_CORE`, the per-run :class:`GraphLayout`, the
:class:`GraphInfo` kept beside a batch, the checks split by cost, the numpy
reference :func:`collate`, and :func:`graph_sample`, from a parsed
:class:`~mace_core.data.configuration.Configuration` to a sample. Framework
packages tensorize; none redefines the core or the layout.

No framework is imported here.
"""

from mace_core.graph.build import SampleError, graph_sample
from mace_core.graph.collate import COLLATED_KEYS, collate, collate_targets
from mace_core.graph.info import GraphInfo, PaddingInfo
from mace_core.graph.layout import FloatDtype, GraphLayout, graph_layout
from mace_core.graph.schema import (
    ANY_TRAILING,
    EDGE_KEYS,
    FLOAT,
    GRAPH_CORE,
    FieldSpec,
    presence_key,
    weight_key,
)
from mace_core.graph.validation import (
    GraphValidationError,
    check_batch_invariants,
    check_layout,
)

__all__ = [
    "ANY_TRAILING",
    "COLLATED_KEYS",
    "EDGE_KEYS",
    "FLOAT",
    "GRAPH_CORE",
    "FieldSpec",
    "FloatDtype",
    "GraphInfo",
    "GraphLayout",
    "GraphValidationError",
    "PaddingInfo",
    "SampleError",
    "check_batch_invariants",
    "check_layout",
    "collate",
    "collate_targets",
    "graph_layout",
    "graph_sample",
    "presence_key",
    "weight_key",
]
