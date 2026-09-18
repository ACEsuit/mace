"""Equivariant building blocks of the v1 PyTorch stack.

Blocks here are plain :class:`torch.nn.Module` subclasses with no TorchScript
decoration. ``torch.compile`` is the compiled path; eager is the reference.
"""

from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.graph_features import FeatureSpec, GraphFeatureEmbedding
from mace_torch.nn.interaction import InteractionBlock
from mace_torch.nn.node_inputs import NodeInputEmbedding
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

__all__ = [
    "EquivariantProductBasisBlock",
    "FeatureSpec",
    "GraphFeatureEmbedding",
    "InteractionBlock",
    "MACEBackbone",
    "NodeInputEmbedding",
]
