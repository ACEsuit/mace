"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import types
from typing import Optional

import torch
from e3nn import o3

from mace.modules.symmetric_contraction import SymmetricContraction
from mace.tools.cg import O3_e3nn
from mace.tools.scatter import scatter_sum

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUET_AVAILABLE = False

try:
    import openequivariance as oeq

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = (
                O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
            )
        if not CUET_AVAILABLE:
            self.enabled = False


def get_layout(cueq_config: Optional["CuEquivarianceConfig"] = None) -> str:
    """Return the irreps layout string for the active backend."""
    if cueq_config is not None and cueq_config.enabled:
        return getattr(cueq_config, "layout_str", "mul_ir")
    return "mul_ir"


@dataclasses.dataclass
class OEQConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    optimize_all: bool = False
    optimize_channelwise: bool = False
    conv_fusion: Optional[str] = "atomic"

    def __post_init__(self):
        if not OEQ_AVAILABLE:
            self.enabled = False


class Linear:
    """Returns either a cuet.Linear or o3.Linear based on config"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            return cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                method="naive",
            )

        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


def with_scatter_sum(conv_tp: torch.nn.Module) -> torch.nn.Module:
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        mji = self.original_forward(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


def with_oeq_conv_fusion(
    conv_tp: torch.nn.Module,
    transpose_in: Optional[torch.nn.Module] = None,
    transpose_out: Optional[torch.nn.Module] = None,
) -> torch.nn.Module:
    """Wraps an oeq.TensorProductConv to match MACE's conv_tp calling convention.

    oeq.TensorProductConv.forward(X, Y, W, rows, cols) performs a fused
    tensor-product + scatter.  MACE interaction blocks call
    conv_tp(node_feats, edge_attrs, tp_weights, edge_index) where
    edge_index[0]=sender, edge_index[1]=receiver.

    When cueq is active with ir_mul layout, optional transpose modules
    convert node_feats from ir_mul → mul_ir before the oeq forward and
    the output from mul_ir → ir_mul after, since oeq always uses mul_ir.
    """
    conv_tp.original_forward = conv_tp.forward
    if not hasattr(conv_tp, "weight_numel"):
        conv_tp.weight_numel = conv_tp.input_args["problem"].weight_numel
    conv_tp.layout_transpose_in = transpose_in
    conv_tp.layout_transpose_out = transpose_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        if self.layout_transpose_in is not None:
            node_feats = self.layout_transpose_in(node_feats)
        out = self.original_forward(
            node_feats,
            edge_attrs,
            tp_weights,
            receiver,
            sender,
        )
        if self.layout_transpose_out is not None:
            out = self.layout_transpose_out(out)
        return out

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


def with_oeq_scatter_sum(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps an oeq.TensorProduct (non-fused) to add scatter like e3nn path."""
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        mji = self.original_forward(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


class TensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct/oeq.TensorProduct followed by a scatter sum"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[list] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        use_conv_fusion: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            if cueq_config.conv_fusion and use_conv_fusion:
                return with_cueq_conv_fusion(
                    cuet.SegmentedPolynomial(
                        cue.descriptors.channelwise_tensor_product(
                            cue.Irreps(cueq_config.group, irreps_in1),
                            cue.Irreps(cueq_config.group, irreps_in2),
                            cue.Irreps(cueq_config.group, irreps_out),
                        )
                        .flatten_coefficient_modes()
                        .squeeze_modes()
                        .polynomial,
                        math_dtype=torch.get_default_dtype(),
                        method="uniform_1d",
                    )
                )
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )
        if (
            OEQ_AVAILABLE
            and oeq_config is not None
            and oeq_config.enabled
            and (oeq_config.optimize_all or oeq_config.optimize_channelwise)
        ):
            dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
            tpp = oeq.TPProblem(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                irrep_dtype=dtype,
                weight_dtype=dtype,
            )

            if oeq_config.conv_fusion is None:
                return with_oeq_scatter_sum(oeq.TensorProduct(tpp))
            if oeq_config.conv_fusion == "atomic":
                t_in, t_out = None, None
                if (
                    CUET_AVAILABLE
                    and cueq_config is not None
                    and cueq_config.enabled
                    and cueq_config.layout_str == "ir_mul"
                ):
                    t_in = cuet.TransposeIrrepsLayout(
                        cue.Irreps(cueq_config.group, irreps_in1),
                        source=cue.ir_mul,
                        target=cue.mul_ir,
                        use_fallback=True,
                    )
                    t_out = cuet.TransposeIrrepsLayout(
                        cue.Irreps(cueq_config.group, irreps_out),
                        source=cue.mul_ir,
                        target=cue.ir_mul,
                        use_fallback=True,
                    )
                return with_oeq_conv_fusion(
                    oeq.TensorProductConv(tpp, deterministic=False),
                    transpose_in=t_in,
                    transpose_out=t_out,
                )

            raise ValueError(f"Unknown conv_fusion option: {oeq_config.conv_fusion}")

        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class FullyConnectedTensorProduct:
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            return cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                method="naive",
            )

        return o3.FullyConnectedTensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


class SymmetricContractionWrapper:
    """Wrapper around SymmetricContraction/cuet.SymmetricContraction"""

    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,
    ):
        use_reduced_cg = use_reduced_cg and CUET_AVAILABLE
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)
        ):
            return cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                original_mace=(not use_reduced_cg),
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )

        return SymmetricContraction(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )


class TransposeIrrepsLayoutWrapper:
    """Wrapper around cuet.TransposeIrrepsLayout"""

    def __new__(
        cls,
        irreps: o3.Irreps,
        source: str,
        target: str,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        if CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled:
            # If layouts are the same, no-op
            if source == target:
                return None
            return cuet.TransposeIrrepsLayout(
                cue.Irreps(cueq_config.group, irreps),
                source=getattr(cue, source),
                target=getattr(cue, target),
                use_fallback=True,
            )

        return None
