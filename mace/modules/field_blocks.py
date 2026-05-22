###########################################################################################
# Long Range Blocks
# Authors: Will Baldwin, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

import math
from abc import abstractmethod
from typing import List, Optional, Tuple, Type

import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.blocks import GeneralNonLinearBiasReadoutBlock
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.radial import RadialMLP

from .wrapper_ops import (
    CuEquivarianceConfig,
    Linear,
    OEQConfig,
    TransposeIrrepsLayoutWrapper,
)


@compile_mode("script")
class MultiLayerFeatureMixer(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        num_interactions: int,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [
                Linear(node_feats_irreps, node_feats_irreps, cueq_config=cueq_config)
                for _ in range(num_interactions)
            ]
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        # all_node_feats: [num_interactions, n_nodes, irreps]
        out = torch.zeros_like(all_node_feats[0])
        for i, lin in enumerate(self.linears):
            out = out + lin(all_node_feats[i])
        return out


@compile_mode("script")
class EnvironmentDependentSpinSourceBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        max_l: int,
        zero_charges: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.zero_charges = zero_charges
        self.irreps_out = 2 * o3.Irreps.spherical_harmonics(max_l)
        self.linear = Linear(irreps_in, self.irreps_out, cueq_config=cueq_config)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        mp = self.linear(node_feats)
        if self.zero_charges:
            mp_z = torch.zeros_like(mp)
            mp_z[:, 1:] = mp[:, 1:]
            mp = mp_z
        return mp.unsqueeze(-2)  # [n_nodes, 1, (max_l+1)^2 * 2]


def _sparse_dot_instructions(
    feat_in1: o3.Irreps, feat_in2: o3.Irreps, feat_out: o3.Irreps
):
    _, instructions = tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    new = []
    for i, j, _k, mode, trainable in instructions:
        new.append((i, j, 0, mode, trainable))
    return new


class PotentialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        potential_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.potential_irreps = potential_irreps
        self.node_feats_irreps = node_feats_irreps
        self.node_attrs_irreps = node_attrs_irreps
        self.cueq_config = cueq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        *args,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


@compile_mode("script")
class AgnosticChargeBiasedLinearPotentialEmbedding(
    PotentialEmbeddingBlock
):  # pylint: disable=arguments-differ
    def _setup(
        self, charges_irreps: o3.Irreps
    ) -> None:  # pylint: disable=arguments-differ
        self.potential_linear = o3.Linear(
            irreps_in=self.potential_irreps,
            irreps_out=self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.node_feats_linear = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.charges_irreps = charges_irreps
        self.charge_embedding = o3.Linear(
            self.charges_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        layout_str = (
            self.cueq_config.layout_str
            if (self.cueq_config is not None and self.cueq_config.enabled)
            else "mul_ir"
        )
        self._potential_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.potential_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._node_feats_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._charges_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._node_feats_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source="mul_ir",
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        potential_to_mul_ir = getattr(self, "_potential_to_mul_ir", None)
        node_feats_to_mul_ir = getattr(self, "_node_feats_to_mul_ir", None)
        charges_to_mul_ir = getattr(self, "_charges_to_mul_ir", None)
        node_feats_from_mul_ir = getattr(self, "_node_feats_from_mul_ir", None)

        potential_in = potential_feats
        node_feats_in = node_feats
        charges_in = local_charges
        if potential_to_mul_ir is not None:
            potential_in = potential_to_mul_ir(potential_in)
        if node_feats_to_mul_ir is not None:
            node_feats_in = node_feats_to_mul_ir(node_feats_in)
        if charges_to_mul_ir is not None:
            charges_in = charges_to_mul_ir(charges_in)

        potential_emb = self.potential_linear(potential_in)
        node_feats_emb = self.node_feats_linear(node_feats_in)
        charges_emb = self.charge_embedding(charges_in)
        if node_feats_from_mul_ir is not None:
            potential_emb = node_feats_from_mul_ir(potential_emb)
            node_feats_emb = node_feats_from_mul_ir(node_feats_emb)
            charges_emb = node_feats_from_mul_ir(charges_emb)

        return potential_emb + node_feats_emb + charges_emb


@compile_mode("script")
class NoNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps):
        super().__init__()
        self.irreps = invar_irreps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@compile_mode("script")
class MLPNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps):
        super().__init__()
        channels = invar_irreps.count(o3.Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels], torch.nn.functional.silu
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FieldUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        field_norm_factor: float,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self.radial_MLP = radial_MLP
        self.register_buffer("field_norm_factor", torch.tensor(field_norm_factor))
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


def instructions_for_sparse_tp(feat_in1, feat_in2, feat_out):
    channels1 = feat_in1.count(o3.Irrep(0, 1))
    channels2 = feat_in2.count(o3.Irrep(0, 1))
    channels3 = feat_out.count(o3.Irrep(0, 1))
    assert channels1 == channels2 and channels1 == channels3
    _, instructions = tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    new_instructions = []
    for instr in instructions:
        i, j, _k, mode, trainable = instr
        new_instructions.append((i, j, 0, mode, trainable))
    return new_instructions


class SparseUvuTensorProduct(torch.nn.Module):
    """Torch-native specialization of sparse `uvu` TensorProducts used in Polar.

    This keeps the same parameter layout as `e3nn.o3.TensorProduct`
    (`weight` + `output_mask`) for checkpoint/weight-transfer compatibility.
    Supported path types:
    - `l x l -> 0` (invariant contraction)
    - `l x 0 -> l` (scalar modulation)
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[Tuple[int, int, int, str, bool]],
        layout: str = "mul_ir",
    ) -> None:
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError(
                f"Unsupported layout '{layout}' for SparseUvuTensorProduct; expected 'mul_ir' or 'ir_mul'"
            )
        self.layout = layout

        # Use e3nn to obtain exact path weights, weight ordering, and init.
        ref_tp = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
        )
        self.weight = torch.nn.Parameter(ref_tp.weight.detach().clone())
        output_mask = ref_tp.output_mask.detach().clone()
        if self.layout == "ir_mul":
            # e3nn reference mask is flattened in mul_ir order; transpose each
            # non-scalar irrep block to align with ir_mul flattened storage.
            remapped = output_mask.clone()
            for out_ir, out_slice in zip(self.irreps_out, self.irreps_out.slices()):
                mul, ir = out_ir
                dim = ir.dim
                block = output_mask[out_slice].view(mul, dim)
                remapped[out_slice] = block.transpose(0, 1).reshape(-1)
            output_mask = remapped
        self.register_buffer("output_mask", output_mask)
        self.weight_numel = ref_tp.weight_numel
        self.instructions = ref_tp.instructions
        self.shared_weights = True
        self.internal_weights = True

        in1_slices = self.irreps_in1.slices()
        in2_slices = self.irreps_in2.slices()
        out_slices = self.irreps_out.slices()
        self._path_meta: List[
            Tuple[int, int, int, int, int, int, int, int, float, int, int, int, int]
        ] = []
        w_offset = 0
        for ins in self.instructions:
            if ins.connection_mode != "uvu":
                raise NotImplementedError(
                    f"Unsupported connection_mode '{ins.connection_mode}' in SparseUvuTensorProduct"
                )
            i1 = int(ins.i_in1)
            i2 = int(ins.i_in2)
            io = int(ins.i_out)
            mul1, ir1 = self.irreps_in1[i1]
            mul2, ir2 = self.irreps_in2[i2]
            mul_out, ir_out = self.irreps_out[io]
            d1 = ir1.dim
            d2 = ir2.dim
            d_out = ir_out.dim
            if mul_out != mul1:
                raise NotImplementedError(
                    "SparseUvuTensorProduct requires output multiplicity to match in1 multiplicity"
                )

            # Mode 0: l x l -> 0, Mode 1: l x 0 -> l
            if d_out == 1 and d1 == d2:
                mode_code = 0
            elif d2 == 1 and d_out == d1:
                mode_code = 1
            else:
                raise NotImplementedError(
                    "SparseUvuTensorProduct only supports (l x l -> 0) and (l x 0 -> l) sparse paths"
                )

            w_size = int(ins.path_shape[0] * ins.path_shape[1])
            self._path_meta.append(
                (
                    in1_slices[i1].start,
                    in1_slices[i1].stop,
                    in2_slices[i2].start,
                    in2_slices[i2].stop,
                    out_slices[io].start,
                    out_slices[io].stop,
                    w_offset,
                    w_offset + w_size,
                    float(ins.path_weight),
                    mode_code,
                    int(mul1),
                    int(mul2),
                    int(d1),
                )
            )
            w_offset += w_size

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError(
                "SparseUvuTensorProduct expects flattened 2D tensors with shape [batch, irreps.dim]"
            )

        batch = x1.shape[0]
        out = x1.new_zeros((batch, self.irreps_out.dim))

        def to_mul_ir(block: torch.Tensor, mul: int, dim: int) -> torch.Tensor:
            if self.layout == "mul_ir":
                return block.view(batch, mul, dim)
            return block.view(batch, dim, mul).transpose(1, 2)

        def from_mul_ir(t: torch.Tensor) -> torch.Tensor:
            if self.layout == "mul_ir":
                return t.reshape(batch, -1)
            return t.transpose(1, 2).reshape(batch, -1)

        for (
            in1_start,
            in1_stop,
            in2_start,
            in2_stop,
            out_start,
            out_stop,
            w_start,
            w_stop,
            path_weight,
            mode_code,
            mul1,
            mul2,
            d1,
        ) in self._path_meta:
            in1_block = x1[:, in1_start:in1_stop]
            in2_block = x2[:, in2_start:in2_stop]
            out_block = out[:, out_start:out_stop]

            # Recover multiplicities/dimensions from current block widths.
            # For uvu paths, out multiplicity equals in1 multiplicity.
            # mode_code determines whether out is scalar (d_out=1) or vector (d_out=d1).
            if mode_code == 0:
                # (l x l -> 0): [B, mul1, d] · [B, mul2, d] -> [B, mul1]
                x1v = to_mul_ir(in1_block, mul1, d1)
                x2v = to_mul_ir(in2_block, mul2, d1)
                w = self.weight[w_start:w_stop].view(mul1, mul2)
                pair = torch.einsum("bud,bvd->buv", x1v, x2v) / math.sqrt(float(d1))
                mixed = torch.einsum("buv,uv->bu", pair, w)
                out[:, out_start:out_stop] = out_block + path_weight * mixed
            else:
                # (l x 0 -> l): x1 scaled per channel by weighted scalar mixture from x2.
                x1v = to_mul_ir(in1_block, mul1, d1)
                scalars = in2_block.view(batch, mul2)
                w = self.weight[w_start:w_stop].view(mul1, mul2)
                mixed = torch.einsum("bv,uv->bu", scalars, w)
                contrib = path_weight * (
                    x1v * mixed.unsqueeze(-1) / math.sqrt(float(d1))
                )
                out_block_mi = to_mul_ir(out_block, mul1, d1)
                out[:, out_start:out_stop] = from_mul_ir(out_block_mi + contrib)

        return out * self.output_mask


@compile_mode("script")
class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: Type[
            PotentialEmbeddingBlock
        ] = AgnosticChargeBiasedLinearPotentialEmbedding,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        num_elements: Optional[int] = None,
        **kwargs,
    ) -> None:
        _ = (nonlinearity_cls, num_elements)
        # product irreps is node_feats_irreps but only l=0
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
            charges_irreps=self.charges_irreps,
            cueq_config=self.cueq_config,
        )

        self.source_embedding = Linear(
            self.node_attrs_irreps,
            invar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
            layout=(
                self.cueq_config.layout_str
                if (self.cueq_config is not None and self.cueq_config.enabled)
                else "mul_ir"
            ),
        )
        self.nonlinearity = RadialMLP(
            [2 * invar_irreps.dim] + [64, 64, 64] + [invar_irreps.dim]
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
            layout=(
                self.cueq_config.layout_str
                if (self.cueq_config is not None and self.cueq_config.enabled)
                else "mul_ir"
            ),
        )

        MLP_irreps = (
            (32 * o3.Irreps.spherical_harmonics(self.charges_irreps.lmax))
            .sort()[0]
            .simplify()
        )
        self.readout = GeneralNonLinearBiasReadoutBlock(
            irreps_in=self.node_feats_irreps,
            MLP_irreps=MLP_irreps,
            gate=torch.nn.functional.silu,
            irreps_out=(self.charges_irreps + o3.Irreps("2x0e")),
            cueq_config=None,
        )
        layout_str = (
            self.cueq_config.layout_str
            if (self.cueq_config is not None and self.cueq_config.enabled)
            else "mul_ir"
        )
        self._readout_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._readout_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=(self.charges_irreps + o3.Irreps("2x0e")),
            source="mul_ir",
            target=layout_str,
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        # create pot feats
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs,
            local_charges,
        )
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        source_embedding = self.source_embedding(node_attrs)
        invariant_descriptors_embedded = torch.cat(
            [invariant_descriptors, source_embedding], dim=-1
        )
        nonlin_feats = self.nonlinearity(invariant_descriptors_embedded)
        new_feats = self.tp_out(node_feats, nonlin_feats)
        readout_to_mul_ir = getattr(self, "_readout_to_mul_ir", None)
        readout_from_mul_ir = getattr(self, "_readout_from_mul_ir", None)
        if readout_to_mul_ir is not None:
            new_feats = readout_to_mul_ir(new_feats)
        multipoles = self.readout(new_feats)
        if readout_from_mul_ir is not None:
            multipoles = readout_from_mul_ir(multipoles)
        return multipoles


class PostScfReadout(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@compile_mode("script")
class OneBodyMLPFieldReadout(PostScfReadout):
    def _setup(self, **kwargs) -> None:
        _ = kwargs
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.linear_up_q = o3.Linear(
            self.charges_irreps, self.node_feats_irreps, biases=True
        )
        self.linear_up_v = o3.Linear(
            self.potential_irreps, self.node_feats_irreps, biases=True
        )
        layout_str = (
            self.cueq_config.layout_str
            if (self.cueq_config is not None and self.cueq_config.enabled)
            else "mul_ir"
        )
        self._q_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.charges_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._v_to_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.potential_irreps,
            source=layout_str,
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self._up_from_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.node_feats_irreps,
            source="mul_ir",
            target=layout_str,
            cueq_config=self.cueq_config,
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_q = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
            layout=(
                self.cueq_config.layout_str
                if (self.cueq_config is not None and self.cueq_config.enabled)
                else "mul_ir"
            ),
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_v = SparseUvuTensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
            layout=(
                self.cueq_config.layout_str
                if (self.cueq_config is not None and self.cueq_config.enabled)
                else "mul_ir"
            ),
        )

        self.mlp = RadialMLP(
            [2 * invar_irreps.dim] + [128, 128, 128] + [1],
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        q_in = charges_induced + charges_0
        q_to_mul_ir = getattr(self, "_q_to_mul_ir", None)
        v_to_mul_ir = getattr(self, "_v_to_mul_ir", None)
        up_from_mul_ir = getattr(self, "_up_from_mul_ir", None)
        if q_to_mul_ir is not None:
            q_in = q_to_mul_ir(q_in)
        q_up = self.linear_up_q(q_in)
        if up_from_mul_ir is not None:
            q_up = up_from_mul_ir(q_up)

        v_in = field_feats
        if v_to_mul_ir is not None:
            v_in = v_to_mul_ir(v_in)
        v_up = self.linear_up_v(v_in)
        if up_from_mul_ir is not None:
            v_up = up_from_mul_ir(v_up)
        invar_feats = torch.cat(
            [
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ],
            dim=-1,
        )
        return self.mlp(invar_feats).squeeze(-1)


# Registries used by config files that pass strings
field_update_blocks = {
    "AgnosticEmbeddedOneBodyVariableUpdate": AgnosticEmbeddedOneBodyVariableUpdate,
}

field_readout_blocks = {
    "OneBodyMLPFieldReadout": OneBodyMLPFieldReadout,
}
