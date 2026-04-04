"""
Pure-torch gated equivariant nonlinearity, drop-in replacement for e3nn nn.Gate.

Eliminates graph breaks caused by:
  - if gates.shape[-1]: (data-dependent control flow)
  - _Sortcut / Extract (fx.Graph codegen with dynamic slicing)
  - ElementwiseTensorProduct (TorchScript)

Supports both mul_ir (e3nn default) and ir_mul (cuequivariance) layouts,
so callers can drop TransposeIrrepsLayout layers around the gate.
"""

from collections.abc import Sequence
from typing import Callable, Optional

import torch
from e3nn import o3

_NORM_CACHE: dict[int, float] = {}


def _normalize2mom_cst(fn: Callable) -> float:
    """Compute the normalize2mom constant for an activation function.

    Same logic as e3nn.math.normalize2mom: draw 1M samples from N(0,1),
    compute sqrt(1 / E[f(z)^2]).  Results are cached by function identity.
    """
    key = id(fn)
    cached = _NORM_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
    second_moment = fn(z).pow(2).mean()
    result = second_moment.pow(-0.5).item()
    _NORM_CACHE[key] = result
    return result


class GatedEquivariantBlock(torch.nn.Module):
    """Graph-break-free gated equivariant nonlinearity.

    Drop-in replacement for ``e3nn.nn.Gate`` with identical numerics.
    The input is the sorted direct sum of (scalars, gates, gated) irreps --
    exactly the same convention as e3nn's Gate.

    Supports ``layout="mul_ir"`` (e3nn default, multiplicity-first) and
    ``layout="ir_mul"`` (cuequivariance, irrep-dimension-first).  When layout
    is ``ir_mul``, the gated multiplication is adjusted so that no external
    transpose layers are needed.

    Parameters match ``e3nn.nn.Gate`` for compatibility, with the addition of
    the ``layout`` keyword.
    """

    def __init__(
        self,
        irreps_scalars: o3.Irreps,
        act_scalars: Sequence[Optional[Callable]],
        irreps_gates: o3.Irreps,
        act_gates: Sequence[Optional[Callable]],
        irreps_gated: o3.Irreps,
        layout: str = "mul_ir",
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"irreps_gated has {irreps_gated.num_irreps} irreps but "
                f"irreps_gates has {irreps_gates.num_irreps}"
            )
        if layout not in ("mul_ir", "ir_mul"):
            raise ValueError(f"layout must be 'mul_ir' or 'ir_mul', got '{layout}'")

        self._layout_is_ir_mul: bool = layout == "ir_mul"

        irreps_in_unsorted = irreps_scalars + irreps_gates + irreps_gated
        irreps_in_sorted, sort_perm, _ = irreps_in_unsorted.sort()
        self._irreps_in = irreps_in_sorted.simplify()

        irreps_scalars_out = self._compute_scalar_output_irreps(
            irreps_scalars, act_scalars
        )
        self._irreps_out = irreps_scalars_out + irreps_gated

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated

        # ---------------------------------------------------------------
        # Build slice indices as plain Python ints (compile-friendly,
        # correct autograd through torch.narrow with int args).
        # ---------------------------------------------------------------
        n_scalar_groups = len(irreps_scalars)
        n_gate_groups = len(irreps_gates)

        scalar_sorted_indices: list[int] = []
        gate_sorted_indices: list[int] = []
        gated_sorted_indices: list[int] = []

        for sorted_idx in range(len(irreps_in_unsorted)):
            orig = int(sort_perm[sorted_idx])
            if orig < n_scalar_groups:
                scalar_sorted_indices.append(sorted_idx)
            elif orig < n_scalar_groups + n_gate_groups:
                gate_sorted_indices.append(sorted_idx)
            else:
                gated_sorted_indices.append(sorted_idx)

        group_offsets: list[int] = []
        offset = 0
        for mul_ir in irreps_in_sorted:
            group_offsets.append(offset)
            offset += mul_ir.dim

        if scalar_sorted_indices:
            self._s_start: int = group_offsets[scalar_sorted_indices[0]]
            self._s_len: int = sum(
                irreps_in_sorted[i].dim for i in scalar_sorted_indices
            )
        else:
            self._s_start = 0
            self._s_len = 0

        if gate_sorted_indices:
            self._g_start: int = group_offsets[gate_sorted_indices[0]]
            self._g_len: int = sum(irreps_in_sorted[i].dim for i in gate_sorted_indices)
        else:
            self._g_start = 0
            self._g_len = 0

        # Gated groups: (start, total_len, ir_dim, mul, gate_offset_in_gate_slice)
        gate_offset_by_gated_orig: list[int] = []
        cum = 0
        for mul_ir in irreps_gated:
            gate_offset_by_gated_orig.append(cum)
            cum += mul_ir.mul

        gated_info: list[tuple[int, int, int, int, int]] = []
        for si in gated_sorted_indices:
            mul_ir = irreps_in_sorted[si]
            gd_start = group_offsets[si]
            gd_len = mul_ir.dim
            ir_dim = mul_ir.ir.dim
            mul = mul_ir.mul
            gated_orig_idx = int(sort_perm[si]) - n_scalar_groups - n_gate_groups
            g_off = gate_offset_by_gated_orig[gated_orig_idx]
            gated_info.append((gd_start, gd_len, ir_dim, mul, g_off))

        self._gated_info: list[tuple[int, int, int, int, int]] = gated_info

        if len(act_scalars) == 1 and len(irreps_scalars) > 1:
            act_scalars = list(act_scalars) * len(irreps_scalars)
        if len(act_gates) == 1 and len(irreps_gates) > 1:
            act_gates = list(act_gates) * len(irreps_gates)

        self._act_scalar: Optional[Callable] = None
        self._scalar_cst: float = 1.0
        if len(act_scalars) > 0 and act_scalars[0] is not None:
            self._act_scalar = act_scalars[0]
            self._scalar_cst = _normalize2mom_cst(act_scalars[0])

        self._act_gate: Optional[Callable] = None
        self._gate_cst: float = 1.0
        if len(act_gates) > 0 and act_gates[0] is not None:
            self._act_gate = act_gates[0]
            self._gate_cst = _normalize2mom_cst(act_gates[0])

    @staticmethod
    def _compute_scalar_output_irreps(
        irreps_scalars: o3.Irreps, act_scalars: Sequence[Optional[Callable]]
    ) -> o3.Irreps:
        """Determine output parity of scalar irreps after activation."""
        if len(act_scalars) == 1 and len(irreps_scalars) > 1:
            act_scalars = list(act_scalars) * len(irreps_scalars)

        out = []
        for (mul, (l_in, p_in)), act in zip(irreps_scalars, act_scalars):
            if act is not None:
                x = torch.linspace(0, 10, 256)
                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0
                p_out = p_act if p_in == -1 else p_in
                out.append((mul, (0, p_out)))
            else:
                out.append((mul, (l_in, p_in)))
        return o3.Irreps(out)

    @property
    def irreps_in(self) -> o3.Irreps:
        return self._irreps_in

    @property
    def irreps_out(self) -> o3.Irreps:
        return self._irreps_out

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        scalars = features.narrow(-1, self._s_start, self._s_len)
        if self._act_scalar is not None:
            scalars = self._act_scalar(scalars) * self._scalar_cst

        if not self._gated_info:
            return scalars

        gates = features.narrow(-1, self._g_start, self._g_len)
        if self._act_gate is not None:
            gates = self._act_gate(gates) * self._gate_cst

        ir_mul = self._layout_is_ir_mul
        gated_parts: list[torch.Tensor] = []
        for gd_start, gd_len, ir_dim, mul, g_off in self._gated_info:
            gated_chunk = features.narrow(-1, gd_start, gd_len)
            gate_chunk = gates.narrow(-1, g_off, mul)

            batch_shape = features.shape[:-1]
            if ir_mul:
                # ir_mul: data is [c1_m1, c1_m2, ..., c2_m1, ...] → (ir_dim, mul)
                gated_3d = gated_chunk.reshape(*batch_shape, ir_dim, mul)
                gate_3d = gate_chunk.unsqueeze(-2)
                result = (gated_3d * gate_3d).reshape(*batch_shape, ir_dim * mul)
            else:
                # mul_ir: data is [m1_c1, m1_c2, ..., m2_c1, ...] → (mul, ir_dim)
                gated_3d = gated_chunk.reshape(*batch_shape, mul, ir_dim)
                gate_3d = gate_chunk.unsqueeze(-1)
                result = (gated_3d * gate_3d).reshape(*batch_shape, mul * ir_dim)
            gated_parts.append(result)

        return torch.cat([scalars] + gated_parts, dim=-1)

    def __repr__(self) -> str:
        layout = "ir_mul" if self._layout_is_ir_mul else "mul_ir"
        return (
            f"{self.__class__.__name__} "
            f"({self.irreps_in} -> {self.irreps_out} | {layout})"
        )
