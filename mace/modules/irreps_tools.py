###########################################################################################
# Elementary tools for handling irreducible representations
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Tuple

import torch
import math
from e3nn import o3
from e3nn.util.jit import compile_mode

def make_tp_irreps(target_irreps, correlation):
    """
    multiply irreps from eg. 4x0e + 4x1o -> 4**correlation + 4 ** correlation
    """
    tp_irreps = o3.Irreps()
    for ir in target_irreps:
        tmp_irreps = o3.Irreps(str(ir))
        tp_irreps += (tmp_irreps * ((tmp_irreps.num_irreps) ** (correlation - 1))).simplify()
    return tp_irreps

def make_tucker_irreps(target_irreps, correlation):
    tp_irreps = o3.Irreps()
    for ir in target_irreps:
        tmp_irreps = o3.Irreps(str(ir))
        num_feats = 0      
        for nu in range(1, correlation + 1):
            num_feats += tmp_irreps.num_irreps ** nu
        tp_irreps += o3.Irreps(f"{num_feats}x{tmp_irreps[0].ir}")
    return tp_irreps

def make_tucker_irreps_flexible(target_irreps, correlation):
    tp_irreps = o3.Irreps()
    for ir in target_irreps:
        tmp_irreps = o3.Irreps(str(ir))
        num_feats = 0      
        for nu in range(1, correlation + 1):
            num_feats += (math.ceil(ir.mul ** (1 / nu))) ** nu
        tp_irreps += o3.Irreps(f"{num_feats}x{tmp_irreps[0].ir}")
    return tp_irreps

# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


def linear_out_irreps(irreps: o3.Irreps, target_irreps: o3.Irreps) -> o3.Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f"{ir_in} not in {target_irreps}")

    return o3.Irreps(irreps_mid)


@compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)

@compile_mode("script")
class inverse_reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = []
        ix = 0
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul, :d]  # [batch, mul, repr]
            field = field.reshape(tensor.shape[0], -1)  # Flatten [batch, mul * repr]
            out.append(field)
            ix += mul
        return torch.cat(out, dim=-1)

def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask
