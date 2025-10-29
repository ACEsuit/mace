import torch

import torch.nn as nn

from e3nn import o3


class MILPooling(nn.Module):
    """
    Multi-head attention-based MIL pooling over per-atom features.
    Input:
        node_feats: Tensor of shape (N_atoms, C) or (N_atoms, C, M)
                    where C is the flattened irreps channel dim.
    Output:
        Scalar energy prediction (Tensor of shape (1,))
    """
    def __init__(
            self,
            irreps: o3.Irreps,
            num_heads: int = 4,
            temperature: float = 1.0,
            dropout: float = 0.1,
    ):
        super().__init__()
        # Total channels after flattening
        self.feat_dim = irreps.num_irreps
        # Multi-head attention scores; each head maps irreps -> scalar (0e)
        self.num_heads = num_heads
        self.attns = nn.ModuleList([
            o3.Linear(
                irreps,
                o3.Irreps("1x0e"),
                internal_weights=True,
                shared_weights=True,
            )
            for _ in range(num_heads)
        ])
        # Merge H head scores into a single score per atom
        self.combine = nn.Linear(num_heads, 1)
        # Temperature for softmax over atoms
        self.temperature = temperature
        # Final MLP from pooled global feature to energy
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 1),
        )
    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        # Flatten if needed: (N, C, M) -> (N, C)
        if node_feats.dim() > 2:
            node_feats = node_feats.reshape(node_feats.size(0), -1)
        # Per-head atomic scores, stacked to (N, H)
        scores = torch.stack(
            [attn(node_feats).squeeze(-1) for attn in self.attns],
            dim=1,
        )  # (N, num_heads)
        # Merge head scores -> (N, 1) -> (N,)
        scores = self.combine(scores).squeeze(-1)  # (N,)
        # Attention over atoms (softmax along atom dimension)
        alphas = torch.softmax(scores / self.temperature, dim=0)  # (N,)
        # Weighted pooling over atoms -> global feature (C,)
        global_feat = torch.sum(alphas.unsqueeze(1) * node_feats, dim=0)
        # Predict energy
        return self.head(global_feat)

# file: mace/models/mil_pooling.py
import torch
import torch.nn as nn
from e3nn import o3

class ConjunctivePooling(nn.Module):
    """
    Conjunctive MIL pooling for MACE.
    Idea:
      1) Project the final irreps to scalar (0e) channels via o3.Linear -> (N_atoms, C0).
      2) Produce a shared attention weight α_i ∈ (0, 1) per atom.
      3) Produce per-atom logits f_i ∈ R^H (H = number of heads/targets).
      4) Output bag logits as mean_i( α_i * f_i ) ∈ R^H.
    Args:
        irreps_in:   Per-atom irreps at the final product layer.
        out_dim:     Number of heads/targets (len(heads)).
        d_attn:      Hidden dimension for the attention MLP.
        dropout:     Dropout applied to scalar features.
    """
    def __init__(
            self,
            irreps_in: o3.Irreps,
            out_dim: int,
            d_attn: int = 8,
            dropout: float = 0.1,
    ):
        super().__init__()
        # Number of scalar 0e channels in input irreps
        self.num_scalar_0e = irreps_in.count(o3.Irrep(0, 1))
        if self.num_scalar_0e == 0:
            raise ValueError(
                f"[ConjunctivePooling] irreps_in={irreps_in} contains no 0e scalars; "
                "add a scalar projection/readout before MIL pooling."
            )
        # Project general irreps to pure 0e scalars: (N_atoms, C0)
        self.to_scalars = o3.Linear(irreps_in, o3.Irreps(f"{self.num_scalar_0e}x0e"))
        # Shared attention head over scalar channels -> α_i ∈ (0,1)
        self.attention_head = nn.Sequential(
            nn.Linear(self.num_scalar_0e, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        # Per-atom classifier producing logits for each head
        self.instance_classifier = nn.Linear(self.num_scalar_0e, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dim = out_dim
    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            node_feats: per-atom representations; should be compatible with e3nn
                        packed format accepted by o3.Linear. If your tensor is of
                        shape (N_atoms, C, M), flatten as needed before passing in.
        Returns:
            bag_logits: Tensor of shape (out_dim,), one scalar per target/head.
        """
        # Map to scalar 0e channels
        scalars = self.to_scalars(node_feats)  # (N_atoms, C0)
        scalars = self.dropout(scalars)
        # Attention weights per atom: (N_atoms, 1)
        attn = self.attention_head(scalars)
        # Per-atom logits per head: (N_atoms, H)
        instance_logits = self.instance_classifier(scalars)
        # Elementwise weighting and bag-level mean: (H,)
        weighted = instance_logits * attn  # broadcast over last dim
        bag_logits = weighted.mean(dim=0)
        return bag_logits

