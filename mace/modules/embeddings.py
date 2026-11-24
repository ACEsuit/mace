from typing import Any, Dict, Optional

import torch
from torch import nn


class GenericJointEmbedding(nn.Module):
    """
    Simple concat‐fusion of any set of node‐ or graph‐level features
    with a base embedding.  All features are embedded (via Embedding or small MLP),
    then concatenated onto `species_emb`, passed through SiLU+Linear.
    """

    def __init__(
        self,
        *,
        base_dim: int,
        embedding_specs: Optional[Dict[str, Any]],
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.specs = dict(embedding_specs.items())
        self.out_dim = out_dim or base_dim

        # build one embedder per feature
        self.embedders = nn.ModuleDict()
        for name, spec in self.specs.items():
            E = spec["emb_dim"]
            use_bias = spec.get("use_bias", True)
            if spec["type"] == "categorical":
                self.embedders[name] = nn.Embedding(spec["num_classes"], E)
            elif spec["type"] == "continuous":
                num_hidden = spec.get("num_hidden_layers", 1)
                layers = []
                d_in = spec["in_dim"]
                for _ in range(num_hidden):
                    layers += [nn.Linear(d_in, E, bias=use_bias), nn.SiLU()]
                    d_in = E
                layers.append(nn.Linear(d_in, E, bias=use_bias))
                self.embedders[name] = nn.Sequential(*layers)
            else:
                raise ValueError(f"Unknown type {spec['type']} for feature {name}")

        # build the single concat→SiLU→Linear head
        total_dim = sum(spec["emb_dim"] for spec in self.specs.values())
        self.project = nn.Sequential(
            nn.Linear(total_dim, self.out_dim, bias=False),
            nn.SiLU(),
        )

    def forward(
        self,
        batch: torch.Tensor,  # [N_nodes,] graph indices
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        features[name] is either [N_graphs, …] or [N_nodes, …]
        and we upsample any per‐graph ones via feat[batch].
        Returns: [N_nodes, out_dim]
        """
        embs = []
        for name, spec in self.specs.items():
            feat = features[name]
            if spec["per"] == "graph":
                feat = feat[batch].unsqueeze(-1)  # now [N_nodes, …]
            if spec["type"] == "categorical":
                feat = (feat + spec.get("offset", 0)).long().squeeze(-1)  # [N_nodes, 1]
            elif spec["type"] == "continuous" and feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            emb = self.embedders[name](feat)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self.project(x)
