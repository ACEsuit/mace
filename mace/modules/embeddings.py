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
                self.embedders[name] = nn.Sequential(
                    nn.Linear(spec["in_dim"], E, bias=use_bias),
                    nn.SiLU(),
                    nn.Linear(E, E, bias=use_bias),
                )
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
            emb = self.embedders[name](feat)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self.project(x)
