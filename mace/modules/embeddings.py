# joint_embedding.py

import torch
from torch import nn
from typing import Dict, Sequence, Optional

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
        embedding_specs: Sequence[tuple[str, Dict]],
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.specs = {name: spec for name, spec in embedding_specs}
        # final output dim
        self.out_dim = out_dim or base_dim

        # build one embedder per feature
        self.embedders = nn.ModuleDict()
        for name, spec in self.specs.items():
            E = spec["emb_dim"]
            if spec["type"] == "categorical":
                self.embedders[name] = nn.Embedding(spec["num_classes"], E)
            elif spec["type"] == "continuous":
                self.embedders[name] = nn.Sequential(
                    nn.Linear(spec["in_dim"], E),
                    nn.SiLU(),
                    nn.Linear(E, E),
                )
            else:
                raise ValueError(f"Unknown type {spec['type']} for feature {name}")

        # build the single concat→SiLU→Linear head
        total_dim = base_dim + sum(spec["emb_dim"] for spec in self.specs.values())
        self.project = nn.Sequential(
            nn.Linear(total_dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(
        self,
        species_emb: torch.Tensor,   # [N_nodes, base_dim]
        batch: torch.Tensor,         # [N_nodes,] graph indices
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        features[name] is either [N_graphs, …] or [N_nodes, …]
        and we upsample any per‐graph ones via feat[batch].
        Returns: [N_nodes, out_dim]
        """
        embs = [species_emb]
        for name, spec in self.specs.items():
            feat = features[name]
            if spec["per"] == "graph":
                feat = feat[batch]   # now [N_nodes, …]
            emb = self.embedders[name](feat)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self.project(x)