from typing import Any, Dict, List, Optional

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
        self.base_dim: int = base_dim
        self.out_dim: int = out_dim or base_dim

        # Store normalized specs in TorchScript-friendly typed structures
        items = list(embedding_specs.items()) if embedding_specs is not None else []
        self.names: List[str] = [name for name, _ in items]
        self.per_graph: List[bool] = []
        self.is_categorical: List[bool] = []
        self.offsets: List[int] = []
        self.emb_dims: List[int] = []

        # build one embedder per feature (keep order deterministic)
        self.embedders = nn.ModuleList()
        for name, spec in items:
            E = int(spec["emb_dim"])  # output emb dim for this feature
            use_bias = bool(spec.get("use_bias", True))
            ftype = str(spec["type"])  # 'categorical' or 'continuous'
            per = str(spec["per"])  # 'graph' or 'atom'

            self.per_graph.append(per == "graph")
            self.is_categorical.append(ftype == "categorical")
            self.offsets.append(int(spec.get("offset", 0)))
            self.emb_dims.append(E)

            if ftype == "categorical":
                num_classes = int(spec["num_classes"])  # required
                self.embedders.append(nn.Embedding(num_classes, E))
            elif ftype == "continuous":
                in_dim = int(spec["in_dim"])  # required
                self.embedders.append(
                    nn.Sequential(
                        nn.Linear(in_dim, E, bias=use_bias),
                        nn.SiLU(),
                        nn.Linear(E, E, bias=use_bias),
                    )
                )
            else:
                raise ValueError(f"Unknown type {ftype} for feature {name}")

        # build the single concat→SiLU→Linear head
        total_dim = int(sum(self.emb_dims))
        self.project = nn.Sequential(
            nn.Linear(total_dim, int(self.out_dim), bias=False),
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
        # Prefer TorchScript-friendly path during scripting or when converted
        if torch.jit.is_scripting() or (
            hasattr(self, "names") and isinstance(self.embedders, nn.ModuleList)
        ):
            embs: List[torch.Tensor] = []
            for j, embedder in enumerate(self.embedders):
                name = self.names[j]
                feat = features[name]
                if self.per_graph[j]:
                    feat = feat[batch].unsqueeze(-1)  # [N_nodes, …]
                if self.is_categorical[j]:
                    feat = (feat + self.offsets[j]).long().squeeze(-1)  # [N_nodes]
                emb = embedder(feat)
                embs.append(emb)
            x = torch.cat(embs, dim=-1)
            return self.project(x)

        # Legacy path: supports older pickled models (ModuleDict + specs dict)
        embs_legacy: List[torch.Tensor] = []
        specs = getattr(self, "specs", {})
        for name, spec in specs.items():
            feat = features[name]
            if spec.get("per", "graph") == "graph":
                feat = feat[batch].unsqueeze(-1)
            if spec.get("type", "categorical") == "categorical":
                feat = (feat + spec.get("offset", 0)).long().squeeze(-1)
            embedder = self.embedders[name]  # type: ignore[index]
            embs_legacy.append(embedder(feat))
        x = torch.cat(embs_legacy, dim=-1)
        return self.project(x)
