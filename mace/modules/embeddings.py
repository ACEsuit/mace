import torch
from e3nn import o3
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mace.modules.blocks import LinearNodeEmbeddingBlock

class JointEmbedding(torch.nn.Module, ABC):
    def __init__(
        self, 
        num_elements: int, 
        embedding_size: int, 
        num_spins: int = 101, 
        num_charges: int = 201,
        cueq_config: Optional[Dict] = None
    ):
        super().__init__()
        self.num_elements = num_elements
        self.embedding_size = embedding_size
        self.num_spins = num_spins
        self.num_charges = num_charges
        self.cueq_config = cueq_config
        
        # Common irreps definitions
        self.node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        self.node_feats_irreps = o3.Irreps([(embedding_size, (0, 1))])
        
    @abstractmethod
    def forward(
        self, 
        species_emb: torch.Tensor, 
        batch: torch.Tensor, 
        total_spin: torch.Tensor, 
        total_charge: torch.Tensor
    ) -> torch.Tensor:
        pass

class FiLMEmbedding(JointEmbedding):
    def __init__(
        self, 
        num_elements: int, 
        embedding_size: int, 
        num_spins: int = 101, 
        num_charges: int = 201,
        cueq_config: Optional[Dict] = None
    ):
        super().__init__(num_elements, embedding_size, num_spins, num_charges, cueq_config)
        
        # Charge and spin embeddings 
        self.spin_embedding = torch.nn.Embedding(num_spins, embedding_size)
        self.charge_embedding = torch.nn.Embedding(num_charges, embedding_size)
        
        # FiLM layers - generate scale and shift for each dimension
        self.film_spin = torch.nn.Linear(embedding_size, 2 * embedding_size)
        self.film_charge = torch.nn.Linear(embedding_size, 2 * embedding_size)
        
        # Final projection
        self.output_projection = LinearNodeEmbeddingBlock(
            irreps_in=self.node_feats_irreps,
            irreps_out=self.node_feats_irreps,
            cueq_config=cueq_config
        )
        
    def forward(
        self, 
        species_emb: torch.Tensor, 
        batch: torch.Tensor, 
        total_spin: torch.Tensor, 
        total_charge: torch.Tensor
    ) -> torch.Tensor:
        spin_embedding = self.spin_embedding(total_spin)      # [num_graphs, embedding_dim]
        charge_embedding = self.charge_embedding(total_charge) # [num_graphs, embedding_dim]        
        spin_params = self.film_spin(spin_embedding)  # [num_graphs, 2*embedding_dim]
        spin_scale, spin_shift = torch.chunk(spin_params, 2, dim=-1)  # each [num_graphs, embedding_dim]
        charge_params = self.film_charge(charge_embedding)
        charge_scale, charge_shift = torch.chunk(charge_params, 2, dim=-1)
        species_cond = species_emb
        species_cond = species_cond * (1 + spin_scale[batch]) + spin_shift[batch]
        species_cond = species_cond * (1 + charge_scale[batch]) + charge_shift[batch]
        return self.output_projection(torch.nn.functional.silu(species_cond))

class AttentionEmbedding(JointEmbedding):
    def __init__(
        self, 
        num_elements: int, 
        embedding_size: int, 
        num_spins: int = 101, 
        num_charges: int = 201,
        cueq_config: Optional[Dict] = None,
        num_heads: int = 4
    ):
        super().__init__(num_elements, embedding_size, num_spins, num_charges, cueq_config)
        
        self.spin_embedding = torch.nn.Embedding(num_spins, embedding_size)
        self.charge_embedding = torch.nn.Embedding(num_charges, embedding_size)
        
        # Multi-head attention for feature fusion
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_size, embedding_size)
        )
        
    def forward(
        self, 
        species_emb: torch.Tensor, # [num_nodes, embedding_dim]
        batch: torch.Tensor, 
        total_spin: torch.Tensor, 
        total_charge: torch.Tensor
    ) -> torch.Tensor:
        # Get embeddings  
        spin_emb = self.spin_embedding(total_spin)[batch]  # [num_nodes, embedding_dim] 
        charge_emb = self.charge_embedding(total_charge)[batch]  # [num_nodes, embedding_dim]
        
        # Stack features for attention [num_nodes, 3, embedding_dim]
        # Each node attends to its own species, spin and charge embeddings
        node_features = torch.stack([species_emb, spin_emb, charge_emb], dim=1) 
        
        # Self-attention across feature types
        attn_output, _ = self.attention(node_features, node_features, node_features)
        
        # Combine via mean pooling and project
        fused_features = attn_output.mean(dim=1)  # [num_nodes, embedding_dim]
        return self.output_layer(fused_features)

# Implementation 3: Conditional normalization embedding
class ConditionalNormEmbedding(JointEmbedding):
    def __init__(
        self, 
        num_elements: int, 
        embedding_size: int, 
        num_spins: int = 101, 
        num_charges: int = 201,
        cueq_config: Optional[Dict] = None
    ):
        super().__init__(num_elements, embedding_size, num_spins, num_charges, cueq_config)
        
        # Condition networks
        self.spin_condition = torch.nn.Embedding(num_spins, 2*embedding_size)  # gamma and beta
        self.charge_condition = torch.nn.Embedding(num_charges, 2*embedding_size)
        
        # Layer norm
        self.layer_norm = torch.nn.LayerNorm(embedding_size)
        
        # Final projection
        self.output_projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.SiLU(),
            torch.nn.Linear(embedding_size, embedding_size)
        )
        
    def forward(
        self, 
        species_emb: torch.Tensor, 
        batch: torch.Tensor, 
        total_spin: torch.Tensor, 
        total_charge: torch.Tensor
    ) -> torch.Tensor:
        # Species embedding
        x = species_emb
        
        # Apply layer normalization
        norm_x = self.layer_norm(x)
        
        # Get conditioning parameters
        spin_params = self.spin_condition(total_spin)  # [num_graphs, 2*embedding_dim]
        spin_gamma, spin_beta = torch.chunk(spin_params, 2, dim=-1)
        
        charge_params = self.charge_condition(total_charge)  # [num_graphs, 2*embedding_dim]
        charge_gamma, charge_beta = torch.chunk(charge_params, 2, dim=-1)
        
        # Apply modulation - broadcast graph features to corresponding nodes
        modulated = norm_x * (1 + spin_gamma[batch]) * (1 + charge_gamma[batch]) + \
                   spin_beta[batch] + charge_beta[batch]
        
        # Final non-linearity and projection
        return self.output_projection(modulated)