"""What turns the backbone's node features into a model's outputs."""

from mace_torch.models.energy import EnergyOutputHead, EnergyTerms

__all__ = ["EnergyOutputHead", "EnergyTerms"]
