from typing import Callable, Dict, Type

from .blocks import (AtomicEnergiesBlock, RadialEmbeddingBlock, LinearNodeEmbeddingBlock, NonLinearBlock, EdgeEmbeddingBlock, 
                     AtomicBaseBlock, VectorizeBlock, ProdBasisBlock, )
from .loss import EnergyForcesLoss, ACELoss, WeightedEnergyForcesLoss
from .models import InvariantMultiACE
from .radial import BesselBasis, PolynomialCutoff
from .utils import compute_mean_std_atomic_inter_energy, compute_mean_rms_energy_forces


scaling_classes: Dict[str, Type[Callable]]  = {
    'std_scaling': compute_mean_std_atomic_inter_energy,
    'rms_forces_scaling': compute_mean_rms_energy_forces,
}

__all__ = [
    'AtomicEnergiesBlock', 'RadialEmbeddingBlock', 'LinearNodeEmbeddingBlock', 'NonLinearBlock', 'PolynomialCutoff',
    'EdgeEmbeddingBlock', 'AtomicBaseBlock', 'VectorizeBlock', 'ProdBasisBlock', 'BesselBasis', 'EnergyForcesLoss', 
    'ACELoss', 'WeightedEnergyForcesLoss', 'interaction_classes', 'InteractionBlock', 'InvariantMultiACE',
    'compute_mean_std_atomic_inter_energy',
]