from typing import Callable, Dict, Type

from .blocks import (
    AgnosticNonlinearInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
    ScaleShiftBlock,
)
from .loss import EnergyForcesLoss, WeightedEnergyForcesLoss, WeightedForcesLoss
from .models import MACE, BOTNet, ScaleShiftBOTNet, ScaleShiftMACE
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
}

scaling_classes: Dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "MACE",
    "ScaleShiftMACE",
    "BOTNet",
    "ScaleShiftBOTNet",
    "EnergyForcesLoss",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
]
