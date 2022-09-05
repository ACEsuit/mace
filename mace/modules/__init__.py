from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AgnosticNonlinearInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    FixedChargeDipoleBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
    ScaleShiftBlock,
)
from .loss import (
    DipoleSingleLoss,
    EnergyForcesLoss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
)
from .models import MACE, AtomicDipolesMACE, BOTNet, ScaleShiftBOTNet, ScaleShiftMACE
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
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
    "rms_dipoles_scaling": compute_rms_dipoles,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "FixedChargeDipoleBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "MACE",
    "ScaleShiftMACE",
    "BOTNet",
    "ScaleShiftBOTNet",
    "AtomicDipolesMACE",
    "EnergyForcesLoss",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
]
