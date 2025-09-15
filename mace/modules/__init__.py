from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearBiasReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticAttResidualInteractionBlock,
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    RealAgnosticResidualNonLinearInteractionBlock,
    ScaleShiftBlock,
)
from .loss import (
    DipolePolarLoss,
    DipoleSingleLoss,
    UniversalLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesL1L2Loss,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesStressLoss,
    WeightedEnergyForcesVirialsLoss,
    WeightedForcesLoss,
    WeightedHuberEnergyForcesStressLoss,
)
from .models import (
    MACE,
    AtomicDielectricMACE,
    AtomicDipolesMACE,
    EnergyDipolesMACE,
    ScaleShiftMACE,
)
from .radial import BesselBasis, GaussianBasis, PolynomialCutoff, ZBLBasis
from .symmetric_contraction import SymmetricContraction
from .utils import (
    compute_avg_num_neighbors,
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
    compute_statistics,
)

interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticAttResidualInteractionBlock": RealAgnosticAttResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
    "RealAgnosticResidualNonLinearInteractionBlock": RealAgnosticResidualNonLinearInteractionBlock,
}

readout_classes: Dict[str, Type[LinearReadoutBlock]] = {
    "LinearReadoutBlock": LinearReadoutBlock,
    "LinearDipoleReadoutBlock": LinearDipoleReadoutBlock,
    "NonLinearDipoleReadoutBlock": NonLinearDipoleReadoutBlock,
    "NonLinearReadoutBlock": NonLinearReadoutBlock,
    "NonLinearBiasReadoutBlock": NonLinearBiasReadoutBlock,
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
    "ZBLBasis",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "LinearDipolePolarReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "NonLinearDipolePolarReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "GaussianBasis",
    "MACE",
    "ScaleShiftMACE",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "WeightedHuberEnergyForcesStressLoss",
    "UniversalLoss",
    "WeightedEnergyForcesL1L2Loss",
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
    "compute_statistics",
    "compute_fixed_charge_dipole",
    "compute_fixed_charge_dipole_polar",
    "compute_dielectric_gradients",
]
