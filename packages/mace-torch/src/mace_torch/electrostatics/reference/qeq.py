"""Analytic molecular potential used by the alternative QEq equation.

Based on NonPBC in ACEsuit/mace-scf (MIT), commit
784a510339aacc4412b61b6926ef1a92bca387c0. This equation uses SI-derived
constants and an unregularized Gaussian interaction, unlike the general
real-space reference. It includes the self term regardless of energy settings.
"""

import math

import torch
from scipy.constants import e, epsilon_0
from torch import nn


class NonperiodicQEqPotential(nn.Module):
    """The projection protocol for the explicit ``qeq_nonperiodic`` method."""

    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.coulomb_constant = e * 1e10 / (4 * math.pi * epsilon_0)

    def prepare(self, geometry):
        positions = geometry.positions
        same_graph = geometry.batch[:, None] == geometry.batch[None, :]
        delta = positions[:, None, :] - positions[None, :, :]
        squared = delta.square().sum(-1)
        # Evaluate the Gaussian's analytic limit at coincident points, also
        # keeping double backward finite on the diagonal and between graphs.
        small = squared < 1e-12 * self.width**2
        radius = torch.sqrt(torch.where(small, torch.ones_like(squared), squared))
        regular = torch.erf(radius / (2 * self.width)) / radius
        limit = (1 - squared / (12 * self.width**2)) / (math.sqrt(math.pi) * self.width)
        return self.coulomb_constant * torch.where(small, limit, regular) * same_graph

    def forward(self, cache, source_feats):
        return cache @ source_feats
