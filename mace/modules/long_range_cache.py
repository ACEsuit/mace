"""Shared k-space geometry for the long-range electrostatics of ``PolarMACE``.

``PolarMACE`` evaluates two long-range operators per forward: the electrostatic
features that drive the fixed-point recursion, and the electrostatic energy that
is added to the total energy. Both are built on the same reciprocal-space
geometry - the k-vectors of the cell, the node phases (cosines/sines), the GTO
density basis and the Coulomb kernel - because both use the same
``GTOBasis(density_max_l, [density_smearing_width], kspace_cutoff)``.

``GTOElectrostaticFeatures.precompute_geometry`` already materialises that
geometry once per forward. ``GTOElectrostaticEnergy`` does not consume it and
recomputes the same tensors on every call. This module adds the missing half:
``CachedGTOElectrostaticEnergy`` can take the cache produced by the features
module and evaluate the energy from it, so a forward builds the geometry once
and both operators read it.

The cached path is deliberately a straight replay of
``GTOElectrostaticEnergy._compute_kspace_energy_common`` (assemble rho(k), apply
the Coulomb kernel, contract, subtract the GTO self-interaction) plus the same
per-boundary-condition corrections. Nothing is approximated or frozen, so for
the same inputs the result is the uncached one bit for bit, gradients included;
that equality is what the regression tests assert.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from graph_longrange.energy import GTOElectrostaticEnergy, energy_product_batch
from graph_longrange.features import (
    apply_coulomb_kernel_batch,
    assemble_fourier_series_batch,
)
from graph_longrange.slabs import slab_dipole_correction_energy

__all__ = ["CachedGTOElectrostaticEnergy"]

# Boundary-condition policies whose energy is defined on the k-space geometry.
_KSPACE_HANDLINGS = ("pbc", "slab", "molecule_in_box", "mixed_periodic", "auto")


class CachedGTOElectrostaticEnergy(GTOElectrostaticEnergy):
    """``GTOElectrostaticEnergy`` that can reuse a precomputed geometry cache.

    The cache is the dict returned by
    ``GTOElectrostaticFeatures.precompute_geometry`` for the same cell,
    positions and ``pbc``. It carries the node phases, the density basis, the
    per-k volume and the Coulomb kernel, which are exactly the tensors
    ``_compute_kspace_energy_common`` rebuilds from scratch.

    ``forward`` is unchanged and remains the reference path; use
    :meth:`forward_from_geometry` when the caller already owns the geometry.
    """

    def forward_from_geometry(
        self,
        geometry: Dict[str, Any],
        source_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the electrostatic energy from a precomputed geometry cache.

        Args:
            geometry: dict from ``GTOElectrostaticFeatures.precompute_geometry``.
            source_feats: ``[n_nodes, n_multipoles]`` density coefficients.

        Returns:
            ``[n_graphs]`` electrostatic energy.

        Raises:
            ValueError: if the cache mode does not match this module's
                ``pbc_handling``. A silent fallback would compute a different
                interaction than the one the cache was built for.
        """
        if not isinstance(geometry, dict):
            raise TypeError("geometry must be the cache dict of precompute_geometry")
        mode = geometry.get("mode")
        if mode == "realspace":
            if self.pbc_handling not in ("realspace", "auto"):
                raise ValueError(
                    f"got a realspace geometry for pbc_handling={self.pbc_handling!r}"
                )
            return self._realspace_energy(
                source_feats=source_feats,
                node_positions=geometry["node_positions"],
                batch=geometry["batch"],
            )
        if mode != "pbc":
            raise ValueError(f"unsupported geometry cache mode: {mode!r}")
        if self.pbc_handling not in _KSPACE_HANDLINGS:
            raise ValueError(
                f"got a k-space geometry for pbc_handling={self.pbc_handling!r}"
            )

        energy = self._kspace_energy_from_geometry(
            geometry=geometry, source_feats=source_feats
        )
        handling = self.pbc_handling
        if handling == "pbc":
            return energy
        if handling == "slab":
            return energy + slab_dipole_correction_energy(
                source_feats,
                geometry["node_positions"],
                geometry["volumes"],
                geometry["batch"],
            )
        if handling == "molecule_in_box":
            return energy + self.monopole_dipole_correction(
                source_feats,
                geometry["node_positions"],
                geometry["volumes"],
                geometry["batch"],
            )
        # "mixed_periodic" and "auto" share the per-graph selection: slab
        # correction for (T, T, F), monopole-dipole correction for the fully
        # non-periodic graphs, nothing otherwise.
        return energy + self._mixed_periodic_correction(
            geometry=geometry, source_feats=source_feats
        )

    def _kspace_energy_from_geometry(
        self, geometry: Dict[str, Any], source_feats: torch.Tensor
    ) -> torch.Tensor:
        """Replay of ``_compute_kspace_energy_common`` against cached tensors."""
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=geometry["cosines"],
            sines=geometry["sines"],
            density_basis_fs=geometry["density_basis_fs"],
            volume_per_k=geometry["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            density=density,
            k_factor_coulomb=geometry["k_factor_coulomb"],
        )
        energy = energy_product_batch(
            density=density,
            potential=potential,
            volume=geometry["volumes"],
            k_vector_batch=geometry["k_vector_batch"],
        )
        return self._subtract_self_interaction_if_needed(
            energy=energy,
            source_feats=source_feats,
            batch=geometry["batch"],
            num_graphs=int(geometry["volumes"].shape[0]),
        )

    def _mixed_periodic_correction(
        self, geometry: Dict[str, Any], source_feats: torch.Tensor
    ) -> torch.Tensor:
        """Replay of ``_forward_mixed_periodic``'s correction selection."""
        molecule_correction = self.monopole_dipole_correction(
            source_feats,
            geometry["node_positions"],
            geometry["volumes"],
            geometry["batch"],
        )
        slab_correction = slab_dipole_correction_energy(
            source_feats,
            geometry["node_positions"],
            geometry["volumes"],
            geometry["batch"],
        )
        pbc = geometry["pbc"]
        slab = torch.tensor([0, 0, 1], dtype=torch.bool, device=pbc.device)
        is_molecule = torch.all(torch.logical_not(pbc), dim=1)
        is_slab = torch.all(torch.logical_xor(slab, pbc), dim=1)
        correction_energy = torch.zeros_like(molecule_correction)
        correction_energy = torch.where(
            is_molecule, molecule_correction, correction_energy
        )
        correction_energy = torch.where(
            is_slab, slab_correction, correction_energy
        )
        return correction_energy