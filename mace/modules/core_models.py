"""Core model classes, and mixins for various quantities of interest

Notes
-----
`MaceCoreModel` is the backbone of MACE models, stores all needed blocks and allows
for customisable readout shapes.

Mixin classes are defined for calculating quantities:
- EnergyModelMixin: inherit from if you want to calculate energies
- ScaleShiftEnergyModelMixin: energies with scaling and shifting
- DipoleModelMixin: dipole learning

"""
from typing import Type, List, Optional, Callable, Dict, Tuple

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.modules import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
    EquivariantProductBasisBlock,
    AtomicEnergiesBlock,
    ScaleShiftBlock,
)
from mace.modules.utils import get_edge_vectors_and_lengths, compute_fixed_charge_dipole
from mace.tools.scatter import scatter_sum


@compile_mode("script")
class MaceCoreModel(torch.nn.Module):
    """Core model for all MACE models

    Includes the following
    - graph parameters: r_max, elements, etc.
    - embeddings (node, edge)
    - readout blocks (subclasses can change settings of these)

    """

    _LINEAR_READOUT_CLASS = LinearReadoutBlock
    _NONLINEAR_READOUT_CLASS = NonLinearReadoutBlock

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        **kwargs,
    ):
        """Core MACE model

        Parameters
        ----------
        r_max
            cutoff of radial embedding applied to individual atoms
        num_bessel
            number of Bessel functions to be used for radial embedding
        num_polynomial_cutoff
        max_ell
            l_max
        interaction_cls
            class to be used for interactions blocks
        interaction_cls_first
            class to be used for the first layer's interaction block
        num_interactions
            number of interaction layers to use
        num_elements
            redundant parameter for the number of elements
        hidden_irreps
            hidden irreducible representations, basically the size of the layer features
            and hence direct control on the size of the model
        MLP_irreps
        avg_num_neighbors
        atomic_numbers
            atomic numbers of elements the model supports
            order & length to agree with atomic_energies
        correlation
        gate
            non-linearity for non-linear readouts
        radial_MLP
        """
        super().__init__(**kwargs)

        # Main Buffers
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readout
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(self._LINEAR_READOUT_CLASS(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = self._last_layer_irreps(hidden_irreps)
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    self._NONLINEAR_READOUT_CLASS(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(self._LINEAR_READOUT_CLASS(hidden_irreps))

    @staticmethod
    def _last_layer_irreps(hidden_irreps) -> o3.Irreps:
        """Irreps to use in the last layer - used for initialisation of subclasses

        core model: drops highest l irreps, unless it's scalar only

        """
        if len(hidden_irreps) == 1:
            return hidden_irreps
        return o3.Irreps(str(hidden_irreps[:-1]))

    def _calculate_layer_interactions(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Calculate the layer interactions - used within forward pass

        Parameters
        ----------
        data

        Returns
        -------
        layer_outputs
            shape: [n_nodes, len(ouptuts)]

        """
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        layer_outputs = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            layer_outputs.append(readout(node_feats).squeeze(-1))  # [n_nodes, ]

        return torch.sum(torch.stack(layer_outputs, dim=0), dim=0)


@compile_mode("script")
class EnergyModelMixin(torch.nn.Module):
    """Mixin class for energy models

    Supplies:
    - e0: atomic energy block

    """

    def __init__(self, atomic_energies: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        # Energy calculation specific bits
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

    def _calc_energy(
        self,
        data: Dict[str, torch.Tensor],
        layer_output_energies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_graphs = data["ptr"].numel() - 1

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])

        # Sum over energy contributions
        node_energy = node_e0 + layer_output_energies
        total_energy = scatter_sum(
            src=node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        return total_energy, node_energy

    def _calculate_e0(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """e0 calculation, element-wise energy shift

        Notes
        -----
        This is really a lookup from the e0 list we have for each node's element.

        """
        return self.atomic_energies_fn(data["node_attrs"])


@compile_mode("script")
class ScaleShiftEnergyModelMixin(EnergyModelMixin):
    """Scaled and shifted energy model"""

    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        """

        Parameters
        ----------
        atomic_inter_scale
            scale of interaction energy
        atomic_inter_shift
            constant shift of interaction energy (per atom)

        **kwargs
        """
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def _calc_energy(
        self,
        data: Dict[str, torch.Tensor],
        layer_output_energies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super()._calc_energy(data, self.scale_shift(layer_output_energies))


@compile_mode("script")
class DipoleModelMixin(torch.nn.Module):
    """Mixin class for dipole models

    Supplies:
    - function to calculate total dipole, including fixed charge baseline

    """

    @staticmethod
    def _calc_total_dipole(
        data: Dict[str, torch.Tensor],
        atomic_dipoles: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates total dipoles - adding fixed charge baseline

        Parameters
        ----------
        data
        atomic_dipoles
            corresponding output of the interaction layers

        Returns
        -------
        total_dipole
            shape: [n_graphs,3]

        """
        num_graphs = data["ptr"].numel() - 1

        # Sum over dipole contributions
        baseline_dipole = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        total_dipole += baseline_dipole

        return total_dipole  # [n_graphs,3]
