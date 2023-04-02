########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
########################################################################################
"""Models

These are the high-level models to be exposed. Internal workings are kept in the class
hierarchy.

"""

from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum
from .blocks import (
    AtomicEnergiesBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearDipoleOnlyReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearDipoleOnlyReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .core_models import (
    MaceCoreModel,
    EnergyModelMixin,
    ScaleShiftEnergyModelMixin,
    DipoleModelMixin,
)
from .utils import (
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)


@compile_mode("script")
class MACE(MaceCoreModel, EnergyModelMixin):
    """MACE model - energy only"""

    _LINEAR_READOUT_CLASS = LinearReadoutBlock
    _NONLINEAR_READOUT_CLASS = NonLinearReadoutBlock

    def __init__(self, **kwargs):
        """MACE energy model

        Parameters
        ----------
        # ------- core model parameters
        r_max
            cutoff of radial embedding applied to individual atoms
        num_bessel
            number of Bessel functions to be used for radial embedding
        num_polynomial_cutoff
        max_ell
            l_max, maximum channels for spherical harmonics
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

        # ------- energy model specific parameters
        atomic_energies
            energy shift (e0) of elements
            order & length to agree with atomic_numbers

        **kwargs
        """
        super().__init__(**kwargs)

    @staticmethod
    def _last_layer_irreps(hidden_irreps) -> o3.Irreps:
        """energy: Select only scalars for last layer"""
        return o3.Irreps(str(hidden_irreps[0]))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward pass - evaluation of model

        Parameters
        ----------
        data
            input graphs, as dictionary
        training
            If we are in training mode. Computational graphs are retained, for reuse.
        compute_force
            Compute forces on all nodes
        compute_virials, compute_stress
            Compute stress and virial, any of these two triggers calculation of both
        compute_displacement
            Compute symmetric displacements

        Returns
        -------
        results
            calculated results
            - `energy`: total energy on each graph
            - `node_energy`: energy of each node, i.e. local energy on each atom
            - `forces`: negative gradient of total energy on each atom (node)
            - `virials` & `stress`: same thing expressed differently
            - `displacement`: symmetric displacement of cell used for stress

        """
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Calculate energies
        total_energy, node_energy = self._calc_energy(
            data, self._calculate_layer_interactions(data)
        )

        # Calculate derivatives if needed
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE, ScaleShiftEnergyModelMixin):
    """Scaled and Shifted MACE energy model

    Same as MACE model, but allows for constant shift and rescaling of energy
    """

    def __init__(self, **kwargs):
        """Scaled & Shifted MACE energy model

        Parameters
        ----------
        # ------- core model parameters
        r_max
            cutoff of radial embedding applied to individual atoms
        num_bessel
            number of Bessel functions to be used for radial embedding
        num_polynomial_cutoff
        max_ell
            l_max, maximum channels for spherical harmonics
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

        # ------- energy model specific parameters
        atomic_energies
            energy shift (e0) of elements
            order & length to agree with atomic_numbers
        atomic_inter_scale
            scale of interaction energy
        atomic_inter_shift
            constant shift of interaction energy (per atom)

        **kwargs
        """
        super().__init__(**kwargs)


@compile_mode("script")
class AtomicDipolesMACE(MaceCoreModel, DipoleModelMixin):
    """MACE model for dipoles only"""

    _LINEAR_READOUT_CLASS = LinearDipoleOnlyReadoutBlock
    _NONLINEAR_READOUT_CLASS = NonLinearDipoleOnlyReadoutBlock

    def __init__(self, hidden_irreps: o3.Irreps, atomic_energies=None, **kwargs):
        """MACE dipole only model

        Parameters
        ----------
        # ------- core model parameters
        r_max
            cutoff of radial embedding applied to individual atoms
        num_bessel
            number of Bessel functions to be used for radial embedding
        num_polynomial_cutoff
        max_ell
            l_max, maximum channels for spherical harmonics
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

        **kwargs

        Notes
        -----
        Requires at least l=1 features to be used for representing dipoles.
        `atomic_energies` parameter added for compatibility, but it's ignored

        """
        assert (
            hidden_irreps.lmax >= 1
        ), "To predict dipoles use at least l=1 hidden_irreps"
        super().__init__(hidden_irreps=hidden_irreps, **kwargs)

    @staticmethod
    def _last_layer_irreps(hidden_irreps) -> o3.Irreps:
        """dipole: Select only l=1 vectors for last layer"""
        return o3.Irreps(str(hidden_irreps[1]))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward pass - evaluation of model

        Parameters
        ----------
        data
            input graphs, as dictionary
        training
            If we are in training mode. Computational graphs are retained, for reuse.
        compute_force, compute_virials, compute_stress, compute_displacement
            compatibility only, RAISES ERROR IF True

        Returns
        -------
        results
            calculated results
            - `dipole`: dipole of the graph, includes atomic and fixed charge dipole
                components as well
            - `atomic_dipoles`: dipoles of each atom

        """

        # dipoles and virials / stress not supported simultaneously
        error_msg = "AtomicDipolesMACE does not support energy & its derivatives"
        assert not compute_force, error_msg
        assert not compute_virials, error_msg
        assert not compute_stress, error_msg
        assert not compute_displacement, error_msg

        # Setup
        data["positions"].requires_grad = True

        # Evaluate layer outputs
        atomic_dipoles = self._calculate_layer_interactions(data)

        return {
            "dipole": self._calc_total_dipole(data, atomic_dipoles),
            "atomic_dipoles": atomic_dipoles,
        }


@compile_mode("script")
class EnergyDipolesMACE(MACE, DipoleModelMixin):
    """MACE model for Energy & Dipoles"""

    _LINEAR_READOUT_CLASS = LinearDipoleReadoutBlock
    _NONLINEAR_READOUT_CLASS = NonLinearDipoleReadoutBlock

    def __init__(self, hidden_irreps: o3.Irreps, **kwargs):
        """MACE energy & dipole model

        Parameters
        ----------
        # ------- core model parameters
        r_max
            cutoff of radial embedding applied to individual atoms
        num_bessel
            number of Bessel functions to be used for radial embedding
        num_polynomial_cutoff
        max_ell
            l_max, maximum channels for spherical harmonics
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

        # ------- energy model specific parameters
        atomic_energies
            energy shift (e0) of elements
            order & length to agree with atomic_numbers

        **kwargs

        Notes
        -----
        Requires at least l=1 features to be used for representing dipoles.
        `atomic_energies` parameter added for compatibility, but it's ignored

        """
        assert (
            hidden_irreps.lmax >= 1
        ), "To predict dipoles use at least l=1 hidden_irreps"
        super().__init__(hidden_irreps=hidden_irreps, **kwargs)

    @staticmethod
    def _last_layer_irreps(hidden_irreps) -> o3.Irreps:
        """energy & dipole: Select scalars and l=1 vectors for last layer"""
        return o3.Irreps(str(hidden_irreps[:2]))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward pass - evaluation of model

        Parameters
        ----------
        data
            input graphs, as dictionary
        training
            If we are in training mode. Computational graphs are retained, for reuse.
        compute_force
            Compute forces on all nodes
        compute_virials, compute_stress, compute_displacement
            compatibility only, RAISES ERROR IF True

        Returns
        -------
        results
            calculated results
            - `energy`: total energy on each graph
            - `node_energy`: energy of each node, i.e. local energy on each atom
            - `forces`: negative gradient of total energy on each atom (node)
            - `dipole`: dipole of the graph, includes atomic and fixed charge dipole
                components as well
            - `atomic_dipoles`: dipoles of each atom

        """
        # dipoles and virials / stress not supported simultaneously
        error_msg = "dipoles and virials / stress not supported simultaneously"
        assert not compute_virials, error_msg
        assert not compute_stress, error_msg
        assert not compute_displacement, error_msg

        # Setup
        data["positions"].requires_grad = True

        # Evaluate layer outputs & unpack
        layer_outputs = self._calculate_layer_interactions(data)
        interaction_energies = layer_outputs[:, 0]
        atomic_dipoles = layer_outputs[:, 1:]

        # Calculate energies
        total_energy, node_energy = self._calc_energy(data, interaction_energies)

        # Calculate derivatives if needed
        forces, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=None,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "dipole": self._calc_total_dipole(data, atomic_dipoles),
            "atomic_dipoles": atomic_dipoles,
        }


class BOTNet(torch.nn.Module):
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
        atomic_energies: np.ndarray,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
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
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=data["num_graphs"]
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data["edge_index"], shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=data["num_graphs"],
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": compute_forces(
                energy=total_energy, positions=data.positions, training=training
            ),
        }

        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=data["num_graphs"]
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data["edge_index"], shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=data["num_graphs"]
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output
