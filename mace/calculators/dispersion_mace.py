import numpy as np
from torch_dftd.nn.dftd3_module import DFTD3Module
from torch_dftd.dftd3_xc_params import get_dftd3_default_params
from typing import Dict
import torch
from ase.units import Bohr
from torch_dftd.functions.edge_extraction import calc_edge_index
from torch_nl import compute_neighborlist


class MACED3Wrapper(torch.nn.Module):
    def __init__(
        self,
        mace_model: torch.nn.Module,
        cutoff: float = 15.0 * Bohr,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = False,
        abc: bool = False,
        cnthr: float = 40.0 * Bohr,
        device="cpu",
        dtype=torch.float32,
        bidirectional: bool = True,
        cutoff_smoothing: str = "None",
        **kwargs
    ):
        super().__init__()
        self.mace_model = mace_model
        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping
        self.abc = abc
        self.old = old
        self.device = torch.device(device)
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        self.dftd3 = DFTD3Module(
            self.params,
            cutoff=cutoff,
            cnthr=cnthr,
            abc=abc,
            dtype=dtype,
            bidirectional=bidirectional,
            cutoff_smoothing=cutoff_smoothing,
        )
        self.dftd3.to(self.device)
        self.mace_model = self.mace_model.to(self.device)

    def voigt_6_to_full_3x3_stress(self, stress_vector):
        s1, s2, s3, s4, s5, s6 = np.transpose(stress_vector)
        return np.transpose([[s1, s6, s5], [s6, s2, s4], [s5, s4, s3]])

    def _prepare_input_dict(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pos = data["positions"].clone().detach().to(self.device)
        Z = data["atomic_numbers"].clone().detach().to(self.device)
        cell = data["cell"].clone().detach().to(self.device)
        pbc = data["pbc"].clone().detach().to(self.device)
        edge_index_d3, batch_mapping, shifts_idx = compute_neighborlist(
            torch.tensor(self.cutoff), pos, cell, pbc, data["batch"], False
        )
        # edge_index = torch.tensor(edge_index_d3, dtype=torch.long).to(self.device)
        # shift_pos = torch.tensor(shifts_d3, dtype=torch.get_default_dtype()).to(
        #     self.device
        # )
        edge_index = edge_index_d3
        shift_pos = torch.einsum(
            "jn,jnm->jm", shifts_idx, cell.view(-1, 3, 3)[batch_mapping]
        )
        return dict(
            pos=pos, Z=Z, cell=cell, pbc=pbc, edge_index=edge_index, shift_pos=shift_pos
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mace_model, name)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_dicts = self._prepare_input_dict(data)
        output_mace = self.mace_model(
            data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )
        output_dftd3 = self.dftd3.calc_energy_and_forces(
            **input_dicts, damping=self.damping
        )[0]
        output_mace["energy"] += torch.tensor(
            output_dftd3["energy"], device=self.device
        )
        output_mace["forces"] += torch.tensor(
            output_dftd3["forces"], device=self.device
        )
        if compute_stress:
            output_mace["stress"] += torch.tensor(
                self.voigt_6_to_full_3x3_stress(output_dftd3["stress"]),
                device=self.device,
            )
        return output_mace
