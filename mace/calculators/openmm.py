from e3nn.util import jit
import torch
import mace
from mace.calculators.neighbour_list_torch import primitive_neighbor_list_torch
from mace import data
from mace.tools import torch_geometric, utils

torch.set_default_dtype(torch.float64)


def compile_model(model_path):
    model = torch.load(model_path)
    res = {}
    res["model"] = jit.compile(model)
    res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    res["r_max"] = model.r_max
    return res


class MACE_openmm(torch.nn.Module):
    def __init__(self, model_path, atoms_obj):
        super().__init__()
        dat = compile_model(model_path)
        config = data.config_from_atoms(atoms_obj)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=dat["z_table"], cutoff=dat["r_max"]
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch_dict = next(iter(data_loader)).to_dict()
        batch_dict.pop("edge_index")
        batch_dict.pop("energy", None)
        batch_dict.pop("forces", None)
        batch_dict.pop("positions")
        # batch_dict.pop("shifts")
        batch_dict.pop("weight")
        self.inp_dict = batch_dict
        self.model = dat["model"]
        self.r_max = dat["r_max"]

    def forward(self, positions):
        sender, receiver, unit_shifts = primitive_neighbor_list_torch(
            quantities="ijS",
            pbc=(False, False, False),
            cell=self.inp_dict["cell"],
            positions=positions,
            cutoff=self.r_max,
            self_interaction=True,  # we want edges from atom to itself in different periodic images
            use_scaled_positions=False,  # positions are not scaled positions
            device="cpu",
        )
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= torch.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]
        # Build output
        edge_index = torch.stack((sender, receiver))  # [2, n_edges]

        # From the docs: With the shift vector S, the distances D between atoms can be computed from
        # D = positions[j]-positions[i]+S.dot(cell)
        # shifts = torch.dot(unit_shifts, self.inp_dict["cell"])  # [n_edges, 3]
        inp_dict_this_config = self.inp_dict.copy()
        inp_dict_this_config["positions"] = positions
        inp_dict_this_config["edge_index"] = edge_index
        # inp_dict_this_config["shifts"] = shifts
        # inp_dict_this_config[""] =
        res = self.model(inp_dict_this_config)
        return (res["energy"], res["forces"])
