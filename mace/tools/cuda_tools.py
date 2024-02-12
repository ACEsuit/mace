from copy import deepcopy
import time
from math import sqrt
from typing import Tuple, List
from functools import partial
import ase
import torch
from e3nn import o3
from mace import data, tools
from e3nn.util import jit
from mace.modules.blocks import SphericalHarmonics
from mace.modules.models import MACE
import torch.utils.benchmark as benchmark_

from mace.tools import torch_geometric, torch_tools
import numpy as np
import matplotlib.pyplot as plt

from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP
from mace_ops.ops.linear import Linear, ElementalLinear
from mace_ops.ops.symmetric_contraction import SymmetricContraction as CUDAContraction


def build_parser():
    """
    Create a parser for the command line tool.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a MACE model for CUDA inference."
    )
    parser.add_argument("--model", type=str, help="Path to the MACE model.")
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_model.model",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Default dtype of the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="",
        help="Path to the benchmark file.",
    )
    return parser


def optimize_cuda_mace(model: MACE) -> None:
    """
    Optimize the MACE model for CUDA inference.
    """
    for param in model.parameters():
        param.requires_grad = False
    dtype = get_model_dtype(model)
    n_layers = int(model.num_interactions)
    sh_irreps = o3.Irreps.spherical_harmonics(3)
    spherical_harmonics = SphericalHarmonics(
        sh_irreps=sh_irreps,
        normalize=True,
        normalization="component",
        backend="opt",
    )
    model.spherical_harmonics = spherical_harmonics
    num_elements = model.node_embedding.linear.irreps_in.num_irreps
    for i in range(n_layers):
        model.interactions[i].linear_up = linear_matmul(model.interactions[i].linear_up)
        model.interactions[i].linear = linear_to_cuda(model.interactions[i].linear)
        model.interactions[i].tp = InvariantMessagePassingTP()
        if "Residual" in model.interactions[i].__class__.__name__:
            bound_method = invariant_residual_interaction_forward.__get__(
                model.interactions[i], model.interactions[i].__class__
            )
            setattr(model.interactions[i], "forward", bound_method)
        else:
            model.interactions[i].skip_tp = element_linear_to_cuda(
                model.interactions[i].skip_tp
            )
            bound_method = invariant_interaction_forward.__get__(
                model.interactions[i], model.interactions[i].__class__
            )
            setattr(model.interactions[i], "forward", bound_method)

        symm_contract = model.products[i].symmetric_contractions
        all_weights = {}
        for j in range(len(symm_contract.contractions)):
            all_weights[str(j)] = {}
            all_weights[str(j)][3] = (
                symm_contract.contractions[j].weights_max.detach().clone().type(dtype)
            )
            all_weights[str(j)][2] = (
                symm_contract.contractions[j].weights[0].detach().clone().type(dtype)
            )
            all_weights[str(j)][1] = (
                symm_contract.contractions[j].weights[1].detach().clone().type(dtype)
            )
        irreps_in = o3.Irreps(model.products[i].symmetric_contractions.irreps_in)
        coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        irreps_out = o3.Irreps(model.products[i].symmetric_contractions.irreps_out)
        symmetric_contractions = CUDAContraction(
            coupling_irreps,
            irreps_out,
            all_weights,
            nthreadX=32,
            nthreadY=4,
            nthreadZ=1,
            dtype=dtype,
        )
        model.products[i].symmetric_contractions = SymmetricContractionWrapper(
            symmetric_contractions
        )
        model.products[i].linear = linear_matmul(model.products[i].linear)
    return model


class SymmetricContractionWrapper(torch.nn.Module):
    def __init__(self, symmetric_contractions):
        super().__init__()
        self.symmetric_contractions = symmetric_contractions

    def forward(self, x, y):
        y = y.argmax(dim=-1).int()
        out = self.symmetric_contractions(x, y).squeeze()
        return out


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    model_dtype = next(model.parameters()).dtype
    return model_dtype


class linear_matmul(torch.nn.Module):
    def __init__(self, linear_e3nn):
        super().__init__()
        num_channels_in = linear_e3nn.__dict__["irreps_in"].num_irreps
        num_channels_out = linear_e3nn.__dict__["irreps_out"].num_irreps
        self.weights = (
            linear_e3nn.weight.data.reshape(num_channels_in, num_channels_out)
            / num_channels_in**0.5
        )

    def forward(self, x):
        return torch.matmul(x, self.weights)


def linear_to_cuda(linear):
    return Linear(
        linear.__dict__["irreps_in"],
        linear.__dict__["irreps_out"],
        linear.instructions,
        linear.weight,
    )


def element_linear_to_cuda(skip_tp):
    print("elementlinear", skip_tp)
    num_elements = skip_tp.__dict__["irreps_in2"].dim
    n_channels = skip_tp.__dict__["irreps_in1"][0].dim
    lmax = skip_tp.__dict__["irreps_in1"].lmax
    ws = skip_tp.weight.data.reshape(
        [lmax + 1, n_channels, num_elements, n_channels]
    ).permute(2, 0, 1, 3)
    ws = ws.flatten(1) / sqrt(num_elements)
    linear_instructions = o3.Linear(
        skip_tp.__dict__["irreps_in1"], skip_tp.__dict__["irreps_out"]
    )
    return ElementalLinear(
        skip_tp.__dict__["irreps_in1"],
        skip_tp.__dict__["irreps_out"],
        linear_instructions.instructions,
        ws,
        num_elements,
    )


def invariant_residual_interaction_forward(
    self,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = torch.tensor(node_feats.shape[0])
    sc = self.skip_tp(node_feats, node_attrs).contiguous()
    node_feats = self.linear_up(node_feats)
    tp_weights = self.conv_tp_weights(edge_feats)
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]).contiguous(),
        sender.int(),
        receiver.int(),
        num_nodes,
    )
    message = self.linear(message) / self.avg_num_neighbors
    return (
        message.contiguous(),
        sc.contiguous(),
    )  # [n_nodes, channels, (lmax + 1)**2]


def invariant_interaction_forward(
    self,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, None]:
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = torch.tensor(node_feats.shape[0])
    node_feats = self.linear_up(node_feats)
    tp_weights = self.conv_tp_weights(edge_feats)
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]).contiguous(),
        sender.int(),
        receiver.int(),
        num_nodes,
    )
    message = self.linear(message) / self.avg_num_neighbors
    message = self.skip_tp(message, node_attrs)
    return (
        message.contiguous(),
        None,
    )  # [n_nodes, channels, (lmax + 1)**2]


def benchmark(
    model: MACE,
    benchmark_file: str,
    name: str,
    size=1,
    atomic_numbers: List[int] = None,
    r_max: float = 4,
) -> None:
    # Load data and prepare input
    try:
        atoms_list = ase.io.read(benchmark_file, format="extxyz", index=":")
    except Exception:
        print("Could not read file {}".format(benchmark_file))
        from ase import build

        # build very large diamond structure
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms_list = [atoms.repeat((size, size, size))]
        print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")
    print("num edges", batch.edge_index.shape)
    # warm up
    for _ in range(500):
        model(batch.to_dict(), training=False, compute_force=True)
    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(1000):
        # t4 = time.time()
        model(batch.to_dict(), training=False, compute_force=True)
        # print("full model", time.time() - t4)
    diff = (time.time() - t0) / 1000
    print("time", diff)
    return diff


def accuracy(
    model: MACE,
    model_opt: MACE,
    benchmark_file: str,
    size=5,
) -> None:
    try:
        atoms_list = ase.io.read(benchmark_file, format="extxyz", index=":")
    except Exception:
        print("Could not read file {}".format(benchmark_file))
        from ase import build

        # build very large diamond structure
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms_list = [atoms.repeat((size, size, size))]
        print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=model.r_max.item()
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")
    print("num edges", batch.edge_index.shape)
    # def print_grad_hook(module, grad_input, grad_output):
    #     print("Gradients at this layer: ", grad_output)

    # for modules in model.modules():
    #     print("modules", modules)
    #     modules.register_backward_hook(print_grad_hook)

    # def energy_model(positions:torch.Tensor):
    #     batch.positions = positions
    #     energy = model_opt(batch, training=False, compute_force=False)["energy"]
    #     return energy
    # print("check the gradient")
    # positions_input = batch.positions.clone().detach().requires_grad_(True)
    # torch.autograd.gradcheck(energy_model, positions_input, eps=1e-2, atol=1e-2)
    for batch in data_loader:
        batch = batch.to("cuda")
        print("num nodes", batch.num_nodes)
        batch_2 = batch.clone()
        batch_3 = batch.clone()
        # make a copy of the model
        output_opt = model_opt(batch.to_dict(), training=False, compute_force=True)
        output_org_float32 = model(batch_3, training=False, compute_force=True)
        model = model.double()
        model.atomic_energies_fn.atomic_energies = (
            model.atomic_energies_fn.atomic_energies.double()
        )
        batch_2.positions = batch_2.positions.double()
        batch_2.node_attrs = batch_2.node_attrs.double()
        batch_2.shifts = batch_2.shifts.double()
        output_org = model(batch_2, training=False, compute_force=True)
        print("energy org", output_org["energy"])
        print("energy opt", output_opt["energy"])
        error_energy = (output_org["energy"] - output_opt["energy"]).abs().mean()
        error_energy_float32 = (
            (output_org_float32["energy"] - output_org["energy"]).abs().mean()
        )
        print("error energy float32", error_energy_float32)
        # print("error energy", error_energy)
        # assert torch.allclose(output_org["energy"], output_opt["energy"], atol=1e-5)
        error_forces = (output_org["forces"] - output_opt["forces"]).abs().mean()
        print("error forces", error_forces)
        error_forces_float32 = (
            (output_org_float32["forces"] - output_org["forces"]).abs().mean()
        )
        print("error forces float32", error_forces_float32)
        # compute relative forces error
        error_relative_forces = (
            output_org["forces"] - output_opt["forces"]
        ).abs().mean() / output_org["forces"].abs().mean()
        # error mean each component relative
        error_relative_components = (
            (
                (output_org["forces"] - output_opt["forces"])
                / (output_org["forces"] + 10e-6)
            )
            .abs()
            .mean()
        )
        conservative_error = output_opt["forces"].sum()
        print("conservative error", conservative_error)
        print("error relative forces", error_relative_forces)
        print("error relative components", error_relative_components)
        # # print("forces opt", output_opt["forces"])
        # # print("forces org", output_org["forces"])
        # assert torch.allclose(output_org["forces"], output_opt["forces"], atol=1e-5)


def main(args=None):
    """
    Optimize a MACE model for CUDA inference.
    """
    parser = build_parser()
    args = parser.parse_args(args)
    torch_tools.set_default_dtype(args.dtype)
    torch_tools.init_device("cuda")
    model = torch.load(args.model).to("cuda")
    model_opt = optimize_cuda_mace(model)
    model_dir = "/".join(args.model.split("/")[:-1])
    model_opt_path = model_dir + "/" + args.output
    model_opt = jit.compile(model_opt)
    # torch.save(model_opt, model_opt_path)
    model_opt.save(model_opt_path)
    model_opt = torch.load(model_opt_path).to("cuda")
    if args.benchmark:
        times_opt = []
        times_org = []
        model = torch.load(args.model).to("cuda")
        model = jit.compile(model)
        sizes = (
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        )
        atomic_numbers = model.atomic_numbers
        r_max = model.r_max.item()
        model_opt = jit.compile(model_opt)
        for size in sizes:
            # model_opt = torch.compile(model_opt, backend="cudagraphs")
            times_opt.append(
                benchmark(
                    model_opt, args.benchmark_file, "opt", size, atomic_numbers, r_max
                )
            )
            times_org.append(
                benchmark(
                    model, args.benchmark_file, "orig", size, atomic_numbers, r_max
                )
            )
        # plot the timing as a function of the number of atoms
        num_atoms = [8 * size**3 for size in sizes]
        num_atoms = np.array(num_atoms)
        times_opt = 1e3 * np.array(times_opt)
        times_org = 1e3 * np.array(times_org)
        print("times opt", times_opt)
        print("times org", times_org)
        plt.plot(num_atoms, times_opt, label="optimized")
        plt.plot(num_atoms, times_org, label="original")
        plt.xlabel("Number of atoms")
        plt.ylabel("Time [ms]")
        plt.legend()
        # save the plot as a pdf
        plt.savefig("benchmark.pdf")
    if args.accuracy:
        model = torch.load(args.model).to("cuda")
        # model = jit.compile(model)
        # model_opt = jit.compile(model_opt)
        accuracy(model, model_opt, args.benchmark_file)

    return None


if __name__ == "__main__":
    main()
