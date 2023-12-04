from copy import deepcopy
from typing import Tuple
import ase
import torch
from e3nn import o3
from mace import data, tools
from mace.modules.blocks import SphericalHarmonics
from mace.modules.models import MACE
from torch.profiler import profile, record_function, ProfilerActivity

from mace.modules.symmetric_contraction import SymmetricContraction
from mace.tools import torch_geometric, torch_tools

from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP
from mace_ops.ops.linear import Linear
from mace_ops.ops.symmetric_contraction import SymmetricContraction as CUDAContraction


def parser():
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
        default="optimized_model.pt",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        help="Default dtype of the model.",
    )
    parser.add_argument(
        "--benchmark",
        type="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="benchmark.xyz",
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
        if "residual" in model.interactions[i].__name__:
            model.interactions[i].forward = invariant_residual_interaction_forward
        else:
            model.interactions[i].forward = invariant_interaction_forward
        model.interactions[i].linear = linear_matmul(model.interactions[i].linear)
        symm_contract = model.products[i].symmetric_contractions
        all_weights = {}
        for i in range(len(symm_contract.contractions)):
            all_weights[str(i)] = {}
            all_weights[str(i)][3] = (
                symm_contract.contractions[i].weights_max.detach().clone().type(dtype)
            )
            all_weights[str(i)][2] = (
                symm_contract.contractions[i].weights[0].detach().clone().type(dtype)
            )
            all_weights[str(i)][1] = (
                symm_contract.contractions[i].weights[1].detach().clone().type(dtype)
            )
        irreps_in = o3.Irreps(model.products[i].symmetric_contractions.irreps_in)
        coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        irreps_out = (o3.Irreps(model.products[i].symmetric_contractions.irreps_out),)
        symmetric_contractions = CUDAContraction(
            coupling_irreps,
            irreps_out,
            all_weights,
            nthreadX=32,
            nthreadY=4,
            nthreadZ=1,
            dtype=dtype,
        )
        model.products[i].symmetric_contractions = symmetric_contractions
        model.products[i].linear = linear_matmul(model.products[i].linear)
        model.products[i].symmetric_contractions = symmetric_contractions
    return model


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


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
    num_nodes = node_feats.shape[0]
    sc = self.skip_tp(node_feats, node_attrs)
    node_feats = self.linear_up(node_feats)
    tp_weights = self.conv_tp_weights(edge_feats)
    first_occurences = self.tp.calculate_first_occurences(
        receiver, num_nodes, torch.Tensor().int()
    )
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]),
        sender,
        receiver,
        first_occurences,
    )
    message = self.linear(message) / self.avg_num_neighbors
    return (
        message,
        sc,
    )  # [n_nodes, channels, (lmax + 1)**2]


def invariant_interaction_forward(
    self,
    node_attrs: torch.Tensor,
    node_feats: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_feats: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    num_nodes = node_feats.shape[0]
    node_feats = self.linear_up(node_feats)
    tp_weights = self.conv_tp_weights(edge_feats)
    first_occurences = self.tp.calculate_first_occurences(
        receiver, num_nodes, torch.Tensor().int()
    )
    message = self.tp.forward(
        node_feats,
        edge_attrs,
        tp_weights.view(tp_weights.shape[0], -1, node_feats.shape[-1]),
        sender,
        receiver,
        first_occurences,
    )
    message = self.linear(message) / self.avg_num_neighbors
    message = self.skip_tp(message, node_attrs)
    return (
        self.reshape(message),
        None,
    )  # [n_nodes, channels, (lmax + 1)**2]


def benchmark(model: MACE, benchmark_file: str, name: str) -> None:
    # Load data and prepare input
    atoms_list = ase.io.read(benchmark_file, format="extxyz", index=":")
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
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            skip_first=20, wait=5, warmup=1, active=1, repeat=2
        ),
        with_stack=True,
    ) as prof:
        with record_function("model_inference"):
            for _ in range(20 + 5 + 1 + 1):
                model(batch, training=False)
                prof.step()
    print("CUDA inference time for {}:".format(name))
    print(
        prof.key_averages(group_by_stack_n=1).table(
            sort_by="cuda_time_total", row_limit=5
        )
    )

    return None


def main(args=None):
    """
    Optimize a MACE model for CUDA inference.
    """
    parser = parser()
    args = parser.parse_args(args)
    torch_tools.set_default_dtype(args.default_dtype)
    torch_tools.init_device("cuda")
    model = torch.load(args.model)
    model_opt = optimize_cuda_mace(model)
    torch.save(model_opt, args.output)
    if args.benchmark:
        benchmark(model_opt, args.benchmark_file, "opt")
        benchmark(model, args.benchmark_file, "orig")
    return None
