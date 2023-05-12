from copy import deepcopy
import ase
import torch
from e3nn import o3
from mace import data, tools
from mace.modules.blocks import SphericalHarmonics
from mace.modules.models import MACE
from torch.profiler import profile, record_function, ProfilerActivity

from mace.modules.symmetric_contraction import SymmetricContraction
from mace.tools import torch_geometric, torch_tools


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
    n_layers = len(model.num_interactions)
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
        symmetric_contractions = SymmetricContraction(
            irreps_in=o3.Irreps(model.products[i].symmetric_contractions.irreps_in),
            irreps_out=o3.Irreps(model.product[i].symmetric_contractions.irreps_out),
            correlation=o3.Irreps(model.products[i].symmetric_contractions.correlation),
            cuda_optimized=True,
            num_elements=num_elements,
        )
        symmetric_contractions.contractions[0].weights["3"] = deepcopy(
            model.products[i].symmetric_contractions.contractions[0].weights_max.data
        )
        symmetric_contractions.contractions[0].weights["2"] = deepcopy(
            model.products[i]
            .symmetric_contractions.contractions[0]
            .weights._parameters.values()[0]
            .data
        )
        symmetric_contractions.contractions[0].weights["1"] = deepcopy(
            model.products[i]
            .symmetric_contractions.contractions[0]
            .weights._parameters.values()[1]
            .data
        )
        model.products[i].symmetric_contractions = symmetric_contractions
    return model


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
