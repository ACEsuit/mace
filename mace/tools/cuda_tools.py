from copy import deepcopy
import torch
from e3nn import o3
from mace.modules.models import MACE

from mace.modules.symmetric_contraction import SymmetricContraction


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
    return parser


def optimize_cuda_mace(model: MACE) -> None:
    """
    Optimize the MACE model for CUDA inference.
    """
    n_layers = len(model.num_interactions)
    for i in range(n_layers):
        symmetric_contractions = SymmetricContraction(
            irreps_in=o3.Irreps(model.products[i].symmetric_contractions.irreps_in),
            irreps_out=o3.Irreps(model.product[i].symmetric_contractions.irreps_out),
            correlation=o3.Irreps(model.products[i].symmetric_contractions.correlation),
            cuda_optimized=True,
            num_elements=model.num_elements,
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


def main(args=None):
    """
    Optimize a MACE model for CUDA inference.
    """
    parser = parser()
    args = parser.parse_args(args)
    model = torch.load(args.model)
    model = optimize_cuda_mace(model)
    torch.save(model, args.output)
    return None
