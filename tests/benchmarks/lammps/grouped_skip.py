"""Benchmark grouped scalar residual maps from MACE-MP-0 small on a CUDA GPU."""

import argparse
import json

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.precision import PrecisionConfig
from mace_core.kernels.canonical import fully_connected_tp_path_normalization
from mace_torch.deploy.loader import load_deployed


def species_matrices(skip):
    """Return the canonical [species, output, input] matrices for this scalar model."""
    d = skip.descriptor
    irreps = Irreps.parse(d.irreps_out)
    if (
        d.irreps_in1 != d.irreps_out
        or irreps.dimension != 128
        or len(irreps.terms) != 1
    ):
        raise ValueError("expected the 128 scalar channels of MACE-MP-0 small")
    irrep = irreps.terms[0][1]
    if (
        irrep.degree != 0
        or irrep.parity != 1
        or Irreps.parse(d.irreps_in2).dimension != 89
    ):
        raise ValueError("expected scalar features and 89 element attributes")
    factor = fully_connected_tp_path_normalization(d.irreps_in1, d.irreps_in2, irrep)
    return skip.to_canonical()["weight"].reshape(89, 128, 128) * factor


def group_nodes(element):
    species, counts = element.unique(sorted=True, return_counts=True)
    order = element.argsort(stable=True)
    return order, order.argsort(), species.unbind(), counts.tolist()


def apply_maps(features, matrices, groups):
    order, inverse, species, counts = groups
    blocks = features[order].split(counts)
    output = [
        torch.nn.functional.linear(block, matrices[z])
        for z, block in zip(species, blocks, strict=True)
    ]
    return torch.cat(output)[inverse]


def benchmark(skips, maps, x, grad, element):
    attributes = torch.nn.functional.one_hot(element, 89).to(x.dtype)

    def pair(mode):
        if mode == "vendor":
            first = skips[0](x.detach(), attributes)
            second = skips[1](x, attributes)
        else:
            groups = group_nodes(element)
            first = apply_maps(x.detach(), maps[0], groups)
            second = apply_maps(x, maps[1], groups)
        return first, second, torch.autograd.grad(second, x, grad_outputs=grad)[0]

    from tests.golden.harness import FP32

    for candidate, original in zip(pair("grouped"), pair("vendor"), strict=True):
        torch.testing.assert_close(candidate, original, atol=FP32.atol, rtol=FP32.rtol)
    samples = {"vendor": [], "grouped": []}
    for mode in samples:
        for _ in range(3):
            pair(mode)
        torch.cuda.synchronize()
    for mode in ("vendor", "grouped", "grouped", "vendor"):
        for _ in range(3):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            pair(mode)
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    model = load_deployed(
        args.checkpoint,
        device="cuda",
        precision=PrecisionConfig(model="float32"),
        backend="cueq",
    )
    skips = [
        layer.skip.eval().requires_grad_(False)
        for layer in model.model.backbone.interactions
    ]
    if len(skips) != 2:
        parser.error("use the two-layer MACE-MP-0 small checkpoint")
    maps = [species_matrices(skip) for skip in skips]
    torch.manual_seed(113)
    rows = 8960
    x = torch.randn(rows, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn_like(x)
    silicon = torch.full(
        (rows,), model.z_table.zs.index(14), device="cuda", dtype=torch.long
    )
    silicon[8631:] = 0
    ids = torch.tensor(
        [model.z_table.zs.index(z) for z in (1, 6, 8, 14)], device="cuda"
    )
    mixed = ids[torch.randperm(rows, device="cuda") % 4]
    result = {
        name: benchmark(skips, maps, x, grad, e)
        for name, e in (("si_with_padding", silicon), ("mixed_four", mixed))
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
