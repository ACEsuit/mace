"""Profile MACE-MP-0 small's two skip connections and one force adjoint in isolation."""

import argparse
import json

import torch
from mace_torch.deploy.loader import load_deployed


def sample(skips, features, attributes, gradient):
    def step():
        skips[0](features.detach(), attributes)
        output = skips[1](features, attributes)
        torch.autograd.grad(output, features, grad_outputs=gradient)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(6):
            step()
        torch.cuda.synchronize()
    events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    forward = [
        event.device_time_total / 1000
        for event in events
        if "fused_tensor_product_fwd_fallback_kernel" in event.name
    ]
    adjoint = [
        event.device_time_total / 1000
        for event in events
        if "fused_tensor_product_fallback_bwd_kernel" in event.name
    ]
    if (len(forward), len(adjoint)) != (12, 6):
        raise RuntimeError(
            "expected two native cuEq forward kernels and one adjoint per sample"
        )
    return {
        "forward_kernel_ms": forward,
        "backward_kernel_ms": adjoint,
        "total_ms": (sum(forward) + sum(adjoint)) / 6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("rows", type=int, nargs="+")
    args = parser.parse_args()
    if min(args.rows) < 1:
        parser.error("row counts must be positive")
    deployed = load_deployed(args.bundle, device="cuda")
    skips = [
        layer.skip.eval().requires_grad_(False)
        for layer in deployed.model.backbone.interactions
    ]
    if len(skips) != 2 or len(deployed.z_table.zs) != 89:
        parser.error("use the 89-element MACE-MP-0 small bundle from this benchmark")

    torch.manual_seed(711)
    features = torch.randn(max(args.rows), 128, device="cuda", dtype=torch.float32)
    gradient = torch.randn_like(features)
    attributes = features.new_zeros((max(args.rows), len(deployed.z_table.zs)))
    attributes[:, deployed.z_table.zs.index(14)] = 1
    results = {}
    for n in args.rows:
        x = features[:n].detach().requires_grad_()
        results[n] = sample(skips, x, attributes[:n], gradient[:n])
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
