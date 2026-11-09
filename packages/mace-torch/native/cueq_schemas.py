"""Record the installed vendor operator schemas at plugin build time."""

import sys
from importlib import import_module
from pathlib import Path

import torch

cuequivariance_ops = import_module("cuequivariance_ops")
root = Path(sys.argv[2]).resolve()
origin = cuequivariance_ops.__file__
if (
    not origin
    or Path(origin).resolve().parent != root
    or (root / "VERSION").read_text().strip() != "0.10.0"
):
    raise RuntimeError(
        "CUEQ_OPS_DIR must match the build interpreter's cuequivariance_ops 0.10.0"
    )
bindings = {
    "uniform_1d": "uniform",
    "segmented_transpose": "transpose",
    "fused_tensor_product": "fused_forward",
    "fused_tensor_product_bwd": "fused_backward",
}
for module in ("uniform_1d", "fused_tensor_product", "segmented_transpose"):
    import_module(f"cuequivariance_ops_torch.{module}")

output = Path(sys.argv[1])
with output.open("w") as header, output.with_name("cueq_args.inc").open("w") as args:
    for name, kernel in bindings.items():
        op = getattr(torch.ops.cuequivariance, name).default._schema
        schema = str(op).split("::", 1)[1]
        header.write(
            f'm.def(R"schema({schema})schema", '
            f"torch::CppFunction::makeFromBoxedFunction<&{kernel}>());\n"
        )
        fields = ", ".join(arg.name for arg in op.arguments)
        args.write(f"struct {name} {{ enum {{ {fields} }}; }};\n")
