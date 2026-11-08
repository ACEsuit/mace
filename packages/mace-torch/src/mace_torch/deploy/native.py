"""Ahead-of-time energy and edge gradients for the native LAMMPS pair style."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import torch
from ase.data import chemical_symbols
from mace_core.clebsch_gordan.irreps import Irreps
from torch import Tensor, nn

from mace_torch.deploy.grouped_skip import lower_scalar_skips
from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.kernels import ops
from mace_torch.kernels.contraction import symmetric_contraction_grouped_aten


@torch.library.custom_op("mace_native::halo", mutates_args=())
def _halo(features: Tensor) -> Tensor:
    """One-domain export exemplar. LAMMPS supplies the actual communication."""
    return features.clone()


@torch.library.custom_op("mace_native::halo_adjoint", mutates_args=())
def _halo_adjoint(adjoint: Tensor) -> Tensor:
    return adjoint.clone()


@_halo.register_fake
@_halo_adjoint.register_fake
def _same_shape(features):
    return torch.empty_like(features)


def _reverse(ctx, gradient):
    return _halo_adjoint(gradient)


_halo.register_autograd(_reverse)


def _exchange_inputs(module, inputs):
    return (_halo(inputs[0]), *inputs[1:])


def reference_lowerings():
    """The registered reference kernels' ATen implementations for native export."""
    return {
        torch.ops.mace.segment_sum.default: ops.segment_sum_aten,
        torch.ops.mace.channelwise_tp_conv.default: ops.channelwise_tp_conv_aten,
        torch.ops.mace.equivariant_linear.default: ops.equivariant_linear_aten,
        torch.ops.mace.symmetric_contraction.default: ops.symmetric_contraction_aten,
        torch.ops.mace.symmetric_contraction_grouped.default: (
            symmetric_contraction_grouped_aten
        ),
    }


class _SiteEnergy(nn.Module):
    attributes: Tensor
    numbers: Tensor

    def __init__(self, deployed):
        super().__init__()
        self.model = deployed.model
        self.head = deployed.heads.index(deployed.select_head())
        self.register_buffer("numbers", torch.tensor(deployed.z_table.zs))
        self.register_buffer(
            "attributes",
            torch.eye(
                len(deployed.z_table.zs), dtype=next(self.model.parameters()).dtype
            ),
        )
        if deployed.manifest.halo_exchange:
            for layer in self.model.backbone.interactions[1:]:
                layer.register_forward_pre_hook(_exchange_inputs)

    def forward(self, vectors, edge_index, elements, owned):
        graph = {
            "positions": vectors.new_zeros((len(elements), 3)),
            "shifts": vectors,
            "edge_index": edge_index,
            "node_attrs": self.attributes.index_select(0, elements),
            "node_atomic_numbers": self.numbers.index_select(0, elements),
            "batch": torch.zeros_like(elements),
            "ptr": elements.new_tensor([0, len(elements)]),
            "head": elements.new_tensor([self.head]),
        }
        site = self.model(graph).node_energies
        gradient = torch.autograd.grad((site * owned).sum(), vectors)[0]
        return site.detach(), gradient.detach()


def compile_native(bundle: Path, *, nodes: int, edges: int, device: str) -> None:
    """Compile a fixed-capacity package, with one node reserved for padding.

    The native caller supplies receiver-minus-sender displacements [E,3] in Å.
    The gradient of the owned site energies is in eV/Å in that same edge order.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    if nodes < 3 or edges < 1:
        raise DeployError("native export needs at least 3 node slots and 1 edge slot")
    if device not in {"cpu", "cuda"}:
        raise DeployError("native export device must be cpu or cuda")
    deployed = load_deployed(bundle, device=device)
    manifest = deployed.manifest
    assert manifest is not None
    backend = manifest.backend or "reference"
    if backend not in {"reference", "cueq"}:
        raise DeployError(f"native export has no kernel bindings for {backend!r}")
    if backend == "cueq" and device != "cuda":
        raise DeployError(
            "native cuEq export requires CUDA and the compiled cuEq operators"
        )
    if backend == "cueq":
        version = files("cuequivariance_ops").joinpath("VERSION")
        if version.read_text().strip() != "0.10.0":
            raise DeployError(
                "the native cuEq bindings require cuequivariance_ops 0.10.0"
            )
    if not manifest.halo_exchange and manifest.num_interactions > 1:
        raise DeployError("multi-layer native export requires --halo-exchange")
    model = _SiteEnergy(deployed).to(device).eval().requires_grad_(False)
    dtype = next(model.parameters()).dtype
    # Keep the probe geometry independent of the rank capacity. Repeating a
    # sub-Å bond thousands of times produces unphysical densities in trained
    # models. Real LAMMPS steps put unused edges beyond the cutoff instead.
    positions = (
        torch.tensor(
            [[0, 0, 0], [0.48, 0.02, 0.03], [0.12, 0.45, -0.05]],
            dtype=dtype,
            device=device,
        )[: nodes - 1]
        * manifest.r_max
    )
    pairs = torch.triu_indices(len(positions), len(positions), offset=1, device=device)
    pairs = torch.stack((pairs, pairs.flip(0)), dim=2).flatten(1)[:, :edges]
    index = torch.full((2, edges), nodes - 1, dtype=torch.long, device=device)
    index[:, : pairs.shape[1]] = pairs
    vectors = positions.new_zeros((edges, 3))
    vectors[:, 0] = 2 * manifest.r_max
    vectors[: pairs.shape[1]] = positions[pairs[1]] - positions[pairs[0]]
    elements = torch.arange(nodes, device=device) % len(deployed.z_table.zs)
    owned = (torch.arange(nodes, device=device) < len(positions)).to(dtype)
    args = (vectors, index, elements, owned)
    # A constant captured derivative could pass at its tracing geometry.
    # Exercise a second geometry, species ordering and ownership mask too.
    shifted = vectors.clone()
    shifted[: pairs.shape[1]] *= vectors.new_tensor([-1.08, -0.94, -1.03])
    partial = owned.clone()
    partial[0] = 0
    variants = (args, (shifted, index.flip(0), elements.roll(1), partial))
    # Keep the original operator values and derivatives as a separate check
    # on the lowering before the strict eager-versus-capture check.
    expected_samples = [
        model(sample[0].detach().requires_grad_(), *sample[1:]) for sample in variants
    ]
    grouped_skips = lower_scalar_skips(model.model.backbone.interactions)
    joint = make_fx(model)(vectors.detach().requires_grad_(), *args[1:])
    aten = make_fx(joint, decomposition_table=reference_lowerings())(*args)
    program = torch.export.export(aten, args, strict=True)
    names = {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    if (
        manifest.num_interactions > 1
        and not {"mace_native.halo.default", "mace_native.halo_adjoint.default"}
        <= names
    ):
        raise DeployError("export did not preserve both MPI communication operators")
    vendor_ops = sorted(name for name in names if name.startswith("cuequivariance."))
    supported = {
        "cuequivariance." + name + ".default"
        for name in (
            "uniform_1d",
            "segmented_transpose",
            "fused_tensor_product",
            "fused_tensor_product_bwd",
        )
    }
    if backend == "cueq" and (not vendor_ops or set(vendor_ops) - supported):
        raise DeployError(
            f"native cuEq requires supported vendor operators; captured {vendor_ops}"
        )
    exported = program.module()
    for sample, expected in zip(variants, expected_samples, strict=True):
        if grouped_skips:
            lowered = model(sample[0].detach().requires_grad_(), *sample[1:])
            for value, original in zip(lowered, expected, strict=True):
                # Site energies can be stored in float64 after a float32 skip.
                # Compare the two contraction orders at their compute precision.
                torch.testing.assert_close(value.to(dtype), original.to(dtype))
            expected = lowered
        with torch.no_grad():
            actual = exported(*sample)
        for tensor, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(tensor, reference)
    metadata = {
        "mace_native_version": "1.0",
        "mace_nodes": str(nodes),
        "mace_edges": str(edges),
        "mace_cutoff": str(manifest.r_max),
        "mace_species": " ".join(chemical_symbols[z] for z in manifest.species),
        "mace_atomic_numbers": " ".join(str(z) for z in manifest.species),
        "mace_device": device,
        "mace_dtype": manifest.dtype,
        "mace_backend": backend,
        "mace_grouped_scalar_skips": str(grouped_skips),
        "mace_cueq_operators": " ".join(vendor_ops),
        "mace_comm_width": str(
            max(
                (
                    Irreps.parse(s).dimension
                    for s in model.model.backbone.layer_irreps[:-1]
                ),
                default=0,
            )
        ),
    }
    torch._inductor.aoti_compile_and_package(
        program,
        package_path=str(bundle / "native.pt2"),
        inductor_configs={"cpp.threads": 1, "aot_inductor.metadata": metadata},
    )
