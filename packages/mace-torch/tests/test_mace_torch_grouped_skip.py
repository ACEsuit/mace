"""Fixed-capacity scalar skips must follow the species of the current graph."""

import pytest
import torch
from mace_torch.deploy.grouped_skip import GroupedScalarSkip
from torch.fx.experimental.proxy_tensor import make_fx


@pytest.mark.parametrize(("nodes", "species"), [(4, 1), (19, 89), (193, 2), (193, 9)])
def test_exported_skip_and_adjoint_with_changing_composition(nodes, species):
    torch.manual_seed(748)
    weights = torch.randn(species, 5, 7)
    skip = GroupedScalarSkip(weights)
    x = torch.randn(nodes, 7)
    seed = torch.randn(nodes, 5)

    def joint(features, attributes, gradient):
        values = skip(features, attributes)
        return values, torch.autograd.grad(values, features, gradient)[0]

    # Trace a single-species rank, then change counts, row order, and which
    # species are absent. The 193-node case crosses several 64-row tiles.
    elements = torch.full((nodes,), species - 1, dtype=torch.long)
    attrs = torch.nn.functional.one_hot(elements, species).to(x)
    graph = make_fx(joint)(x.detach().requires_grad_(), attrs, seed)
    compiled = torch.export.export(graph, (x, attrs, seed), strict=True).module()
    for labels in (
        elements,
        torch.arange(nodes) % species,
        torch.randint(0, species, (nodes,)),
        torch.where(torch.arange(nodes) < 65, 0, species - 1),
    ):
        attributes = torch.nn.functional.one_hot(labels, species).to(x)
        actual, adjoint = compiled(x, attributes, seed)
        selected = weights[labels]
        torch.testing.assert_close(actual, (selected @ x.unsqueeze(-1)).squeeze(-1))
        torch.testing.assert_close(
            adjoint, (selected.transpose(1, 2) @ seed.unsqueeze(-1)).squeeze(-1)
        )


def test_lowering_respects_canonical_weights_irreps_and_precision():
    from mace_core.kernels.specs import FullyConnectedTPSpec
    from mace_torch.backends.precision import cast_at_boundary
    from mace_torch.backends.reference import ReferenceFullyConnectedTP
    from mace_torch.deploy.grouped_skip import lower_scalar_skips

    def make(inp="3x0e", out="7x0e", **kwargs):
        kernel_spec = FullyConnectedTPSpec(
            irreps_in1=inp,
            irreps_in2="5x0e",
            irreps_out=out,
            precision="float32",
            **kwargs,
        )
        return ReferenceFullyConnectedTP(kernel_spec)

    torch.manual_seed(934)
    scalar, vector, wider, casting = (
        make(),
        make("2x1o", "2x1o"),
        make(accumulate="float64"),
        make(),
    )
    cast_at_boundary(casting, "float32", "float64")
    original = (scalar, vector, wider, casting)
    layers = torch.nn.ModuleList([torch.nn.ModuleDict({"skip": op}) for op in original])
    scalar.initialize_weights(358)
    x = torch.randn(127, 3, dtype=torch.float32, requires_grad=True)
    attrs = torch.nn.functional.one_hot(torch.randperm(127) % 5, 5).to(x)
    checkpoint_weights = scalar.weight.detach().clone()
    expected = scalar(x, attrs)
    seed = torch.randn_like(expected)
    assert lower_scalar_skips(layers) == 1
    assert all(
        layer.skip is op for layer, op in zip(layers[1:], original[1:], strict=True)
    )
    assert torch.equal(scalar.to_canonical()["weight"], checkpoint_weights)
    actual = layers[0].get_submodule("skip")(x, attrs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, seed)[0],
        torch.autograd.grad(expected, x, seed)[0],
    )


@pytest.mark.slow
def test_native_bundle_selects_and_executes_scalar_skips(tmp_path):
    from mace_torch.deploy.bundle import export_lammps
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.deploy.native import _SiteEnergy
    from mace_torch.train import run_data_stage, run_model_stage
    from mace_torch.train.checkpoint import write_model
    from test_mace_torch_full_batch import CATALOGUE, configuration

    config = configuration(tmp_path)
    config = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={"hidden_irreps": "0e", "interaction_first": "residual"}
            )
        }
    )
    data = run_data_stage(config, CATALOGUE)
    trained = run_model_stage(config, data, CATALOGUE)
    write_model(tmp_path / "source", trained.engine, trained.metadata)
    dtype = torch.get_default_dtype()
    precision = "float32" if dtype == torch.float32 else "float64"
    bundle = export_lammps(
        tmp_path / "source.json",
        halo_exchange=True,
        native_capacity=(6, 4),
        dtype=precision,
    )
    compiled = torch._inductor.aoti_load_package(str(bundle / "native.pt2"))
    assert compiled.get_metadata()["mace_grouped_scalar_skips"] == "2"
    original = _SiteEnergy(load_deployed(bundle)).eval().requires_grad_(False)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    vectors = torch.tensor(
        [[2.1, 0.2, 0.1], [-2.1, -0.2, -0.1], [1.7, 0.4, 0.2], [-1.7, -0.4, -0.2]]
    )
    owned = torch.tensor([1, 1, 0, 0, 0, 0], dtype=dtype)
    for species in (torch.zeros(6, dtype=torch.long), torch.tensor([1, 0, 1, 0, 0, 0])):
        expected = original(vectors.detach().requires_grad_(), edges, species, owned)
        actual = compiled(vectors, edges, species, owned)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value.to(dtype), reference.to(dtype))
