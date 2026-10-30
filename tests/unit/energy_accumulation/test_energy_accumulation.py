"""Opt-in energy reduction precision, using tiny offline CPU models.

The network and its outputs keep their original dtype. Only the reported
graph-energy reductions are widened; the interaction energy used for forces
and stress follows the existing path.
"""

import copy

import numpy as np
import pytest
import torch
from ase import Atoms

# MACE establishes e3nn's checkpoint-loading compatibility before import.
from mace import modules

from e3nn import o3
from e3nn.util import jit

from mace.calculators import MACECalculator
from mace.modules.models import ScaleShiftMACE
from mace.tools import torch_tools
from mace.tools.scatter import scatter_sum
from tests.golden.anchors import anchor_batch


def _tiny_model(*, pair_repulsion=False):
    # Neither construction nor graph creation changes the caller's RNG/dtype.
    with torch.random.fork_rng(devices=[]), torch_tools.default_dtype("float32"):
        torch.manual_seed(9107)
        return ScaleShiftMACE(
            r_max=3.0,
            num_bessel=3,
            num_polynomial_cutoff=5,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("4x0e + 4x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([-12345.6789, -98765.4321]),
            avg_num_neighbors=3.0,
            atomic_numbers=[1, 8],
            correlation=2,
            atomic_inter_scale=1.25,
            atomic_inter_shift=0.125,
            pair_repulsion=pair_repulsion,
        )


@pytest.fixture(name="structures")
def fixture_structures():
    return [
        Atoms(
            "OH2",
            positions=[[0, 0, 0], [0.95, 0, 0], [-0.25, 0.92, 0]],
            cell=[8, 8, 8],
            pbc=True,
        ),
        Atoms(
            "O2H2",
            positions=[[0, 0, 0], [1.4, 0, 0], [-0.7, 0.7, 0], [2.1, 0.7, 0]],
            cell=[9, 9, 9],
            pbc=True,
        ),
    ]


def _forward_with_terms(model, batch, **kwargs):
    """Observe native block outputs; do not reimplement the model forward."""
    terms = {}
    handles = []

    def capture(name):
        def hook(_module, _args, output):
            terms[name] = output.detach().clone()

        return hook

    for name, block in (
        ("e0", model.atomic_energies_fn),
        ("interaction", model.scale_shift),
        ("embedding", getattr(model, "embedding_readout", None)),
    ):
        if block is not None:
            handles.append(block.register_forward_hook(capture(name)))
    try:
        output = model(batch.clone().to_dict(), **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    node_heads = batch["head"][batch["batch"]]
    nodes = torch.arange(len(batch["batch"]))
    terms["e0"] = terms["e0"][nodes, node_heads]
    if "embedding" in terms:
        terms["embedding"] = terms["embedding"][nodes, node_heads]
    return output, terms


def _grouped_sum(values, batch, dtype=torch.float64):
    """Independent per-structure sums, including different-size graphs."""
    return torch.stack(
        [
            values[batch["batch"] == i].sum(dtype=dtype)
            for i in range(batch["ptr"].numel() - 1)
        ]
    )


def _wide_oracle(terms, batch, dtype):
    energy = _grouped_sum(terms["e0"], batch)
    if "embedding" in terms:
        energy = energy + _grouped_sum(terms["embedding"], batch)
    return (energy + _grouped_sum(terms["interaction"], batch)).to(dtype)


class _ZeroJointEmbedding(torch.nn.Module):
    def forward(self, batch, features):
        return features["extra"]


class _ExtraReadout(torch.nn.Module):
    def forward(self, node_feats, node_heads):
        return node_feats[:, :1] * 0.125 + 3.25


def _add_embedding(model, batch):
    model.embedding_specs = {"extra": {}}
    model.joint_embedding = _ZeroJointEmbedding()
    model.embedding_readout = _ExtraReadout()
    n_features = model.node_embedding(batch["node_attrs"]).shape[-1]
    batch["extra"] = torch.zeros(
        (len(batch["batch"]), n_features), dtype=batch["positions"].dtype
    )


@pytest.mark.parametrize("with_embedding", [False, True])
def test_default_and_none_keep_legacy_numeric_semantics(structures, with_embedding):
    model = _tiny_model()
    batch = anchor_batch(model, structures, torch.float32)
    if with_embedding:
        _add_embedding(model, batch)
    default, terms = _forward_with_terms(model, batch, compute_stress=True)
    explicit_none = model(
        batch.clone().to_dict(), compute_stress=True, energy_accumulation_dtype=None
    )
    for name, value in default.items():
        if value is None:
            assert explicit_none[name] is None
        else:
            assert torch.equal(value, explicit_none[name]), name

    # The legacy total is E0 (+ optional embedding) + interaction_energy,
    # with separate FP32 scatter reductions, not a reduced node_energy.
    num_graphs = len(structures)
    legacy = scatter_sum(terms["e0"], batch["batch"], dim=0, dim_size=num_graphs)
    if with_embedding:
        legacy = legacy + scatter_sum(
            terms["embedding"], batch["batch"], dim=0, dim_size=num_graphs
        )
    assert torch.equal(default["energy"], legacy + default["interaction_energy"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_embedding", [False, True])
def test_wide_reduction_is_per_graph_and_preserves_dtype(
    structures, dtype, with_embedding
):
    model = _tiny_model().to(dtype)
    batch = anchor_batch(model, structures, dtype)
    assert batch["ptr"].tolist() == [0, 3, 7]
    if with_embedding:
        _add_embedding(model, batch)
    output, terms = _forward_with_terms(
        model, batch, energy_accumulation_dtype="float64"
    )
    assert output["energy"].dtype is dtype
    assert output["energy"].shape == (2,)
    torch.testing.assert_close(
        output["energy"], _wide_oracle(terms, batch, dtype), rtol=0, atol=0
    )


def test_wide_reduction_improves_known_cancellation_before_narrowing(structures):
    model = _tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.atomic_energies_fn.atomic_energies.fill_(1.0e8)
        model.scale_shift.scale.fill_(1.0)
        model.scale_shift.shift.fill_(-1.0e8 + 16.0)
    batch = anchor_batch(model, structures[:1], torch.float32)
    legacy = model(batch.clone().to_dict(), compute_force=False)["energy"]
    improved = model(
        batch.clone().to_dict(),
        compute_force=False,
        energy_accumulation_dtype="float64",
    )["energy"]
    # All operands are exactly representable in FP32; the known physical
    # sum is 3 * (1e8 + (-1e8 + 16)) = 48 eV. This isolates accumulation
    # error from weight/input quantization and arbitrary NN parity claims.
    exact = torch.tensor([48.0], dtype=torch.float64)
    assert improved.dtype is torch.float32
    assert torch.equal(improved.double(), exact)
    assert torch.max(torch.abs(legacy.double() - exact)).item() >= 16.0


def test_zbl_is_counted_once_and_force_stress_paths_are_unchanged(structures):
    model = _tiny_model(pair_repulsion=True)
    batch = anchor_batch(model, structures, torch.float32)
    legacy = model(batch.clone().to_dict(), compute_stress=True)
    improved, terms = _forward_with_terms(
        model, batch, compute_stress=True, energy_accumulation_dtype="float64"
    )
    for name in ("forces", "stress", "virials", "interaction_energy", "node_energy"):
        assert torch.equal(legacy[name], improved[name]), name
    assert torch.equal(improved["energy"], _wide_oracle(terms, batch, torch.float32))
    without_zbl = _tiny_model()(batch.clone().to_dict(), compute_force=False)
    assert not torch.equal(improved["node_energy"], without_zbl["node_energy"])


def test_wide_reported_energy_remains_differentiable(structures):
    model = _tiny_model()
    batch = anchor_batch(model, structures, torch.float32)
    legacy = model(batch.clone().to_dict(), compute_force=False)
    legacy["energy"].sum().backward()
    legacy_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    output = model(
        batch.to_dict(), compute_force=False, energy_accumulation_dtype="float64"
    )
    output["energy"].sum().backward()
    grads = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad) > 0 for grad in grads)
    for name, parameter in model.named_parameters():
        if name in legacy_grads:
            torch.testing.assert_close(
                parameter.grad, legacy_grads[name], rtol=0, atol=0
            )
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("strategy", ["float16", "float32", "invalid"])
def test_model_rejects_invalid_strategy(structures, strategy):
    model = _tiny_model()
    batch = anchor_batch(model, structures, torch.float32)
    with pytest.raises(ValueError, match="None or 'float64'"):
        model(batch.to_dict(), energy_accumulation_dtype=strategy)


def test_torchscript_keeps_default_and_opt_in_energy_results(structures):
    eager = _tiny_model()
    batch = anchor_batch(eager, structures, torch.float32)
    expected_default = eager(batch.clone().to_dict())
    expected_wide = eager(batch.clone().to_dict(), energy_accumulation_dtype="float64")
    scripted = jit.compile(copy.deepcopy(eager))
    for kwargs, expected in (
        ({}, expected_default),
        ({"energy_accumulation_dtype": "float64"}, expected_wide),
    ):
        actual = scripted(batch.clone().to_dict(), **kwargs)
        for name in ("energy", "node_energy", "interaction_energy"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    # Eager force/stress identity is covered separately. JIT may reassociate
    # backward operations and does not guarantee bitwise force parity.


@pytest.mark.parametrize("strategy", [None, "float64"])
def test_ase_forwards_strategy_only_when_explicit_and_retains_values(
    structures, strategy
):
    model = _tiny_model()
    batch = anchor_batch(model, structures[:1], torch.float32)
    kwargs = {} if strategy is None else {"energy_accumulation_dtype": strategy}
    expected = model(batch.to_dict(), **kwargs)
    seen = []

    def record(_module, _args, forwarded):
        seen.append(dict(forwarded))

    handle = model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        calculator = MACECalculator(
            models=model,
            device="cpu",
            default_dtype="float32",
            energy_accumulation_dtype=strategy,
        )
        atoms = structures[0].copy()
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    finally:
        handle.remove()
    assert ("energy_accumulation_dtype" in seen[-1]) is (strategy is not None)
    if strategy is not None:
        assert seen[-1]["energy_accumulation_dtype"] == strategy
    assert energy == float(expected["energy"].detach()[0])
    np.testing.assert_array_equal(forces, expected["forces"].detach().numpy())


class _OverriddenScaleShift(ScaleShiftMACE):
    def forward(self, data, **kwargs):
        return super().forward(data, **kwargs)


def test_ase_rejects_a_subclass_that_overrides_the_energy_path():
    model = _tiny_model()
    model.__class__ = _OverriddenScaleShift
    with pytest.raises(ValueError, match="ScaleShiftMACE.forward"):
        MACECalculator(
            models=model,
            device="cpu",
            default_dtype="float32",
            energy_accumulation_dtype="float64",
        )


def test_ase_rejects_an_invalid_strategy():
    with pytest.raises(ValueError, match="None or 'float64'"):
        MACECalculator(
            models=_tiny_model(),
            device="cpu",
            default_dtype="float32",
            energy_accumulation_dtype="float16",
        )
