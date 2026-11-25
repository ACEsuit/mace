"""The attention-residual interaction, which `--interaction` offers and nothing built.

`RealAgnosticAttResidualInteractionBlock` is in `interaction_classes` and in the
`--interaction` choices, so a user can train with it today. Nothing in the tree
constructed it, which the functionality inventory surfaced as a gap: a block
reachable from the command line that no test has ever run.

It does work. This pins that, and pins it in the two ways that can fail
separately -- the layers really are that class (a wiring mistake would silently
give the default residual block), and the forward produces finite energies and
forces of the right shape.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import default_dtype

ATT_RESIDUAL = "RealAgnosticAttResidualInteractionBlock"
NONLINEAR_RESIDUAL = "RealAgnosticResidualNonLinearInteractionBlock"
PLAIN_RESIDUAL = "RealAgnosticResidualInteractionBlock"
DEFERRED_CUTOFF_BLOCKS = [NONLINEAR_RESIDUAL, ATT_RESIDUAL]


@pytest.fixture(name="water_batch")
def water_batch_fixture():
    table = AtomicNumberTable([1, 8])
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.3, 0.9, 0.0]])
    config = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={"energy": 0.0, "forces": np.zeros((len(atoms), 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)), table, len(atoms)


def _model(name, table):
    torch.manual_seed(0)
    return modules.MACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[name],
        interaction_cls_first=modules.interaction_classes[name],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )


def test_the_attention_residual_interaction_builds_and_runs(water_batch):
    batch, table, num_atoms = water_batch
    model = _model(ATT_RESIDUAL, table)

    wanted = modules.interaction_classes[ATT_RESIDUAL]
    assert [type(layer) for layer in model.interactions] == [wanted, wanted], (
        "the model was built with something other than the block under test, so "
        "this test would pass for the default residual block too"
    )

    out = model(batch.to_dict(), training=False)
    energy = out["energy"]
    forces = out["forces"]
    assert energy.shape == (1,)
    assert forces.shape == (num_atoms, 3)
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()


def test_the_attention_residual_interaction_is_not_the_plain_one(water_batch):
    """Same seed, same shapes, different numbers.

    Both blocks are residual and both accept the same arguments, so a mix-up
    between them changes no shape and raises nothing. The energies have to
    differ, or `--interaction` is offering two names for one model.
    """
    batch, table, _ = water_batch
    att = _model(ATT_RESIDUAL, table)(batch.to_dict(), training=False)["energy"]
    plain = _model(PLAIN_RESIDUAL, table)(batch.to_dict(), training=False)["energy"]
    assert not torch.allclose(att, plain), (
        f"the attention-residual block reproduced the plain residual energy "
        f"({float(att[0]):.12f}), so it is not doing anything of its own"
    )


def _trimer_batch(distance: float, table: AtomicNumberTable, r_max: float = 5.0):
    atoms = Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + distance, 0.0, 0.0],
        ],
    )
    config = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={},
        property_weights={},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=r_max)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)).to_dict()


@pytest.mark.parametrize("block_name", DEFERRED_CUTOFF_BLOCKS)
def test_deferred_cutoff_blocks_are_continuous_at_r_max_by_default(block_name):
    """Default construction (without passing apply_cutoff=False) must still
    give continuous energies and forces as a neighbour crosses r_max."""
    table = AtomicNumberTable([1, 8])
    r_max = 5.0
    with default_dtype(torch.float64):
        model = _model(block_name, table)
        assert model.apply_cutoff is False
        assert model.radial_embedding.apply_cutoff is False

        inside = model(_trimer_batch(r_max - 1e-6, table, r_max), training=False)
        outside = model(_trimer_batch(r_max + 1e-6, table, r_max), training=False)

    step = abs(float((inside["energy"][0] - outside["energy"][0]).detach()))
    assert step < 1e-9, f"{block_name} energy jumped by {step:.3e} eV at r_max"
    np.testing.assert_allclose(
        inside["forces"].detach().numpy(),
        outside["forces"].detach().numpy(),
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.parametrize("block_name", DEFERRED_CUTOFF_BLOCKS)
def test_deferred_cutoff_envelope_is_applied_once(block_name):
    """Edge basis goes without the envelope; the interaction block applies it
    once to the convolution weights (and to edge_density when present)."""
    table = AtomicNumberTable([1, 8])
    r_max = 5.0
    seen = {}

    def keep(key, position=None):
        def hook(module, args, output=None):  # pylint: disable=unused-argument
            value = output if position is None else args[position]
            seen.setdefault(key, []).append(value)

        return hook

    with default_dtype(torch.float64):
        model = _model(block_name, table)
        atoms = Atoms(
            "OH2",
            positions=[[0.0, 0.0, 0.0], [2.5, 0.4, 0.0], [r_max - 0.5, -0.3, 0.2]],
        )
        config = data.Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            properties={},
            property_weights={},
        )
        atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=r_max)
        batch = next(
            iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
        ).to_dict()
        for inter in model.interactions:
            inter.register_forward_pre_hook(
                lambda module, args, kwargs: seen.setdefault("call", []).append(kwargs),
                with_kwargs=True,
            )
            inter.conv_tp_weights.register_forward_hook(keep("weights"))
            inter.conv_tp.register_forward_pre_hook(keep("weights_used", 2))
            if hasattr(inter, "density_fn"):
                inter.density_fn.register_forward_hook(keep("density_fn"))
                inter.linear_1.register_forward_hook(keep("linear_1_out"))
                inter.linear_res.register_forward_hook(keep("linear_res_out"))
                inter.equivariant_nonlin.register_forward_pre_hook(
                    keep("nonlin_in", 0)
                )
        model(dict(batch), training=False)

        _, lengths = get_edge_vectors_and_lengths(
            batch["positions"], batch["edge_index"], batch["shifts"]
        )
        envelope = model.radial_embedding.cutoff_fn(lengths)
        bare = model.radial_embedding.bessel_fn(lengths)

    assert len(seen["call"]) == 2
    assert torch.all(envelope < 0.99) and torch.all(envelope > 0.0)
    for layer, call in enumerate(seen["call"]):
        torch.testing.assert_close(call["edge_feats"], bare, rtol=0, atol=0)
        torch.testing.assert_close(call["cutoff"], envelope, rtol=0, atol=0)
        torch.testing.assert_close(
            seen["weights_used"][layer], seen["weights"][layer] * envelope
        )
        if block_name == NONLINEAR_RESIDUAL:
            inter = model.interactions[layer]
            density = scatter_sum(
                torch.tanh(seen["density_fn"][layer] ** 2) * envelope,
                index=batch["edge_index"][1],
                dim=0,
                dim_size=batch["positions"].shape[0],
            )
            expected = (
                seen["linear_1_out"][layer] / (density * inter.beta + inter.alpha)
                + seen["linear_res_out"][layer]
            )
            torch.testing.assert_close(seen["nonlin_in"][layer], expected)


@pytest.mark.parametrize(
    "first_name,rest_name",
    [
        (NONLINEAR_RESIDUAL, PLAIN_RESIDUAL),
        (PLAIN_RESIDUAL, NONLINEAR_RESIDUAL),
    ],
)
def test_mixed_deferred_cutoff_blocks_force_apply_cutoff_false(first_name, rest_name):
    table = AtomicNumberTable([1, 8])
    r_max = 5.0
    with default_dtype(torch.float64):
        torch.manual_seed(0)
        model = modules.MACE(
            r_max=r_max,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=modules.interaction_classes[rest_name],
            interaction_cls_first=modules.interaction_classes[first_name],
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("16x0e + 16x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([1.0, 3.0]),
            avg_num_neighbors=8,
            atomic_numbers=table.zs,
            correlation=3,
            radial_type="bessel",
        )
        assert model.apply_cutoff is False
        assert model.radial_embedding.apply_cutoff is False
        inside = model(_trimer_batch(r_max - 1e-6, table, r_max), training=False)
        outside = model(_trimer_batch(r_max + 1e-6, table, r_max), training=False)
    step = abs(float((inside["energy"][0] - outside["energy"][0]).detach()))
    assert step < 1e-9, f"({first_name}, {rest_name}) jumped by {step:.3e} eV at r_max"


@pytest.mark.parametrize(
    "model_cls,atomic_energies",
    [
        (modules.AtomicDipolesMACE, None),
        (modules.AtomicDielectricMACE, None),
        (modules.EnergyDipolesMACE, np.zeros(2)),
    ],
)
def test_dipole_and_dielectric_models_defer_cutoff_for_nonlinear_blocks(
    model_cls, atomic_energies
):
    common_kwargs = dict(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
    )
    nonlinear = modules.interaction_classes[NONLINEAR_RESIDUAL]
    m_deferred = model_cls(
        **common_kwargs,
        interaction_cls=nonlinear,
        interaction_cls_first=nonlinear,
        apply_cutoff=True,
    )
    assert m_deferred.apply_cutoff is False
    assert m_deferred.radial_embedding.apply_cutoff is False

    plain = modules.interaction_classes[PLAIN_RESIDUAL]
    m_plain = model_cls(
        **common_kwargs,
        interaction_cls=plain,
        interaction_cls_first=plain,
        apply_cutoff=True,
    )
    assert m_plain.apply_cutoff is True
    assert m_plain.radial_embedding.apply_cutoff is True

