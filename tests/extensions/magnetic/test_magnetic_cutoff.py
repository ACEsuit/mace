"""The magnetic energy is continuous as a neighbour crosses r_max.

Both magnetic interactions build their radial weights from the edge basis
concatenated with the sender's moment features. The moment features do not
vanish at r_max, so an envelope inside the edge basis is not enough: a
neighbour's message would stay finite up to r_max and disappear when the
neighbour leaves the neighbour list, and the energy would step. So the edge
basis goes without the envelope, and the interactions apply it once, to the
convolution weights and to the edge density.

An Fe dimer is the smallest structure where one neighbour crosses r_max: just
inside, there is one edge each way; just outside, there are none.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data
from mace.modules import interaction_classes
from mace.modules.extensions import MagneticScaleShiftMACE
from mace.modules.utils import get_edge_vectors_and_lengths
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import default_dtype

R_MAX = 3.5
TABLE = AtomicNumberTable([26])

SOC_BLOCKS = [
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
]


def _model(name):
    torch.manual_seed(0)
    return MagneticScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes[name],
        interaction_cls_first=interaction_classes[name],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(TABLE.zs)),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=[2, 2],
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=10,
        max_m_ell=2,
        use_magmom_one_body=False,
        use_reduced_cg=False,
    )


def _batch(positions, magmom):
    config = data.Configuration(
        atomic_numbers=np.full(len(positions), 26),
        positions=np.array(positions),
        properties={"magmom": np.array(magmom)},
        property_weights={"magmom": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=TABLE, cutoff=R_MAX)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)).to_dict()


def _dimer(distance, magmom):
    return _batch([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], magmom)


@pytest.mark.parametrize("name", SOC_BLOCKS)
@pytest.mark.parametrize(
    "magmom",
    [
        [[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]],
        [[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    ids=["fm", "afm", "zero"],
)
def test_energy_and_magforces_are_continuous_at_r_max(name, magmom):
    with default_dtype(torch.float64):
        model = _model(name)
        inside = model(_dimer(R_MAX - 1e-6, magmom), training=False)
        outside = model(_dimer(R_MAX + 1e-6, magmom), training=False)

    # Without the envelope on the weights, the blocks step by 1e-2 to 1 eV here.
    step = float(inside["energy"][0] - outside["energy"][0])
    assert abs(step) < 1e-9, f"energy steps by {step:.3e} eV at r_max"
    np.testing.assert_allclose(
        inside["magforces"].detach().numpy(),
        outside["magforces"].detach().numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        inside["forces"].detach().numpy(),
        outside["forces"].detach().numpy(),
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.parametrize("name", SOC_BLOCKS)
def test_the_envelope_is_applied_once(name):
    """The edge basis without it, the convolution weights and the edge density
    with it once, the moment-space weights without it."""
    seen = {}

    def keep(key, position=None):
        def hook(module, args, output=None):  # pylint: disable=unused-argument
            value = output if position is None else args[position]
            seen.setdefault(key, []).append(value)

        return hook

    with default_dtype(torch.float64):
        model = _model(name)
        batch = _batch(
            [[0.0, 0.0, 0.0], [2.1, 0.4, 0.0], [R_MAX - 0.05, -0.3, 0.2]],
            [[0.3, 0.1, 2.1], [-0.2, 0.4, -1.9], [0.5, -0.3, 2.0]],
        )
        for inter in model.interactions:
            inter.register_forward_pre_hook(
                lambda module, args, kwargs: seen.setdefault("call", []).append(kwargs),
                with_kwargs=True,
            )
            inter.conv_tp_weights.register_forward_hook(keep("weights"))
            inter.conv_tp.register_forward_pre_hook(keep("weights_used", 2))
            inter.conv_tp_weights_magmom.register_forward_hook(keep("magmom_weights"))
            inter.magmom_conv_tp.register_forward_pre_hook(
                keep("magmom_weights_used", 2)
            )
            inter.density_fn.register_forward_hook(keep("density_fn"))
            inter.magmom_linear.register_forward_hook(keep("message"))
            # the first module to see the message divided by (density + 1)
            normalised = getattr(inter, "magmom_skip_tp", inter.reshape)
            normalised.register_forward_pre_hook(keep("normalised", 0))
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
        assert seen["magmom_weights_used"][layer] is seen["magmom_weights"][layer]
        density = scatter_sum(
            torch.tanh(seen["density_fn"][layer] ** 2) * envelope,
            index=batch["edge_index"][1],
            dim=0,
            dim_size=batch["positions"].shape[0],
        )
        torch.testing.assert_close(
            seen["message"][layer], seen["normalised"][layer] * (density + 1)
        )
