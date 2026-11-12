"""The magnetic energy is continuous as a neighbour crosses r_max.

Both magnetic interactions build their radial weights from the edge basis
concatenated with the sender's moment features. The moment features do not
vanish at r_max, so an envelope inside the edge basis is not enough: a
neighbour's message stays finite up to r_max and disappears when the neighbour
leaves the neighbour list, so the energy steps.

With smooth_cutoff, the default, the edge basis goes without the envelope and
the interactions apply it once, to the convolution weights and to the edge
density. The moment-space weights go without it: their tensor product is
linear in the convolution's output, which already carries it. Models with
smooth_cutoff=False, and checkpoints saved before the flag existed, keep the
envelope inside the edge basis and compute what they did before.

An Fe dimer is the smallest structure where one neighbour crosses r_max: just
inside, there is one edge each way; just outside, there are none.
"""

import os
import subprocess
import sys
from pathlib import Path

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
from mace.tools.scripts_utils import extract_model, remove_pt_head
from mace.tools.torch_tools import default_dtype

REPO_ROOT = Path(__file__).resolve().parents[3]
R_MAX = 3.5
TABLE = AtomicNumberTable([26])

SOC_BLOCKS = [
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
]

#: Three moments on three Fe atoms, every pair within r_max and one close to it.
TRIMER = (
    [[0.0, 0.0, 0.0], [2.1, 0.4, 0.0], [R_MAX - 0.05, -0.3, 0.2]],
    [[0.3, 0.1, 2.1], [-0.2, 0.4, -1.9], [0.5, -0.3, 2.0]],
)


def _model(name, heads=None, **kwargs):
    torch.manual_seed(0)
    num_heads = 1 if heads is None else len(heads)
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
        atomic_energies=np.zeros((num_heads, len(TABLE.zs))),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=[2, 2],
        gate=torch.nn.functional.silu,
        atomic_inter_shift=[0.1, -0.2][:num_heads],
        atomic_inter_scale=[1.3, 0.7][:num_heads],
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=10,
        max_m_ell=2,
        use_magmom_one_body=False,
        use_reduced_cg=False,
        heads=heads,
        **kwargs,
    )


def _saved_before_smooth_cutoff(name, **kwargs):
    """A checkpoint saved before the flag existed: built as it was then, which
    smooth_cutoff=False reproduces, and without the attribute."""
    model = _model(name, smooth_cutoff=False, **kwargs)
    del model.smooth_cutoff
    return model


#: How a model reaches this code, and whether it gets the envelope outside the
#: edge basis.
SETTINGS = {
    "default": (lambda name, **kw: _model(name, **kw), True),
    "smooth_cutoff=False": (
        lambda name, **kw: _model(name, smooth_cutoff=False, **kw),
        False,
    ),
    "smooth_cutoff=False,apply_cutoff=False": (
        lambda name, **kw: _model(name, smooth_cutoff=False, apply_cutoff=False, **kw),
        False,
    ),
    "saved_before_smooth_cutoff": (_saved_before_smooth_cutoff, False),
}


def _batch(positions, magmom, head=None, heads=None):
    config = data.Configuration(
        atomic_numbers=np.full(len(positions), 26),
        positions=np.array(positions),
        properties={"magmom": np.array(magmom)},
        property_weights={"magmom": 1.0},
        head="Default" if head is None else head,
    )
    atomic_data = data.AtomicData.from_config(
        config, z_table=TABLE, cutoff=R_MAX, heads=heads
    )
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)).to_dict()


def _dimer(distance, magmom):
    return _batch([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], magmom)


def _outputs(model, batch):
    out = model(dict(batch), training=False)
    return {key: out[key].detach().numpy() for key in ("energy", "forces", "magforces")}


def _assert_same_outputs(model, other, batch, atol=1e-12):
    expected, actual = _outputs(model, batch), _outputs(other, batch)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=0.0, atol=atol, err_msg=key)


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

    # The uncorrected blocks step by 1e-2 to 1 eV here.
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
@pytest.mark.parametrize(
    "setting", [s for s, (_, smooth) in SETTINGS.items() if not smooth]
)
def test_without_smooth_cutoff_the_energy_still_steps_at_r_max(name, setting):
    """What those models were trained as, kept: this is not a smooth model."""
    build, _ = SETTINGS[setting]
    magmom = [[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]]
    with default_dtype(torch.float64):
        model = build(name)
        inside = model(_dimer(R_MAX - 1e-6, magmom), training=False)
        outside = model(_dimer(R_MAX + 1e-6, magmom), training=False)
    step = float(inside["energy"][0] - outside["energy"][0])
    assert abs(step) > 1e-6, f"energy steps by only {step:.3e} eV at r_max"


@pytest.mark.parametrize("name", SOC_BLOCKS)
@pytest.mark.parametrize("setting", SETTINGS)
def test_the_envelope_is_applied_once(name, setting):
    """Each quantity the envelope can reach, read off the forward itself.

    With smooth_cutoff: the edge basis without it, the convolution weights and
    the edge density with it once, the moment-space weights without it.
    Otherwise: the edge basis with it and nothing after.
    """
    build, smooth = SETTINGS[setting]
    seen = {}

    def keep(key):
        def hook(module, args, output):  # pylint: disable=unused-argument
            seen.setdefault(key, []).append(output)

        return hook

    def keep_input(key, position):
        def hook(module, args):  # pylint: disable=unused-argument
            seen.setdefault(key, []).append(args[position])

        return hook

    with default_dtype(torch.float64):
        model = build(name)
        batch = _batch(*TRIMER)
        for inter in model.interactions:
            inter.register_forward_pre_hook(
                lambda module, args, kwargs: seen.setdefault("call", []).append(kwargs),
                with_kwargs=True,
            )
            inter.conv_tp_weights.register_forward_hook(keep("weights"))
            inter.conv_tp.register_forward_pre_hook(keep_input("weights_used", 2))
            inter.conv_tp_weights_magmom.register_forward_hook(keep("magmom_weights"))
            inter.magmom_conv_tp.register_forward_pre_hook(
                keep_input("magmom_weights_used", 2)
            )
            inter.density_fn.register_forward_hook(keep("density_fn"))
            inter.magmom_linear.register_forward_hook(keep("message"))
            after_density = (
                inter.magmom_skip_tp
                if hasattr(inter, "magmom_skip_tp")
                else inter.reshape
            )
            after_density.register_forward_pre_hook(keep_input("normalised", 0))
        model(dict(batch), training=False)

        _, lengths = get_edge_vectors_and_lengths(
            batch["positions"], batch["edge_index"], batch["shifts"]
        )
        envelope = model.radial_embedding.cutoff_fn(lengths)
        bare = model.radial_embedding.bessel_fn(lengths)
        receiver = batch["edge_index"][1]

    assert len(seen["call"]) == len(model.interactions) == 2
    assert torch.all(envelope < 0.99) and torch.all(envelope > 0.0)
    for layer, call in enumerate(seen["call"]):
        torch.testing.assert_close(
            call["edge_feats"], bare if smooth else bare * envelope, rtol=0, atol=0
        )
        if smooth:
            torch.testing.assert_close(call["cutoff"], envelope, rtol=0, atol=0)
        else:
            assert call["cutoff"] is None
        factor = envelope if smooth else 1.0
        torch.testing.assert_close(
            seen["weights_used"][layer], seen["weights"][layer] * factor
        )
        # Not even multiplied by one: the very tensor the MLP returned.
        assert seen["magmom_weights_used"][layer] is seen["magmom_weights"][layer]
        density = scatter_sum(
            torch.tanh(seen["density_fn"][layer] ** 2) * factor,
            index=receiver,
            dim=0,
            dim_size=batch["positions"].shape[0],
        )
        torch.testing.assert_close(
            seen["message"][layer], seen["normalised"][layer] * (density + 1)
        )


@pytest.mark.parametrize("name", SOC_BLOCKS)
@pytest.mark.parametrize("setting", SETTINGS)
def test_a_model_rebuilt_from_its_configuration_computes_the_same(name, setting):
    """extract_model rebuilds from the configuration and loads the state dict,
    which cannot restore a plain attribute: the flag has to come through the
    configuration, for an old checkpoint as much as for a new model."""
    build, smooth = SETTINGS[setting]
    with default_dtype(torch.float64):
        model = build(name)
        rebuilt = extract_model(model)
        assert rebuilt.smooth_cutoff is smooth
        assert rebuilt.radial_embedding.apply_cutoff is not smooth
        _assert_same_outputs(model, rebuilt, _batch(*TRIMER))


@pytest.mark.parametrize("setting", SETTINGS)
def test_a_selected_head_computes_what_it_did_in_the_multihead_model(setting):
    build, smooth = SETTINGS[setting]
    heads = ["pt_head", "dft"]
    with default_dtype(torch.float64):
        model = build(SOC_BLOCKS[1], heads=heads)
        for head in heads:
            single = remove_pt_head(model, head)
            assert single.smooth_cutoff is smooth
            assert single.radial_embedding.apply_cutoff is not smooth
            expected = _outputs(model, _batch(*TRIMER, head=head, heads=heads))
            actual = _outputs(single, _batch(*TRIMER))
            for key, value in expected.items():
                np.testing.assert_allclose(
                    actual[key], value, rtol=0.0, atol=1e-12, err_msg=key
                )


@pytest.mark.parametrize(
    "backend", [pytest.param("cueq", marks=pytest.mark.cueq), "oeq"]
)
@pytest.mark.parametrize("setting", SETTINGS)
def test_a_converted_model_keeps_its_cutoff_placement(backend, setting):
    """The converters rebuild from the configuration too."""
    # pylint: disable=import-outside-toplevel
    if backend == "cueq":
        from mace.cli.convert_e3nn_cueq import run
    else:
        from mace.cli.convert_e3nn_oeq import run
    build, smooth = SETTINGS[setting]
    with default_dtype(torch.float64):
        model = build(SOC_BLOCKS[1])
        converted = run(model, device="cpu", return_model=True)
    assert converted.smooth_cutoff is smooth
    assert converted.radial_embedding.apply_cutoff is not smooth


#: Compiled in a fresh interpreter: a compiled magnetic model loaded earlier in
#: the same process has already defined these classes in TorchScript's global
#: compilation unit, and scripting them again fails with "Can't redefine
#: method", whatever the model.
COMPILE_AND_COMPARE = """
import sys
import torch
from e3nn.util import jit
from tests.extensions.magnetic import test_magnetic_cutoff as t
with t.default_dtype(torch.float64):
    model = t.SETTINGS[sys.argv[1]][0](t.SOC_BLOCKS[0])
    compiled = jit.compile(model)
    t._assert_same_outputs(model, compiled, t._batch(*t.TRIMER), atol=1e-10)
"""


@pytest.mark.parametrize("setting", ["default", "saved_before_smooth_cutoff"])
def test_torchscript_compiles_and_agrees(setting):
    # The plain block only: the residual block's forward is annotated as
    # returning None beside the message and returns its skip connection, so
    # TorchScript rejects it, with or without this flag.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + sys.path)
    result = subprocess.run(
        [sys.executable, "-c", COMPILE_AND_COMPARE, setting],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
