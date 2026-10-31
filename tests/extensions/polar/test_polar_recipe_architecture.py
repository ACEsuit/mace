"""`mace_run_train` must be able to build the released MACE-Polar architecture.

The released `polar-1-m` / `polar-1-l` checkpoints end in a `NonLinearBiasReadoutBlock`
and were built with `apply_cutoff=False`. `--apply_cutoff` was already a CLI flag, but
the readout class was reachable only by constructing `PolarMACE` in Python, so the
published training recipe built a different architecture from the one it ships with:
a `NonLinearReadoutBlock` last readout (no `linear_mid`, no bias on `linear_2`).

`--readout_cls` closes that gap. These tests pin it shut, at both released sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mace import tools
from mace.calculators.foundations_models import mace_polar
from mace.tools.model_script_utils import configure_model
from mace.tools.scripts_utils import extract_config_mace_model

pytestmark = pytest.mark.polar

# The published polar-1 recipe: everything shared between the two sizes.
POLAR_1_SHARED = [
    "--name=polar-1",
    "--model=PolarMACE",
    "--interaction=RealAgnosticResidualNonLinearInteractionBlock",
    "--interaction_first=RealAgnosticResidualNonLinearInteractionBlock",
    "--r_max=6.0",
    "--num_radial_basis=8",
    "--num_cutoff_basis=5",
    "--max_ell=3",
    "--correlation=3",
    "--radial_type=bessel",
    "--radial_MLP=[64, 64, 64]",
    "--MLP_irreps=16x0e",
    "--gate=silu",
    "--use_agnostic_product=True",
    "--avg_num_neighbors=30.0",
    "--num_recursion_steps=2",
    "--kspace_cutoff_factor=1.5",
    "--atomic_multipoles_max_l=1",
    "--atomic_multipoles_smearing_width=1.5",
    "--field_feature_max_l=1",
    "--field_feature_widths=[1.5, 3.0]",
    "--field_feature_norms=[20.0, 20.0, 0.5, 0.5]",
    "--field_norm_factor=1.0",
    "--field_si=False",
    "--include_electrostatic_self_interaction=True",
    "--add_local_electron_energy=True",
    "--quadrupole_feature_corrections=False",
    "--fixedpoint_update_config={'type': 'AgnosticEmbeddedOneBodyVariableUpdate',"
    " 'potential_embedding_cls': 'AgnosticChargeBiasedLinearPotentialEmbedding',"
    " 'nonlinearity_cls': 'MLPNonLinearity'}",
    "--field_readout_config={'type': 'OneBodyMLPFieldReadout'}",
]

# num_interactions / hidden_irreps / edge_irreps, per released size.
POLAR_1_SIZES = {
    "polar-1-m": (2, "512x0e + 512x1o", "128x0e + 128x1o"),
    "polar-1-l": (3, "512x0e + 512x1o + 512x2e", "128x0e + 128x1o + 128x2e"),
}

# The two settings the release used and the recipe did not state.
RELEASED_ARCH = ["--readout_cls=NonLinearBiasReadoutBlock", "--apply_cutoff=False"]


def _size_flags(size: str, narrow: bool = False) -> list[str]:
    """Recipe flags for a released size, optionally narrowed to keep a test cheap.

    Narrowing the channel count changes no architectural decision: the layer count and
    the irrep sequence -- the two things that differ between the released sizes -- stay
    those of the release.
    """
    num_interactions, hidden, edge = POLAR_1_SIZES[size]
    if narrow:
        hidden, edge = hidden.replace("512x", "32x"), edge.replace("128x", "16x")
    return [
        f"--num_interactions={num_interactions}",
        f"--hidden_irreps={hidden}",
        f"--edge_irreps={edge}",
    ]


def _build(argv: list[str], z_table: tools.AtomicNumberTable) -> torch.nn.Module:
    """Build a model exactly as `mace_run_train` would, minus the data pipeline."""
    args = tools.build_default_arg_parser().parse_args(argv)
    # run_train derives these from the model/loss rather than the parser, and takes
    # the scale/shift from the head statistics; no data loader is involved here.
    args.compute_energy, args.compute_dipole = True, False
    args.mean, args.std = [0.0], [1.0]
    model, _ = configure_model(
        args, None, np.zeros(len(z_table.zs)), heads=["Default"], z_table=z_table
    )
    return model


@pytest.mark.parametrize("size", list(POLAR_1_SIZES))
@pytest.mark.parametrize(
    "readout", ["NonLinearReadoutBlock", "NonLinearBiasReadoutBlock"]
)
@pytest.mark.parametrize("apply_cutoff", [True, False])
def test_cli_builds_the_requested_readout_and_cutoff(size, readout, apply_cutoff):
    """Both settings reach the model, independently of each other."""
    z_table = tools.AtomicNumberTable([1, 6, 7, 8])
    model = _build(
        POLAR_1_SHARED
        + _size_flags(size, narrow=True)
        + [f"--readout_cls={readout}", f"--apply_cutoff={apply_cutoff}"],
        z_table,
    )
    num_interactions, _, _ = POLAR_1_SIZES[size]

    assert [type(r).__name__ for r in model.readouts] == ["LinearReadoutBlock"] * (
        num_interactions - 1
    ) + [readout]
    assert model.apply_cutoff is apply_cutoff
    assert model.radial_embedding.apply_cutoff is apply_cutoff

    # The bias variant is the one with an extra hidden layer; the plain one has no
    # trace of it, which is what makes the two distinguishable in a state dict.
    # (Scoped to the energy readouts: PolarMACE's field blocks carry their own.)
    prefix = f"readouts.{num_interactions - 1}."
    linear_mid = [
        k for k in model.state_dict() if k.startswith(prefix) and "linear_mid" in k
    ]
    assert bool(linear_mid) is (readout == "NonLinearBiasReadoutBlock")


@pytest.mark.parametrize("size", list(POLAR_1_SIZES))
def test_extract_config_round_trips_both_settings(size):
    """A fine-tune of a recipe-built model inherits what the recipe asked for."""
    z_table = tools.AtomicNumberTable([1, 6, 7, 8])
    model = _build(
        POLAR_1_SHARED + _size_flags(size, narrow=True) + RELEASED_ARCH,
        z_table,
    )
    config = extract_config_mace_model(model)
    assert config["readout_cls"].__name__ == "NonLinearBiasReadoutBlock"
    assert config["apply_cutoff"] is False


@pytest.mark.network
@pytest.mark.parametrize("size", list(POLAR_1_SIZES))
def test_released_checkpoint_loads_into_the_recipe_model(size):
    """Every released weight fits a model the CLI builds -- and only with the flags."""
    released = mace_polar(model=size, device="cpu", return_raw_model=True)
    z_table = tools.AtomicNumberTable([int(z) for z in released.atomic_numbers])
    reference = released.state_dict()

    built = _build(POLAR_1_SHARED + _size_flags(size) + RELEASED_ARCH, z_table)
    assert built.apply_cutoff == released.apply_cutoff
    assert [type(r).__name__ for r in built.readouts] == [
        type(r).__name__ for r in released.readouts
    ]

    shapes = {k: tuple(v.shape) for k, v in built.state_dict().items()}
    assert not [k for k in shapes if k not in reference]
    assert not [k for k in shapes if shapes[k] != tuple(reference[k].shape)]

    missing, unexpected = built.load_state_dict(reference, strict=False)
    assert not missing
    # The release carries e3nn Gate bookkeeping this e3nn does not register: an empty
    # weight and a mask derived from the irreps. Neither holds a trained number.
    assert all(k.endswith((".mul.weight", ".mul.output_mask")) for k in unexpected)
    assert all(torch.equal(v, reference[k]) for k, v in built.state_dict().items())

    # Without the two flags -- the recipe as published -- the last readout does not fit,
    # and the cutoff silently reverts to the setting the release was not built with.
    as_published = _build(POLAR_1_SHARED + _size_flags(size), z_table)
    last = f"readouts.{len(released.readouts) - 1}."
    assert {
        k: tuple(v.shape)
        for k, v in as_published.state_dict().items()
        if k.startswith(last)
    } != {k: tuple(v.shape) for k, v in reference.items() if k.startswith(last)}
    assert as_published.apply_cutoff is True
