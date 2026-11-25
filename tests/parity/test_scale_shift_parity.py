"""Each legacy command line's scale and shift, as a `ScaleShiftSpec`.

The legacy training builds its model through `configure_model`, which reads
`--model` and `--scaling` and computes the statistics from the training set.
The spec that stands for that command line is given the same statistics,
computed by the legacy functions on the same loader, and must give back the
scale and the shift the built model holds.
"""

import numpy as np
import pytest
import torch
from mace_core.config import ScaleShiftSpec

from mace import data, modules
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric, torch_tools, utils
from tests.golden.anchors import load_training_structures

Z_TABLE = utils.AtomicNumberTable([1, 6, 8])
ATOMIC_ENERGIES = np.array([-0.1, -0.2, -0.3])

ARGV = [
    "--name",
    "scale_shift_parity",
    "--train_file",
    "unused.xyz",
    "--r_max",
    "4.0",
    "--hidden_irreps",
    "4x0e",
    "--num_interactions",
    "2",
    "--default_dtype",
    "float64",
]


@pytest.fixture(name="loader", scope="module")
def fixture_loader():
    with torch_tools.default_dtype("float64"):
        graphs = [
            data.AtomicData.from_config(
                config_from_atoms(atoms, KeySpecification.from_defaults()),
                z_table=Z_TABLE,
                cutoff=4.0,
            )
            for atoms in load_training_structures(limit=4)
        ]
    return torch_geometric.dataloader.DataLoader(graphs, batch_size=2, shuffle=False)


def _configure(loader, model, scaling):
    from mace.tools.arg_parser import build_default_arg_parser
    from mace.tools.model_script_utils import configure_model

    with torch_tools.default_dtype("float64"):
        args = build_default_arg_parser().parse_args(
            [*ARGV, "--model", model, "--scaling", scaling]
        )
        args.compute_energy = True
        args.compute_dipole = False
        args.compute_polarizability = False
        args.compute_magforces = False
        built, _ = configure_model(
            args, loader, ATOMIC_ENERGIES, heads=["Default"], z_table=Z_TABLE
        )
    return built


@pytest.mark.parametrize(
    ("model", "scaling", "spec", "statistic"),
    [
        ("MACE", "rms_forces_scaling", ScaleShiftSpec(), "rms_forces"),
        ("MACE", "std_scaling", ScaleShiftSpec(scale="std"), "energy_std"),
        (
            "ScaleShiftMACE",
            "rms_forces_scaling",
            ScaleShiftSpec(scale="rms_forces", shift="mean"),
            "rms_forces",
        ),
        (
            "ScaleShiftMACE",
            "std_scaling",
            ScaleShiftSpec(scale="std", shift="mean"),
            "energy_std",
        ),
    ],
)
def test_the_spec_gives_the_scale_and_shift_the_legacy_model_holds(
    fp64, loader, model, scaling, spec, statistic
):
    built = _configure(loader, model, scaling)
    assert type(built).__name__ == "ScaleShiftMACE"
    mean, spread = modules.scaling_classes[scaling](loader, ATOMIC_ENERGIES)
    scale, shift = spec.scale_and_shift(
        energy_mean=float(np.asarray(mean).item()),
        **{statistic: float(np.asarray(spread).item())},
    )
    assert torch.atleast_1d(built.scale_shift.scale).tolist() == [scale]
    assert torch.atleast_1d(built.scale_shift.shift).tolist() == [shift]
