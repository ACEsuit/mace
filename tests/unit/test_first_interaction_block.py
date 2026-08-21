"""Which first interaction block a `--model MACE` run actually gets.

`configure_model` keeps an allowlist of three blocks for the first layer, and
rewrites anything else to `RealAgnosticInteractionBlock` without saying so
(mace/tools/model_script_utils.py:282-289). So `--interaction_first` can name a
registered, valid block, be accepted by the parser, and train a different
architecture than the one asked for, with nothing in the log to show it.

The allowlist itself is worth pinning because it grew: the non-linear residual
block is a valid first layer now, and an assembly that re-imposed the old
two-entry restriction would be a silent regression rather than an error.

The coercion itself is deliberately not covered. The inventory carries it as
DROP -- a config value the tool overwrites without a word is worse than a
rejected one, and v1 fails the combination in config validation instead -- and a
test asserting today's silent rewrite would have to be deleted to make that
change. What is pinned here is only the part that survives: which blocks are
accepted.
"""

import logging

import numpy as np
import pytest

from mace import data
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools import torch_geometric, torch_tools

from tests.unit.test_e0s_characterization import (  # reuse the real parse-and-build path
    BASE_ARGV,
    Z_TABLE,
    load_training_structures,
)

ALLOWED = [
    "RealAgnosticInteractionBlock",
    "RealAgnosticDensityInteractionBlock",
    "RealAgnosticResidualNonLinearInteractionBlock",
]
def build(interaction_first):
    """Parse a real command line and return (model, args) as run_train would."""
    from mace.tools.arg_parser import build_default_arg_parser  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from mace.tools.model_script_utils import configure_model  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    structures = load_training_structures(limit=4)
    atomic_energies = np.array([-0.1, -0.2, -0.3])
    with torch_tools.default_dtype("float64"):
        graphs = [
            data.AtomicData.from_config(
                config_from_atoms(atoms, KeySpecification.from_defaults()),
                z_table=Z_TABLE,
                cutoff=4.0,
            )
            for atoms in structures
        ]
        loader = torch_geometric.dataloader.DataLoader(
            graphs, batch_size=2, shuffle=False
        )
        argv = list(BASE_ARGV)
        argv[argv.index("--model") + 1] = "MACE"
        argv += ["--interaction_first", interaction_first]
        args = build_default_arg_parser().parse_args(argv)
        args.compute_energy = True
        args.compute_dipole = False
        args.compute_polarizability = False
        args.compute_magforces = False
        built, _ = configure_model(
            args, loader, atomic_energies, heads=["Default"], z_table=Z_TABLE
        )
    return built, args


def first_block_name(built):
    return type(built.interactions[0]).__name__


# ---------------------------------------------------------------------------
# the allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALLOWED)
def test_an_allowlisted_first_block_is_the_one_built(name):
    """All three, so a shrunk allowlist fails here rather than in someone's run."""
    built, args = build(name)

    assert args.interaction_first == name
    assert first_block_name(built) == name


def test_the_allowlist_has_exactly_these_three_entries():
    """Stated as a set, because the interesting failure is an entry quietly
    leaving: the non-linear residual block was added to it recently and a
    rewrite that restored the older two would train a different architecture for
    anyone who had adopted it.
    """
    from mace.tools import model_script_utils  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    import ast  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    import inspect  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    tree = ast.parse(inspect.getsource(model_script_utils))
    lists = [
        {element.value for element in node.elts if isinstance(element, ast.Constant)}
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and any(
            isinstance(element, ast.Constant)
            and element.value == "RealAgnosticInteractionBlock"
            for element in node.elts
        )
    ]

    assert set(ALLOWED) in lists, lists


def test_the_density_block_is_reachable_as_a_first_layer():
    """`RealAgnosticDensityInteractionBlock` is the foundation-model first layer,
    and the only route to it from the CLI is this allowlist."""
    built, _ = build("RealAgnosticDensityInteractionBlock")

    assert first_block_name(built) == "RealAgnosticDensityInteractionBlock"
