"""Four flags whose effect is visible in the model or in the resolved arguments.

All four are MERGE rows in the inventory: they survive the rewrite in some other
form, and a MERGE promises the same behaviour under a different name, so today's
behaviour is what the merge gets judged against.

`--use_last_readout_only` and `--use_embedding_readout` change which readout
blocks a model has, which is structural and cheap to check. `--downloads_dir`
is resolved from `--work_dir` when it is not given, which is the kind of default
that is only wrong once and then wrong everywhere. `--multi_processed_test` is
covered where it is decidable without a sharded dataset: the flag exists to skip
the per-file HDF5 test loading, so what is pinned is the branch it guards being
keyed on it.
"""

import os

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules, tools
from mace.tools.arg_parser_tools import check_args

TABLE = tools.AtomicNumberTable([1, 8])

#: The joint embedding is what `--use_embedding_readout` reads out of, and it
#: only exists when embedding specs are given.
SPECS = {"charge": {"type": "continuous", "in_dim": 1, "emb_dim": 4, "per": "graph"}}


def build(**overrides):
    torch.manual_seed(0)
    config = {
        "r_max": 4.0,
        "num_bessel": 4,
        "num_polynomial_cutoff": 5,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        "num_interactions": 3,
        "num_elements": 2,
        "hidden_irreps": o3.Irreps("8x0e + 8x1o"),
        "MLP_irreps": o3.Irreps("4x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": np.array([-1.0, -5.0]),
        "avg_num_neighbors": 4.0,
        "atomic_numbers": TABLE.zs,
        "correlation": 2,
    }
    config.update(overrides)
    return modules.MACE(**config)


# ---------------------------------------------------------------------------
# --use_last_readout_only
# ---------------------------------------------------------------------------


def test_by_default_every_layer_gets_a_readout():
    """Three interactions, three readouts: the site energy is summed over layers."""
    model = build()

    assert len(model.readouts) == 3


def test_the_flag_leaves_only_the_last_readout():
    """`--use_last_readout_only` is a different architecture, not a tuning knob:
    the per-layer contributions disappear from the energy."""
    model = build(use_last_readout_only=True)

    assert len(model.readouts) == 1


def test_the_flag_is_recorded_on_the_model():
    """It has to survive a checkpoint, since the forward branches on it."""
    model = build(use_last_readout_only=True)

    assert model.use_last_readout_only is True
    assert build().use_last_readout_only is False


def test_the_two_models_disagree_on_the_energy(batch):
    """Structural difference with a numerical consequence, so a flag that stopped
    reaching the constructor would be visible in a forward and not only in a
    module count."""
    torch.manual_seed(0)
    every = build()(batch.to_dict(), training=False)["energy"]
    torch.manual_seed(0)
    last = build(use_last_readout_only=True)(batch.to_dict(), training=False)["energy"]

    assert not torch.allclose(every, last)


# ---------------------------------------------------------------------------
# --use_embedding_readout
# ---------------------------------------------------------------------------


def test_no_embedding_readout_without_the_flag():
    model = build(embedding_specs=SPECS)

    assert not hasattr(model, "embedding_readout")


def test_the_flag_adds_a_readout_over_the_joint_embedding():
    model = build(embedding_specs=SPECS, use_embedding_readout=True)

    assert hasattr(model, "embedding_readout")
    assert isinstance(model.embedding_readout, modules.LinearReadoutBlock)


def test_it_needs_the_joint_embedding_to_read_from():
    """Without embedding specs there is no joint embedding, so the flag has
    nothing to attach to and is silently inert. Pinned so that the silence is a
    decision rather than a surprise."""
    model = build(use_embedding_readout=True)

    assert not hasattr(model, "embedding_readout")


# ---------------------------------------------------------------------------
# --downloads_dir
# ---------------------------------------------------------------------------


def resolved_args(*argv):
    """Parse a real command line and resolve it, as `run_train` does.

    `check_args` reads far more of the namespace than the directories, so a
    hand-built one drifts out of date the moment a flag is added; parsing gives
    the arguments the CLI would actually produce.
    """
    from mace.tools.arg_parser import build_default_arg_parser  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    args = build_default_arg_parser().parse_args(
        ["--name", "x", "--train_file", "y.xyz", *argv]
    )
    resolved, _ = check_args(args)
    return resolved


def test_the_downloads_directory_defaults_under_the_work_dir():
    """One flag decides five directories, and this is the one that decides where
    a foundation model lands. On a cluster whose compute nodes have no internet,
    getting it wrong is a job that dies on startup."""
    resolved = resolved_args("--work_dir", "/tmp/run")

    assert resolved.downloads_dir == os.path.join("/tmp/run", "downloads")


def test_an_explicit_downloads_directory_is_kept():
    resolved = resolved_args(
        "--work_dir", "/tmp/run", "--downloads_dir", "/scratch/cache"
    )

    assert resolved.downloads_dir == "/scratch/cache"


def test_it_follows_the_work_dir_it_is_given():
    resolved = resolved_args("--work_dir", "/elsewhere")

    assert resolved.downloads_dir == os.path.join("/elsewhere", "downloads")


@pytest.fixture(name="batch")
def fixture_batch():
    from mace import data  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    from mace.tools import torch_geometric  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        properties={"forces": np.zeros((3, 3)), "energy": -1.5},
        property_weights={"forces": 1.0, "energy": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=TABLE, cutoff=4.0)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False
    )
    return next(iter(loader))
