"""Loading the committed anchors, and building the graphs they consume.

`harness.py` may not import the framework, so the two operations every
anchor-based test starts with -- read the checkpoint, turn an ase structure
into the batch the forward pass eats -- have to live somewhere else. They
live here rather than in one test file because five suites now need them
(`test_tiny_anchors`, the numerics characterization, the train-step gradient
golden, and the parity work that follows), and a second copy of `_batch` is
exactly how two suites end up silently measuring different things.

Two subtleties worth reading before copying this code.

**The graph is built inside a `default_dtype` scope, not cast afterwards.**
`AtomicData` reads the process-wide default dtype, which is float32 under
pytest. Building in float32 and casting up rounds the positions first, and the
anchor then reproduces its own reference only to about 2e-8 relative -- under
the fp64 row, so it reads as agreement, while making a bit-exact comparison
impossible.

**`KeySpecification.from_defaults()` is passed explicitly, because the default
argument is an empty one.** `config_from_atoms` defaults to a bare
`KeySpecification()` (mace/data/utils.py:174), which reads no property out of
the structure at all, and `AtomicData.from_config` then falls back to
`energy = 0.0` and `forces = zeros` (mace/data/atomic_data.py:308-321) while
the matching weights fall back to 1.0 -- so a loss taken on that batch is the
loss against zero labels, at full weight, with nothing to distinguish it from
the real one but its value. `from_defaults()` maps the `REF_energy` /
`REF_forces` keys `fixtures/tiny_train.xyz` actually carries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
from ase import Atoms

from mace import data
from mace.data.utils import KeySpecification
from mace.tools import torch_geometric, torch_tools, utils

from . import harness

#: anchor name -> where its pieces live and what class it must be.
ANCHORS: Dict[str, dict] = {
    "tiny_scaleshift": {
        "model": harness.MODELS_DIR / "tiny_scaleshift.model",
        "sidecar": harness.MODELS_DIR / "tiny_scaleshift.build.json",
        "reference": harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json",
        "class": "ScaleShiftMACE",
    },
    "tiny_mace": {
        "model": harness.MODELS_DIR / "tiny_mace.model",
        "sidecar": harness.MODELS_DIR / "tiny_mace.build.json",
        "reference": harness.REFERENCES_DIR / "tiny_mace_e3nn_cpu_fp64.json",
        "class": "MACE",
    },
}

#: the seeded synthetic training set the trainable anchor was fitted on. Not
#: an evaluation fixture -- it is the only committed structure set that
#: carries reference labels, so it is what a loss or a gradient is taken on.
TRAIN_SET = harness.FIXTURES_DIR / "tiny_train.xyz"


def anchor_path(name: str) -> Path:
    return ANCHORS[name]["model"]


def load_anchor(name: str, dtype: torch.dtype = torch.float64) -> torch.nn.Module:
    """Read a committed anchor checkpoint and cast it to ``dtype``."""
    model = torch.load(anchor_path(name), weights_only=False, map_location="cpu")
    return model.to(dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return {torch.float32: "float32", torch.float64: "float64"}[dtype]


def anchor_batch(
    model: torch.nn.Module,
    structures: Sequence[Atoms],
    dtype: torch.dtype = torch.float64,
):
    """Collate ``structures`` into the Batch ``model`` consumes.

    The Batch object itself, not its dict: the loss functions read `ref.ptr`
    and the per-config weights off the batch, and only the forward pass wants
    the dict.
    """
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    with torch_tools.default_dtype(_dtype_name(dtype)):
        graphs = [
            data.AtomicData.from_config(
                data.config_from_atoms(atoms, KeySpecification.from_defaults()),
                z_table=z_table,
                cutoff=float(model.r_max),
            )
            for atoms in structures
        ]
        loader = torch_geometric.dataloader.DataLoader(
            graphs, batch_size=len(graphs), shuffle=False
        )
        batch = next(iter(loader))
    return batch


def anchor_graph(
    model: torch.nn.Module,
    atoms: Atoms,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:
    """One structure as the input dict of ``model.forward``.

    The trailing cast is belt and braces for any tensor the `default_dtype`
    scope does not reach; the scope is what makes the numbers right.
    """
    graph = anchor_batch(model, [atoms], dtype).to_dict()
    return {
        key: (
            value.to(dtype)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value
        )
        for key, value in graph.items()
    }


def load_training_structures(
    *, isolated_atoms: bool = False, limit: int | None = None
) -> list:
    """Read `fixtures/tiny_train.xyz`, by default without the isolated atoms.

    The three single-atom configurations carry no neighbours and no forces
    worth fitting; they are there so the E0 extraction has something to find,
    which is a different test's business.
    """
    from ase.io import read  # noqa: PLC0415  (ase.io pulls in a lot)

    structures: Iterable[Atoms] = read(TRAIN_SET, ":")
    kept = [
        atoms
        for atoms in structures
        if isolated_atoms or atoms.info.get("config_type") != "IsolatedAtom"
    ]
    return kept if limit is None else kept[:limit]
