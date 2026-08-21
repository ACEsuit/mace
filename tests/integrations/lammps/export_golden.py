"""The committed LAMMPS-export golden: how it is built, and how it is replayed.

The plain-TorchScript export format is itself slated for replacement, but the
*numbers* the exported artefact produces are what the replacement is measured
against, so they are frozen here.

The rule this module exists to satisfy
======================================

Contract tests may not reach into ``mace/`` for their assertions -- that is
what lets the whole suite be re-run against a different engine. The LAMMPS
harness in this directory (``_harness.py``) breaks that rule by construction:
building the domain-decomposed input LAMMPS hands to the pair style needs a
neighbour list, and the only neighbour list on hand is the package's own.

The two ways out are to accept the import or to reimplement the neighbour
list, and both are bad -- the first makes the golden depend on the code it
measures, the second commits a second neighbour list that can drift from the
first without anybody noticing.

So neither is taken. **The input is committed alongside the outputs.** The
``_harness.py`` import lives in :func:`build_input`, which runs only when the
golden is regenerated; the test path calls :func:`replay`, which takes the
committed arrays and the exported artefact and imports nothing but ``torch``.
A rewrite therefore does not need this repository's neighbour list to
reproduce the golden -- it needs the file. And if a change to the neighbour
list ever moved the input, the committed copy would no longer match what the
harness builds, which :func:`build_input` is re-run to discover rather than
something a passing test would hide.

What is *not* covered, and why
==============================

The ML-IAP format is exported and its declared interface is snapshotted, but
its numerics are not. Driving it needs LAMMPS to exchange ghost node features
between interaction layers (``forward_exchange``), which exists only in the
KOKKOS coupling; the committed anchors have two layers, so a faithful
stand-in would have to reimplement an exchange protocol this repository has
no reference implementation of, and a golden built on a guessed protocol is
worse than none. The refusal that behaviour produces *is* pinned, because it
is load-bearing: without it the failure was a bare ``AttributeError`` from
inside the second layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

#: The committed fixture the cluster is built from, and how many replicas per
#: axis. ``triclinic_bulk`` rather than an orthorhombic cell on purpose: a
#: cell with a right angle and a planar molecule makes whole force components
#: vanish by symmetry, and a golden whose numbers are structurally zero
#: cannot tell a sign error from a correct answer.
#:
#: Four replicas per axis, and the number is a measurement rather than a
#: round one. The receptive field is two layers of 3.5 Angstrom against a
#: roughly 4 Angstrom cell, so the local block's environment is only complete
#: in the limit; the residual against the periodic reference is 9.4e-8 eV at
#: three replicas, 2.3e-8 at four and exactly zero (4e-16) at five, and the
#: forces are 1.2e-6, 1.8e-7 and 2e-15 eV/Ang. Three does not fit inside the
#: 1e-6 row, five costs 128 kB of committed edge list, so four it is -- with
#: about five times the headroom the bound needs.
FIXTURE = "triclinic_bulk"
N_REPEAT = 4

#: Everything the exported wrapper's forward path reads. Recorded and
#: replayed verbatim: a key the model starts reading that is not here makes
#: the replay fail loudly on a missing key, which is the outcome to want --
#: the alternative is a quietly different input.
INPUT_KEYS = (
    "positions",
    "node_attrs",
    "edge_index",
    "shifts",
    "unit_shifts",
    "cell",
    "batch",
    "ptr",
)

#: Recorded as a shape rather than as numbers. The cluster is *open* -- the
#: periodicity is materialised as ghost atoms, which is the whole point of
#: the LAMMPS layout -- so every shift is exactly zero and writing several
#: thousand zeros into a committed file buys nothing. :func:`build_input`
#: asserts they really are zero rather than assuming it.
ZERO_KEYS = ("shifts", "unit_shifts")

INTEGRAL_KEYS = ("edge_index", "batch", "ptr")


def build_input(model, atoms) -> Dict[str, object]:
    """Build the LAMMPS-style input for ``model`` on ``atoms``. Regeneration only.

    This is the one function in the golden that imports the package, and no
    test calls it.
    """
    import torch  # noqa: PLC0415
    from ase.atoms import Atoms  # noqa: PLC0415

    from tests.integrations.lammps._harness import model_batch  # noqa: PLC0415

    n_atoms = len(atoms)
    supercell = atoms.repeat((N_REPEAT,) * 3)
    # ase orders the replicas outer and the atoms inner, so an atom's index
    # modulo the cell size is the unit-cell atom it is an image of.
    image_of = np.arange(len(supercell)) % n_atoms
    centre = (
        (N_REPEAT // 2) * N_REPEAT * N_REPEAT
        + (N_REPEAT // 2) * N_REPEAT
        + (N_REPEAT // 2)
    )
    local_index = np.arange(centre * n_atoms, (centre + 1) * n_atoms)

    cluster = Atoms(
        numbers=supercell.numbers, positions=supercell.positions, pbc=False
    )
    batch = model_batch(model, cluster)
    local_or_ghost = np.zeros(len(cluster))
    local_or_ghost[local_index] = 1.0

    recorded: Dict[str, object] = {}
    for key in INPUT_KEYS:
        array = np.asarray(batch[key].detach().cpu())
        if key in ZERO_KEYS:
            assert not array.any(), (
                f"{key} is not all zero on an open cluster, so the compact "
                f"recording below would drop real information"
            )
            recorded[key] = {"zeros": list(array.shape)}
        else:
            recorded[key] = array.tolist()
    recorded["local_or_ghost"] = local_or_ghost.tolist()
    recorded["image_of"] = image_of.tolist()
    recorded["local_index"] = local_index.tolist()
    recorded["fixture"] = atoms.info.get("golden_name", FIXTURE)
    recorded["n_repeat"] = N_REPEAT
    assert torch is not None  # imported for the dtype the batch was built in
    return recorded


def _tensors(recorded: Dict[str, object], torch):
    """The committed input as the tensors the artefact consumes."""
    data = {}
    for key in INPUT_KEYS:
        values = recorded[key]
        if isinstance(values, dict):
            data[key] = torch.zeros(tuple(values["zeros"]), dtype=torch.float64)
            continue
        data[key] = torch.tensor(
            values,
            dtype=torch.int64 if key in INTEGRAL_KEYS else torch.float64,
        )
    return data


def replay(artifact_path: Path, recorded: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Run an exported libtorch artefact on the committed input.

    Imports ``torch`` and nothing else: this is the assertion path, and it
    must work against any artefact that speaks the same interface, whichever
    stack produced it.
    """
    import torch  # noqa: PLC0415

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        loaded = torch.jit.load(str(artifact_path), map_location="cpu")
        data = _tensors(recorded, torch)
        local_or_ghost = torch.tensor(
            recorded["local_or_ghost"], dtype=torch.float64
        )
        out = loaded(data, local_or_ghost)
    finally:
        torch.set_default_dtype(previous)

    image_of = np.asarray(recorded["image_of"], dtype=np.int64)
    n_local = int(image_of.max()) + 1
    forces = np.asarray(out["forces"].detach())
    # LAMMPS's reverse communication: a ghost image's force belongs to the
    # local atom it is an image of. Folding here is what makes the recorded
    # force block one row per *unit-cell* atom rather than one per cluster
    # atom, so a change in how many replicas the harness builds does not
    # change the golden's shape.
    folded = np.zeros((n_local, 3))
    np.add.at(folded, image_of, forces)
    return {
        "total_energy_local": np.asarray(
            out["total_energy_local"].detach()
        ).reshape(()),
        "folded_forces": folded,
        "node_energy": np.asarray(out["node_energy"].detach()),
    }


def mliap_interface(artifact_path: Path) -> Dict[str, object]:
    """The declared ML-IAP interface of an exported artefact.

    Not TorchScript -- the ML-IAP export is a pickled module -- so it is read
    with ``torch.load`` and inspected through the attributes the ML-IAP
    runtime itself reads: LAMMPS asks for the element list, the cutoff and
    the descriptor/parameter counts before it ever calls the model, and a
    change in any of them changes what a LAMMPS input script has to say.
    """
    import torch  # noqa: PLC0415

    loaded = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    return {
        "element_types": list(loaded.element_types),
        "num_species": int(loaded.num_species),
        "rcutfac": float(loaded.rcutfac),
        "ndescriptors": int(loaded.ndescriptors),
        "nparams": int(loaded.nparams),
        "dtype": str(loaded.dtype),
        "num_interactions": int(loaded.model.num_interactions),
        "lammps_mliap_flag": bool(getattr(loaded.model.model, "lammps_mliap", False)),
    }
