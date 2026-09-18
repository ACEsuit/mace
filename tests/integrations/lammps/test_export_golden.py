"""The LAMMPS export produces the committed numbers. Contract tier, no binary.

``mace_create_lammps_model`` turns a checkpoint into the artefact a LAMMPS
pair style loads. The plain TorchScript format is on its way out, but what
replaces it has to produce the same physics, so the numbers are frozen here
and the replacement is measured against this file rather than against a
re-run of the code it replaces.

Everything in the assertion path reads a committed file or an exported
artefact -- the input the artefact is fed is committed alongside the outputs
(see ``export_golden.py`` for why, and for what that buys). The one import of
the package is in the regeneration path, which no test enters.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.golden import harness
from tests.helpers import REPO_ROOT, run_mace_train
from tests.integrations.lammps.export_golden import (
    FIXTURE,
    mliap_interface,
    replay,
)

CREATE_LAMMPS_MODEL = REPO_ROOT / "mace" / "cli" / "create_lammps_model.py"
ANCHOR = harness.MODELS_DIR / "tiny_scaleshift.model"
LIBTORCH_GOLDEN = harness.REFERENCES_DIR / "lammps_export_libtorch_fp64.json"
MLIAP_GOLDEN = harness.REFERENCES_DIR / "lammps_export_mliap_interface.json"
ANCHOR_REFERENCE = harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"

TOL = harness.FP64_CPU_REFERENCE


@pytest.fixture(name="anchor_copy")
def fixture_anchor_copy(tmp_path):
    """The CLI writes next to its input, so it is never given the committed one."""
    dest = tmp_path / "tiny_scaleshift.model"
    shutil.copy(ANCHOR, dest)
    return dest


@pytest.fixture(name="golden", scope="module")
def fixture_golden():
    assert LIBTORCH_GOLDEN.exists(), (
        f"{LIBTORCH_GOLDEN} is missing; regenerate it with "
        f"`python tests/golden/regenerate.py --target lammps "
        f"--i-know-what-i-am-doing`"
    )
    return json.loads(LIBTORCH_GOLDEN.read_text(encoding="utf-8"))


def export_libtorch(model_copy, **flags):
    run_mace_train(flags, extra_argv=[str(model_copy)], script=CREATE_LAMMPS_MODEL)
    artifact = model_copy.parent / (model_copy.name + "-lammps.pt")
    assert artifact.exists(), "libtorch export produced no artefact"
    return artifact


# ---------------------------------------------------------------------------
# The libtorch format
# ---------------------------------------------------------------------------


def test_the_exported_artifact_reproduces_the_committed_numbers(anchor_copy, golden):
    """The golden proper: same input in, same energy and forces out.

    The input is the committed one, so this compares the artefact against a
    frozen measurement and not against a second run of the same code. A
    rewrite's export passes this by reading the file.
    """
    artifact = export_libtorch(anchor_copy)
    produced = replay(artifact, golden["input"])
    expected = golden["outputs"]

    problems = []
    for channel in ("total_energy_local", "folded_forces", "node_energy"):
        got = np.asarray(produced[channel], dtype=float)
        want = np.asarray(expected[channel], dtype=float)
        if got.shape != want.shape:
            problems.append(f"{channel}: shape {want.shape} -> {got.shape}")
            continue
        diff = np.abs(got - want)
        if diff.max() > TOL.atol:
            index = np.unravel_index(int(np.argmax(diff)), diff.shape)
            problems.append(
                f"{channel}: worst at {tuple(int(i) for i in index)}, got "
                f"{float(got[index]):.12g}, golden {float(want[index]):.12g}, "
                f"|diff| {float(diff.max()):.3g} > {TOL.atol:g}"
            )
    assert not problems, (
        "the LAMMPS export no longer produces the committed numbers:\n  "
        + "\n  ".join(problems)
        + f"\n\nTolerance row '{TOL.name}': {TOL.rationale}"
    )


def test_the_exported_artifact_reproduces_the_periodic_calculation(
    anchor_copy, golden
):
    """The exported artefact is doing the physics, not just reproducing itself.

    LAMMPS evaluates an open cluster of local plus ghost atoms and keeps only
    the local atoms' site energies; the sum has to be the energy of the
    periodic cell, and the ghost-image forces folded back onto their owners
    have to be its forces. Compared against the committed anchor reference --
    the same file the calculator and the model surfaces are asserted against
    -- so the export cannot drift away from the rest of the stack while
    staying self-consistent.

    The agreement is limited by the *cluster*, not by arithmetic: three
    replicas of a 4 Angstrom cell do not quite cover the 7 Angstrom receptive
    field, and the measured residual is 9.3e-8 eV on the energy and 5.5e-7
    eV/Ang on the worst force component, against the 1e-6 bound. If this ever
    fails by a small multiple, check ``N_REPEAT`` before suspecting the model.
    Exact ghost parity, at five replicas, is ``test_ghost_parity.py``.
    """
    artifact = export_libtorch(anchor_copy)
    produced = replay(artifact, golden["input"])
    reference = harness.load_reference(ANCHOR_REFERENCE)["fixtures"][FIXTURE]

    periodic_energy = float(np.asarray(reference["outputs"]["energy"]["value"]))
    assert float(produced["total_energy_local"]) == pytest.approx(
        periodic_energy, abs=TOL.atol
    ), (
        "the local-atom energy of the exported artefact does not reproduce "
        "the periodic energy of the same structure"
    )

    periodic_forces = np.asarray(
        reference["outputs"]["forces"]["value"], dtype=float
    )
    folded = np.asarray(produced["folded_forces"], dtype=float)
    assert folded.shape == periodic_forces.shape
    assert np.abs(folded - periodic_forces).max() <= TOL.atol, (
        f"the folded ghost forces differ from the periodic forces by "
        f"{np.abs(folded - periodic_forces).max():.3g}"
    )


def test_the_golden_input_is_the_one_the_artifact_is_fed(golden):
    """The committed input is self-describing, and the description is checked.

    A recorded input nobody validates is a recorded input that can quietly
    stop matching the arrays beside it -- so the internal consistency of the
    block is asserted here rather than assumed by the two tests above.
    """
    recorded = golden["input"]
    n_atoms = len(recorded["positions"])
    n_edges = len(recorded["edge_index"][0])

    assert len(recorded["node_attrs"]) == n_atoms
    assert len(recorded["local_or_ghost"]) == n_atoms
    assert len(recorded["image_of"]) == n_atoms
    assert len(recorded["batch"]) == n_atoms
    assert recorded["ptr"] == [0, n_atoms]
    assert len(recorded["edge_index"]) == 2
    assert len(recorded["edge_index"][1]) == n_edges
    for key in ("shifts", "unit_shifts"):
        assert recorded[key] == {"zeros": [n_edges, 3]}, (
            f"{key} is recorded as something other than a zero block; the "
            f"cluster is open, so a non-zero shift means the recording no "
            f"longer describes what LAMMPS hands the model"
        )

    local = np.asarray(recorded["local_or_ghost"])
    assert set(np.unique(local)) == {0.0, 1.0}
    assert local.sum() == len(recorded["local_index"]), (
        "the local mask and the local index list disagree about how many "
        "atoms this domain owns"
    )
    assert int(np.asarray(recorded["image_of"]).max()) + 1 == int(local.sum())

    edges = np.asarray(recorded["edge_index"])
    assert edges.min() >= 0 and edges.max() < n_atoms, (
        "an edge points outside the cluster"
    )
    assert golden["provenance"]["fixture"] == FIXTURE


def test_the_package_import_stays_inside_the_regeneration_path():
    """The rule ``export_golden.py`` exists to keep, checked rather than
    asserted in prose.

    The module needs the package's neighbour list to *build* the golden and
    must not need it to *replay* one -- that is the entire reason the input
    is committed. So the one import is confined to ``build_input``, and this
    fails if it ever migrates to module scope or into the replay path, which
    is how the property would otherwise be lost: silently, in a commit that
    only meant to tidy the imports.
    """
    import ast  # noqa: PLC0415

    from tests.integrations.lammps import export_golden  # noqa: PLC0415

    source = Path(export_golden.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    regeneration_only = {"build_input"}

    def names(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                yield from ((a.name, child.lineno) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module:
                yield (child.module, child.lineno)

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    inside = {
        line
        for name in regeneration_only
        for _module, line in names(functions[name])
    }

    stray = [
        f"line {line}: {module}"
        for module, line in names(tree)
        if (module.startswith("mace") or module.startswith("tests.integrations"))
        and line not in inside
    ]
    assert not stray, (
        "export_golden.py reaches into the package outside build_input:\n  "
        + "\n  ".join(stray)
        + "\nThe replay path has to work against an artefact produced by any "
        "stack, so it may import nothing but torch."
    )


def test_the_float32_export_is_single_precision(anchor_copy):
    """`--dtype float32` is a deployment knob and its only visible effect is
    the precision of the artefact's own parameters."""
    import torch  # noqa: PLC0415

    artifact = export_libtorch(anchor_copy, dtype="float32")
    loaded = torch.jit.load(artifact, map_location="cpu")
    assert next(loaded.parameters()).dtype == torch.float32


# ---------------------------------------------------------------------------
# The ML-IAP format
# ---------------------------------------------------------------------------


@pytest.mark.cueq
def test_the_mliap_export_declares_the_committed_interface(anchor_copy):
    """The ML-IAP artefact's interface is snapshotted; its numerics are not.

    What LAMMPS reads off this object before it ever calls the model -- the
    element list, the cutoff, the descriptor and parameter counts -- is what
    a pair_style line has to agree with, so a change in any of it breaks
    every input script in the field. That is committed.

    The numbers are not, and the reason is recorded in ``export_golden.py``:
    past the first interaction layer the ML-IAP path needs LAMMPS to exchange
    ghost node features, which only the KOKKOS coupling can do, so a
    stand-in would have to reimplement a protocol this repository has no
    reference for. The refusal that produces is pinned by the sibling test.
    """
    run_mace_train(
        {"format": "mliap"},
        extra_argv=[str(anchor_copy)],
        script=CREATE_LAMMPS_MODEL,
    )
    artifact = anchor_copy.parent / (anchor_copy.name + "-mliap_lammps.pt")
    assert artifact.exists(), "mliap export produced no artefact"

    assert MLIAP_GOLDEN.exists(), (
        f"{MLIAP_GOLDEN} is missing; regenerate it with "
        f"`python tests/golden/regenerate.py --target lammps "
        f"--i-know-what-i-am-doing`"
    )
    committed = json.loads(MLIAP_GOLDEN.read_text(encoding="utf-8"))["interface"]
    produced = mliap_interface(artifact)
    assert produced == committed, (
        f"the ML-IAP export's declared interface changed:\n"
        f"  committed {committed}\n  produced  {produced}\n"
        f"Every LAMMPS input script in the field depends on these, so this is "
        f"a deployment break rather than a test to update."
    )


@pytest.mark.cueq
def test_the_mliap_export_refuses_a_multilayer_model_without_ghost_exchange(
    anchor_copy,
):
    """The refusal is the contract, because the alternative was silence.

    A multi-layer model under a stock, non-KOKKOS LAMMPS used to reach the
    second interaction layer and die on a bare ``AttributeError`` with
    nothing pointing at the build. It now refuses up front and names both the
    cause and the two fixes. That message is what makes the real-tier
    restriction discoverable, so it is pinned as text.
    """
    import torch  # noqa: PLC0415

    run_mace_train(
        {"format": "mliap"},
        extra_argv=[str(anchor_copy)],
        script=CREATE_LAMMPS_MODEL,
    )
    artifact = anchor_copy.parent / (anchor_copy.name + "-mliap_lammps.pt")
    loaded = torch.load(artifact, map_location="cpu", weights_only=False)

    class StockLammpsData:
        """A non-KOKKOS ML-IAP data object: no ``forward_exchange``."""

        nlocal = 6
        ntotal = 12
        npairs = 30
        elems = np.zeros(12, dtype=np.int64)

    with pytest.raises(RuntimeError) as raised:
        loaded.compute_forces(StockLammpsData())
    message = str(raised.value)
    assert "forward_exchange" in message, message
    assert "PKG_KOKKOS" in message, message
    assert "single-layer" in message, message
