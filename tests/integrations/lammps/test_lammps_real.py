"""Real tier: drive an actual LAMMPS binary with an exported MACE model.

Uses the ML-IAP unified path (`pair_style mliap unified`), which works with a
stock LAMMPS build that has the ML-IAP package and Python support — no MACE
plugin compilation required. The native `pair_style mace` route needs a
patched LAMMPS build and is intentionally out of scope here (containerized
job, future work).

The model here has ONE interaction layer, and that is not a size choice: past
the first layer MACE asks LAMMPS to exchange ghost node features through
`forward_exchange`, which exists only in the KOKKOS ML-IAP coupling
(`src/KOKKOS/mliap_unified_couple_kokkos.pyx`). conda-forge builds LAMMPS with
`PKG_KOKKOS=OFF` on every CPU variant, so no version of the package this job
installs can run a multi-layer model — the ghost-exchange branch is covered in
the contract tier instead (`test_mliap_exchange.py`), and the guard that turns
the multi-layer case into an actionable error is tested there too.

Capability `bin_lammps`: skipped wherever LAMMPS is absent, enforced
(skip=fail) in the nightly job that installs it. The mliap export also needs
cueq (the exporter converts to the cueq layout).
"""

import shutil

import numpy as np
import pytest
import torch

from tests.helpers import REPO_ROOT, run_mace_train
from tests.integrations.lammps._harness import model_batch, water_unit_cell

CREATE_LAMMPS_MODEL = REPO_ROOT / "mace" / "cli" / "create_lammps_model.py"

pytestmark = [pytest.mark.bin_lammps, pytest.mark.cueq]


@pytest.fixture(name="mliap_artifact")
def fixture_mliap_artifact(trained_tiny_1layer_model_path, tmp_path):
    dest = tmp_path / "tiny.model"
    shutil.copy(trained_tiny_1layer_model_path, dest)
    run_mace_train(
        {"format": "mliap"}, extra_argv=[str(dest)], script=CREATE_LAMMPS_MODEL
    )
    return dest.parent / (dest.name + "-mliap_lammps.pt")


def test_lammps_mliap_energy_matches_python(mliap_artifact):
    import lammps
    from lammps.mliap import activate_mliappy

    # `create_atoms ... single` silently discards a point that falls outside
    # the box, and the cell's oxygen sits at y=-2. The cell is periodic, so
    # wrapping moves it to y=2 without changing the energy either side
    # computes -- it only keeps both sides on the same three atoms.
    atoms = water_unit_cell()
    atoms.wrap()

    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
    assert lmp.has_style("pair", "mliap"), "LAMMPS build lacks the ML-IAP package"
    activate_mliappy(lmp)

    cell = atoms.cell.lengths()
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        f"region box block 0 {cell[0]} 0 {cell[1]} 0 {cell[2]}",
        "create_box 2 box",
        "mass 1 15.999",  # O
        "mass 2 1.008",  # H
    ]
    for cmd in lines:
        lmp.command(cmd)
    type_of = {8: 1, 1: 2}
    for i, (num, pos) in enumerate(zip(atoms.numbers, atoms.positions), start=1):
        lmp.command(
            f"create_atoms {type_of[num]} single {pos[0]} {pos[1]} {pos[2]} "
            "units box"
        )
    assert lmp.get_natoms() == len(atoms), (
        f"LAMMPS created {lmp.get_natoms()} of the {len(atoms)} atoms; a "
        "dropped atom shows up only as a wrong energy further down"
    )
    lmp.command(f"pair_style mliap unified {mliap_artifact} 0")
    lmp.command("pair_coeff * * O H")
    lmp.command("run 0")

    e_lammps = lmp.get_thermo("pe")
    lmp.close()

    model = torch.load(
        mliap_artifact.parent / "tiny.model", map_location="cpu"
    ).double()
    out = model(model_batch(model, atoms), training=False, compute_force=False)
    e_python = out["energy"].item()

    assert np.isfinite(e_lammps)
    assert np.isclose(e_lammps, e_python, atol=1e-4), (
        f"LAMMPS pe={e_lammps} vs python {e_python}"
    )
