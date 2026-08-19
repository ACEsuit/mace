"""References for the two published non-energy models. Needs a download.

Not part of ``--target all``: regenerating these fetches published releases,
and MACE-Polar additionally needs the ``graph_longrange`` extra. Folding it
into the default sequence would mean ``--target all`` fails on a plain
development box that can regenerate everything else perfectly well.
"""

from __future__ import annotations

from tests.golden import harness
from tests.golden.paths import REPO_ROOT

ORDER = 42
HELP = "references for the published dipole and polarizability models"
IN_ALL = False

#: The MACE-Polar checkpoint the golden pins. The smallest of the three
#: published sizes on purpose: all three share one architecture, so the extra
#: minutes and megabytes of the larger ones buy no additional coverage of the
#: assembly a rewrite has to reproduce. Changing this key changes which
#: published model is pinned, so it is a regeneration, not a tweak.
POLAR_MODEL = "polar-1-s"

#: Stated, not "everything this model can evaluate". The golden manifest is
#: shared with every other family, so a bare load_fixtures() silently grows
#: the moment one of them commits a fixture, and a reference whose fixture
#: set moves under it is not edit-locked at all. MACE-Polar is an energy
#: model and gets the full anchor set; naming it is what keeps it that.
#: Named, not selected by the "molecular" tag alone. Every magnetic fixture
#: carries that tag too, so a tag-only selection silently absorbs the
#: magnetic group the moment it is committed -- and MACE-MDP covers iron, so
#: it would evaluate them happily and shift a reference nobody meant to move.
MDP_FIXTURES = ("dimer_short", "isolated_atom", "water_cluster")

#: MACE-Polar is an energy model and gets the full anchor set.
POLAR_FIXTURES = (
    "dimer_short",
    "isolated_atom",
    "slab_vacuum",
    "slab_zero_vacuum",
    "triclinic_bulk",
    "water_cluster",
)


def _dielectric_projection(out: dict) -> dict:
    """The same for AtomicDielectricMACE, whose surface is the wider one.

    ``dmu_dr`` and ``dalpha_dr`` are only produced when the forward is asked
    for them, and they never reach any calculator's results at all, so the
    model route is the only door to them.
    """
    from tests.golden.routes import as_numpy  # pylint: disable=import-outside-toplevel

    projected = {
        "charges": as_numpy(out["charges"]),
        "dipole": as_numpy(out["dipole"][0]),
        "atomic_dipoles": as_numpy(out["atomic_dipoles"]),
        "polarizability": as_numpy(out["polarizability"][0]),
        "polarizability_sh": as_numpy(out["polarizability_sh"][0]),
    }
    for key in ("dmu_dr", "dalpha_dr"):
        if out.get(key) is not None:
            projected[key] = as_numpy(out[key])
    return projected


def run() -> None:
    """Snapshot the two published non-energy models. Needs a download."""
    import torch  # pylint: disable=import-outside-toplevel

    from mace.calculators.foundations_models import (  # pylint: disable=import-outside-toplevel
        mace_mdp,
        mace_polar,
    )

    from tests.golden.routes import ForwardRoute  # pylint: disable=import-outside-toplevel

    # Both loaders are called with device and dtype spelled out. mace_polar
    # defaults to float32 (mace/calculators/foundations_models.py:343) and
    # mace_mdp to float64, and a golden that inherited either default would be
    # one loader signature away from silently becoming an fp32 reference
    # asserted at the fp64 row.
    polar_calc = mace_polar(
        model=POLAR_MODEL, device="cpu", default_dtype="float64"
    )
    snapshot = harness.snapshot_outputs(
        polar_calc,
        harness.load_fixtures(names=list(POLAR_FIXTURES)),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={
            "model_class": type(polar_calc.models[0]).__name__,
            "foundation_model": POLAR_MODEL,
        },
    )
    path = harness.write_reference(
        harness.REFERENCES_DIR / "polar_foundation_cpu_fp64.json",
        snapshot,
        provenance={
            "source": f"mace_polar(model={POLAR_MODEL!r})",
            "recipe": "tests/golden/regenerate.py --target foundation-references",
            "description": (
                "Published MACE-Polar foundation model: the energy/force "
                "surface plus the electrostatics keys the calculator exposes "
                "(dipole, charges, spins, the three energy decompositions, "
                "the density coefficients and the Fukui functions). PolarMACE "
                "emits no polarizability; that key is pinned on MACE-MDP."
            ),
            "evaluated_with": (
                "mace.calculators.MACECalculator via mace_polar, e3nn, CPU, "
                "float64 (the published weights are float32 and are upcast, "
                "which is exact)"
            ),
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    print(f"  reference {'polar':15s} -> {path.relative_to(REPO_ROOT)}")

    mdp_model = (
        mace_mdp(device="cpu", default_dtype="float64", return_raw_model=True)
        .to(torch.float64)
        .eval()
    )
    snapshot = harness.snapshot_outputs(
        ForwardRoute(
            mdp_model,
            _dielectric_projection,
            forward_kwargs={"compute_dielectric_derivatives": True},
        ),
        harness.load_fixtures(names=list(MDP_FIXTURES)),
        dtype="float64",
        device="cpu",
        backend="e3nn",
        metadata={"model_class": type(mdp_model).__name__},
    )
    path = harness.write_reference(
        harness.REFERENCES_DIR / "mdp_foundation_cpu_fp64.json",
        snapshot,
        provenance={
            "source": "mace_mdp()",
            "recipe": "tests/golden/regenerate.py --target foundation-references",
            "description": (
                "Published MACE-MDP model (AtomicDielectricMACE), the only "
                "class in the tree that emits a polarizability. Taken through "
                "the forward, because the position derivatives dmu_dr and "
                "dalpha_dr reach no calculator's results; the calculator's "
                "four shared channels are asserted against this same file."
            ),
            "evaluated_with": (
                "AtomicDielectricMACE.forward via tests/golden/routes."
                "ForwardRoute, compute_dielectric_derivatives=True, e3nn, "
                "CPU, float64"
            ),
            "tolerance_row": harness.FP64_CPU_REFERENCE.name,
        },
        allow_overwrite=True,
    )
    print(f"  reference {'mdp':15s} -> {path.relative_to(REPO_ROOT)}")

# One family, two targets: the published models are the same story as the
# tiny anchor, told at production scale.
DOC = "dipoles"
