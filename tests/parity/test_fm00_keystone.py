"""The keystone: a trained legacy model, run through the whole v1 stack.

Every piece has been checked on its own by now. This is the first time a real
trained model goes in one end and energies and forces come out the other, and
it is the only test that can catch a mistake in how the pieces fit rather than
in the pieces.

It compares against the **live** frozen model in the same process rather than
against a stored number, so a disagreement is between two implementations
evaluated on the same inputs with nothing in between.
"""

from __future__ import annotations

from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from mace_core.config.precision import PrecisionConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import load_checkpoint, save_checkpoint

from tests.parity.anchors import load_anchor
from tests.parity.fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    model_arguments,
    transfer_weights,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
ENERGY = ObservableSpec(
    name="energy",
    irreps="0e",
    per_atom=False,
    units="eV",
    derivatives=[{"wrt": "pos", "name": "forces", "sign": -1, "units": "eV/A"}],
)

#: The fp64 row of the golden tolerance table. Stated here because this package
#: cannot import that harness, and a change to it is its own reviewed PR.
ENERGY_TOLERANCE = 1e-6
FORCE_TOLERANCE = 1e-6


def convert(legacy):
    """The whole conversion, config and weights."""
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    config = build_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        ["default"],
        AtomicNumberTable(numbers),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
        zbl_in_scale_shift=getattr(legacy, "scale_shift", None) is not None,
        widen_node_energy=getattr(legacy, "scale_shift", None) is not None,
    )
    model = MACEModel(
        ReferenceBackend(),
        observables=[ENERGY],
        energy_head=head,
        **model_arguments(config),
    )
    transfer_weights(legacy, model, config["correlation"])
    return model, config


def legacy_batch(legacy, atoms):
    from mace import data as legacy_data
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools import torch_geometric

    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    item = legacy_data.AtomicData.from_config(
        legacy_data.config_from_atoms(atoms),
        z_table=LegacyTable(numbers),
        cutoff=float(legacy.r_max),
    )
    loader = torch_geometric.DataLoader(dataset=[item], batch_size=1, shuffle=False)
    return next(iter(loader))


def v1_graph(atoms, numbers, cutoff):
    from mace_core.data import KeySpecification, configuration_from_atoms
    from mace_torch.graph import build_batch
    from mace_torch.neighbors import resolve_neighbor_backend

    return build_batch(
        [configuration_from_atoms(atoms, KeySpecification.from_defaults())],
        atomic_number_table=AtomicNumberTable(numbers),
        cutoff=cutoff,
        neighbor_backend=resolve_neighbor_backend(),
        dtype=torch.get_default_dtype(),
    )


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
def test_a_converted_anchor_matches_the_live_legacy_model(fp64, anchor):
    """Energy and forces, on every structure of the training fixture.

    Both model classes, so the two places the frozen tree puts the pair
    repulsion are exercised: inside the scaled sum in one and outside it in the
    other.
    """
    legacy = load_anchor(anchor)
    model, config = convert(legacy)
    engine = DerivativeEngine(model, ENERGY)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]

    structures = ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", index=":")
    interesting = [atoms for atoms in structures if len(atoms) > 1]
    assert len(interesting) >= 5, "the fixture has too little in it to say much"

    worst_energy = worst_force = 0.0
    for atoms in interesting:
        reference = legacy(
            legacy_batch(legacy, atoms).to_dict(), training=False, compute_force=True
        )
        result = engine(v1_graph(atoms, numbers, config["cutoff"]), compute=("forces",))
        worst_energy = max(
            worst_energy,
            abs(
                float(reference["energy"].detach())
                - float(result.total_energy.detach())
            ),
        )
        worst_force = max(
            worst_force,
            float((reference["forces"].detach() - result.forces.detach()).abs().max()),
        )

    assert worst_energy < ENERGY_TOLERANCE, (
        f"the converted model's energy differs by {worst_energy:.3e} eV"
    )
    assert worst_force < FORCE_TOLERANCE, (
        f"the converted model's forces differ by {worst_force:.3e} eV/A"
    )
    # Far below the gate rather than just inside it: the only inexact step in
    # the whole conversion is the basis projection, at 1e-14.
    assert worst_energy < 1e-12 and worst_force < 1e-12, (
        f"the conversion is inside the gate at {worst_energy:.3e} eV and "
        f"{worst_force:.3e} eV/A, but far enough above rounding that something "
        f"is approximating where nothing should be"
    )


def test_a_converted_anchor_survives_the_checkpoint(fp64, tmp_path):
    """Converted, written, read back through the ordinary loader, still right.

    The loader has no converter-specific path: if the neutral format could not
    carry a converted model, this is where it would show.
    """
    legacy = load_anchor("tiny_scaleshift.model")
    model, config = convert(legacy)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    values, scale, shift = energy_constants_to_canonical(legacy)

    save_checkpoint(tmp_path / "converted", model, config)

    def rebuild(stored):
        head = EnergyOutputHead(
            ResolvedE0s(values),
            ["default"],
            AtomicNumberTable(numbers),
            ScaleShiftSpec("std", scale, shift),
            PrecisionConfig(),
        )
        return MACEModel(
            ReferenceBackend(),
            observables=[ENERGY],
            energy_head=head,
            **model_arguments(stored),
        )

    restored = load_checkpoint(tmp_path / "converted", rebuild(config))

    atoms = ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", index=3)
    reference = legacy(
        legacy_batch(legacy, atoms).to_dict(), training=False, compute_force=True
    )
    graph = v1_graph(atoms, numbers, config["cutoff"])
    result = DerivativeEngine(restored, ENERGY)(graph, compute=("forces",))

    assert (
        abs(float(reference["energy"].detach()) - float(result.total_energy.detach()))
        < 1e-12
    )


#: The fixtures these anchors can be run on: the three elements they were
#: trained for, and every cell regime the graph builder distinguishes.
TINY_FIXTURES = (
    "dimer_short",
    "isolated_atom",
    "slab_vacuum",
    "slab_zero_vacuum",
    "triclinic_bulk",
    "water_cluster",
)


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
@pytest.mark.parametrize("fixture", TINY_FIXTURES)
def test_energy_forces_and_stress_match_on_every_tiny_fixture(
    fp64, isolated, anchor, fixture
):
    """The whole fixture set, one case per cell regime.

    Two of them are periodic in two directions only, which is the case that
    separates the graph builder's three effective-cell regimes, and one is a
    single atom, which is the case where every derivative is zero and a wrong
    sign would hide.
    """
    legacy = load_anchor(anchor)
    model, config = convert(legacy)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]

    atoms = ase.io.read(GOLDEN / f"fixtures/{fixture}.xyz", index=0)
    reference = legacy(
        legacy_batch(legacy, atoms).to_dict(),
        training=False,
        compute_force=True,
        compute_stress=True,
    )
    result = DerivativeEngine(model, ENERGY)(
        v1_graph(atoms, numbers, config["cutoff"]), compute=("forces", "stress")
    )

    energy = abs(
        float(reference["energy"].detach()) - float(result.total_energy.detach())
    )
    force = float((reference["forces"].detach() - result.forces.detach()).abs().max())
    assert energy < ENERGY_TOLERANCE, f"{fixture}: energy differs by {energy:.3e} eV"
    assert force < FORCE_TOLERANCE, f"{fixture}: forces differ by {force:.3e} eV/A"

    if reference.get("stress") is not None:
        stress = float(
            (
                reference["stress"].detach().reshape(-1)
                - result.stress.detach().reshape(-1)
            )
            .abs()
            .max()
        )
        assert stress < ENERGY_TOLERANCE, f"{fixture}: stress differs by {stress:.3e}"
        if not any(atoms.pbc):
            assert float(result.stress.abs().max()) == 0.0, (
                f"{fixture} has no periodic direction, so it has no volume to "
                f"divide by and its stress has to be masked to zero"
            )


def test_the_converted_energy_differentiates_twice(fp64, isolated):
    """The path force training backpropagates through.

    On the two-atom fixture, because `gradgradcheck` evaluates the model once
    per degree of freedom per direction and the claim is about the derivative
    rather than about the size of the structure.
    """
    legacy = load_anchor("tiny_scaleshift.model")
    model, config = convert(legacy)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    atoms = ase.io.read(GOLDEN / "fixtures/dimer_short.xyz", index=0)
    graph = v1_graph(atoms, numbers, config["cutoff"])

    def energy(positions):
        moving = dict(graph)
        moving["positions"] = positions
        return DerivativeEngine(model, ENERGY)(moving, compute=()).total_energy

    start = graph["positions"].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(energy, (start,), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(energy, (start,), eps=1e-6, atol=1e-5)


def test_a_model_without_the_pair_repulsion_converts_too(fp64, isolated):
    """Every committed anchor has the repulsion on, so the envelope order was
    once read off the repulsion and defaulted to 6 without it. A legacy model
    with the default envelope of order 5 and no repulsion then converted into a
    different function, off by 0.18 eV on 64 atoms of diamond."""
    from ase.build import bulk
    from e3nn import o3

    from mace import modules

    torch.manual_seed(0)
    legacy = modules.ScaleShiftMACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-1.0]),
        avg_num_neighbors=8.0,
        atomic_numbers=[6],
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        use_reduced_cg=False,
        pair_repulsion=False,
    ).to(torch.float64)
    atoms = bulk("C", "diamond", a=3.57, cubic=True).repeat(2)
    atoms.positions += np.random.default_rng(0).normal(scale=0.05, size=(64, 3))
    reference = legacy(
        legacy_batch(legacy, atoms).to_dict(), training=False, compute_force=True
    )
    model, config = convert(legacy)
    result = DerivativeEngine(model, ENERGY)(
        v1_graph(atoms, [6], config["cutoff"]), compute=("forces",)
    )
    assert config["cutoff_order"] == 5
    assert (
        abs(float(reference["energy"]) - float(result.total_energy)) < ENERGY_TOLERANCE
    )
    assert float((reference["forces"] - result.forces).abs().max()) < FORCE_TOLERANCE
