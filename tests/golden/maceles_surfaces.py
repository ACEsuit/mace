"""How the ``MACELES`` anchor is driven, for both of its goldens.

Shared by ``tests/golden/regenerate.py`` and ``tests/golden/test_tiny_maceles.py``
so that the reference and the assertion cannot drift into evaluating two
slightly different things -- the failure mode that makes a golden either
mysterious or, worse, vacuously green.

There are two surfaces here because the family needs both, and neither one
subsumes the other:

* the **model forward**, because it is the only door the latent quantities and
  the long-range energy come out of. ``MACECalculator`` puts three of them in
  its results (``LES_alphas``, ``LES_kappas``, ``bec``) and drops
  ``les_energy``, ``latent_charges``, ``latent_dipoles`` and ``latent_quads``
  on the floor, so a calculator-only golden would claim to pin the LES family
  while pinning less than half of it;
* the **calculator**, because the external-field surface exists nowhere else.
  ``external_field`` reaches the model (it is written into the batch and ends
  up inside the Ewald sum as the field that induces the dipoles), while
  ``eps_infty``, ``keep_neutral`` and ``electric_field_unit`` never do: they
  scale a Born-charge force correction the *calculator* adds after the forward
  has returned (``mace/calculators/mace.py:815-864``). Evaluating the forward
  alone would pin none of that.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from tests.golden import harness

#: The external-field configuration of the field reference. Recorded in the
#: reference's metadata rather than as input channels, because only the first
#: of the four is a per-structure quantity that reaches the graph -- the other
#: three are configuration of the evaluation (see the note at the foot of
#: tests/golden/calculator_keys.py).
FIELD_SETTINGS: Dict[str, Any] = {
    # V/Ang, deliberately not along a lattice vector and not symmetric, so a
    # transposed or axis-permuted field would change the numbers.
    "external_field": [0.1, 0.0, -0.05],
    # The high-frequency dielectric constant. Enters as a susceptibility
    # correction, epsilon_r = eps_infty / (1 + chi) with chi from the latent
    # polarizabilities and the cell volume, and then as sqrt(epsilon_r) on the
    # field -- so this is the only knob in the reference that couples the
    # polarizability to the forces.
    "eps_infty": 2.5,
    # Removes the mean Born charge before the field force is built, i.e.
    # imposes the acoustic sum rule on the field contribution.
    "keep_neutral": True,
    # Unity, so the reference is in eV/Ang. Passed explicitly because the
    # default is a silent multiplier on the whole field correction.
    "electric_field_unit": 1.0,
}

#: The two fixtures the field reference is taken on, and why it is not all
#: six: with ``eps_infty`` set, the calculator divides by ``atoms.get_volume()``
#: (``mace/calculators/mace.py:833``), and ase refuses a volume for anything
#: whose cell is not full rank -- the two aperiodic structures, the dimer, and
#: the slab whose vacuum row is all zeros all raise
#: ``ValueError: You have N lattice vectors: volume not defined`` before any
#: MACE code runs. That refusal is pinned as a contract in the test file
#: rather than worked around here.
FIELD_FIXTURES = ("triclinic_bulk", "slab_vacuum")

#: What the model-surface reference records. Everything the LES head produces,
#: plus the energy/forces/stress the base class does.
#:
#: `node_feats` is deliberately absent: it is 128 floats per atom, which would
#: quadruple the reference file to pin the descriptor of an untrained network,
#: and it is a base-class output rather than an LES one.
MODEL_CHANNELS = (
    "energy",
    "les_energy",
    "forces",
    "stress",
    "latent_charges",
    "latent_dipoles",
    "latent_alphas",
    "latent_kappas",
    "latent_quads",
    "BEC",
)


def load_anchor(path=None) -> torch.nn.Module:
    """Load the committed anchor in float64."""
    path = path or (harness.MODELS_DIR / "tiny_maceles.model")
    return torch.load(path, weights_only=False, map_location="cpu").to(torch.float64)


def graph_for(model: torch.nn.Module, atoms, external_field=None) -> Dict[str, Any]:
    """One structure as the batch the model consumes, built in float64.

    The dtype scope is not decoration. ``AtomicData`` reads the process-wide
    default, which is float32 under pytest, and a float32 graph cast up
    afterwards has already rounded the positions -- close enough to the fp64
    row to look like agreement, far enough to make a bit-exact comparison with
    the calculator route impossible. ``tests/golden/test_tiny_anchors.py``
    measured that at about 2e-8 relative.
    """
    from mace import data  # noqa: PLC0415
    from mace.tools import torch_geometric, torch_tools, utils  # noqa: PLC0415

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    with torch_tools.default_dtype("float64"):
        config = data.config_from_atoms(atoms)
        atomic_data = data.AtomicData.from_config(
            config, z_table=z_table, cutoff=float(model.r_max)
        )
        loader = torch_geometric.dataloader.DataLoader(
            [atomic_data], batch_size=1, shuffle=False
        )
        graph = next(iter(loader)).to_dict()
    graph = {
        key: (
            value.to(torch.float64)
            if torch.is_tensor(value) and torch.is_floating_point(value)
            else value
        )
        for key, value in graph.items()
    }
    if external_field is not None:
        # Written into the batch exactly as the calculator writes it
        # (mace/calculators/mace.py:685-690), after the graph is built.
        graph["external_field"] = torch.tensor(external_field, dtype=torch.float64)
    return graph


class ModelSurface:
    """A ``golden_outputs`` hook over the ``MACELES`` forward.

    Graph-level quantities are indexed down to the single graph here rather
    than squeezed inside the schema: a leading axis of one is not the same
    claim as "this is one graph", and letting the harness squeeze it would
    make a two-graph batch snapshot as though it were a single structure.
    """

    golden_surface = harness.SURFACE_MODEL

    def __init__(self, model: torch.nn.Module, external_field=None):
        self.model = model
        self.external_field = external_field

    def golden_outputs(self, atoms) -> Dict[str, Any]:
        from mace.tools import torch_tools  # noqa: PLC0415

        periodic = harness.is_periodic(atoms)
        with torch_tools.default_dtype("float64"):
            out = self.model(
                graph_for(self.model, atoms, self.external_field),
                training=False,
                compute_force=True,
                compute_stress=periodic,
            )

        def array(name):
            return out[name].detach().numpy()

        results: Dict[str, Any] = {
            "energy": float(out["energy"][0].detach()),
            # Graph-level like the energy, and returned as [n_graphs]. This is
            # the whole point of the family: it is the term added to the total
            # *outside* the scale-shift, so pinning the total alone could not
            # tell a change in the long-range part from a change in the
            # short-range one.
            "les_energy": float(out["les_energy"][0].detach()),
            "forces": array("forces"),
            "latent_charges": array("latent_charges"),
            "latent_dipoles": array("latent_dipoles"),
            "latent_alphas": array("latent_alphas"),
            "latent_kappas": array("latent_kappas"),
            "latent_quads": array("latent_quads"),
            "BEC": array("BEC"),
        }
        if periodic:
            results["stress"] = out["stress"][0].detach().numpy()
        return results


def field_calculator(model: torch.nn.Module, settings: Optional[Mapping] = None):
    """A ``MACECalculator`` carrying the external-field surface."""
    from mace.calculators import MACECalculator  # noqa: PLC0415

    settings = dict(FIELD_SETTINGS if settings is None else settings)
    return MACECalculator(
        models=[model],
        device="cpu",
        default_dtype="float64",
        compute_bec=True,
        **settings,
    )


def bec_force_correction(
    bec: np.ndarray,
    alphas: np.ndarray,
    volume: float,
    settings: Mapping,
) -> np.ndarray:
    """The field force the calculator adds, recomputed independently.

    A transcription of ``mace/calculators/mace.py:815-864`` for the isotropic
    case, so a test can assert the documented formula rather than assert that
    the calculator agrees with itself. Kept here next to ``FIELD_SETTINGS``
    because the two have to describe the same evaluation.
    """
    contribution = np.sum(bec, axis=1) if bec.ndim == 4 else np.asarray(bec)
    if settings.get("keep_neutral", False):
        contribution = contribution - np.mean(contribution, axis=0)
    epsilon_0 = 5.52635e-3  # e^2 eV^-1 Ang^-1
    eps_infty = settings.get("eps_infty", None)
    if eps_infty is None:
        epsilon_r = 1.0
    else:
        chi = np.squeeze(alphas).sum() / volume / epsilon_0
        epsilon_r = eps_infty / (1.0 + chi)
    field = np.asarray(settings["external_field"], dtype=np.float64)
    return (
        np.einsum("nij,i->nj", contribution, field * epsilon_r**0.5)
        * settings.get("electric_field_unit", 1.0)
    )


__all__ = [
    "FIELD_FIXTURES",
    "FIELD_SETTINGS",
    "MODEL_CHANNELS",
    "ModelSurface",
    "bec_force_correction",
    "field_calculator",
    "graph_for",
    "load_anchor",
]
