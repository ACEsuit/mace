"""Where every legacy model output goes in the declarative specification.

The frozen models emit 43 distinct keys from their ``forward`` methods. This
module says, for each one, what it becomes: a declared observable, a derivative
of one, or a row saying explicitly that it is not an observable and naming the
mechanism that owns it instead. A key with no row fails the test beside this
file, so a key added to a legacy forward cannot slip through unclassified.

The 43 are **not listed here**. They are read out of ``mace/modules/models.py``
and ``mace/modules/extensions.py`` by ``tests/golden/surface_scan.py``, which
follows keys assigned onto the returned object as well as dict literals -- the
self-consistent model assigns its three diagnostics after the fact, so an
extraction that stopped at return literals would stop at 40 and lose them
silently.

Two pieces of metadata are likewise derived rather than retyped. The golden
harness already declares a kind and a unit for every one of these keys, so the
per-atom/per-graph classification and the unit string come from there and the
rows below carry only what a schema cannot know: the spherical-tensor shape,
and the decision.

One convention worth stating once. A rank-2 Cartesian quantity -- a stress, a
virial, a polarizability -- is stored by the legacy models as a full 3x3, nine
numbers, while its irreps declaration ``0e+2e`` spans six. That is not a
contradiction: the declaration says what the quantity *is* under rotation, and
the 3x3 is a layout with three redundant entries. Reconciling the two is the
head's job, and it is written down here because the mismatch otherwise looks
like an error in these rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from tests.golden import harness, surface_scan

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The two files the ticket's count is defined over. The scanner discovers
#: forwards across the whole package -- the LAMMPS and torchsim wrappers define
#: one too -- and those are deployment layers with their own tickets, so this
#: surface is named rather than discovered.
LEGACY_MODEL_SOURCES = (
    REPO_ROOT / "mace" / "modules" / "models.py",
    REPO_ROOT / "mace" / "modules" / "extensions.py",
)

#: The ase calculator, layer (b) of the three-layer surface.
LEGACY_CALCULATOR_SOURCES = (REPO_ROOT / "mace" / "calculators" / "mace.py",)

PER_ATOM_KINDS = frozenset(
    {
        harness.PER_ATOM_SCALAR,
        harness.PER_ATOM_VECTOR,
        harness.PER_ATOM_TENSOR,
        harness.PER_ATOM_MATRIX,
    }
)
PER_GRAPH_KINDS = frozenset(
    {
        harness.GRAPH_SCALAR,
        harness.GRAPH_VECTOR,
        harness.GRAPH_TENSOR,
        harness.GRAPH_ARRAY,
    }
)

#: The inputs a derivative row may be taken against. ``pos`` and ``cell`` are
#: the two every model has and are declared in the shipped defaults; ``magmom``
#: is the third the frozen tree actually differentiates against, and is
#: declared by whichever configuration turns the magnetic model on.
DECLARED_INPUTS = frozenset({"pos", "cell", "magmom"})


@dataclass(frozen=True)
class Spec:
    """The key becomes a declared observable.

    An irreps declaration is two independent facts, and lumping them together
    is what made a third of these rows look unanswerable. What the quantity is
    under rotation is a property of the quantity. How wide it is can be a
    property of the *model*: the number of readout layers, the maximum multipole
    order, whether an anisotropic readout was declared. A row that names both is
    complete even when it cannot write a single literal, because the model fills
    the rest in when it is declared, exactly as the ticket's own
    ``128x0e+128x1o+128x2e`` example does.

    So a row states either ``irreps`` or ``irreps_pattern`` plus ``set_by``, and
    there is no third state. Nothing here is deferred to another ticket: every
    one of the 43 was worked out, and a future row that cannot be fails the test
    rather than acquiring a TODO.

    Attributes:
        irreps: The declaration, when the quantity fixes it outright.
        irreps_pattern: The declaration with the model-dependent part named.
            Prose, not the grammar: it is read by people, and the grammar would
            have to grow placeholders to hold it.
        set_by: What the model supplies. Required with ``irreps_pattern``,
            because "it depends on the model" is not an answer until it says on
            what.
        note: Anything about the row a reader would otherwise have to rederive.
    """

    irreps: str | None = None
    irreps_pattern: str | None = None
    set_by: str = ""
    note: str = ""


@dataclass(frozen=True)
class Derivative:
    """The key is a derivative of another quantity, and is renamed by the rule.

    Attributes:
        of: The differentiated quantity.
        wrt: The declared input it is differentiated against.
        sign: The sign the frozen tree reports, so that
            ``legacy value = sign * d(of)/d(wrt)``. Read off the source and,
            where the note says so, measured. It is stated per row rather than
            taken from the rule because a row that silently agreed with the
            rule and a row nobody checked look identical. The test asserts the
            two agree, with no way to annotate a disagreement: a pair the rule
            gets wrong is either a misclassified row or a real gap in the
            grammar, and both have to be resolved rather than recorded.
        note: Why the row is worth a second look, where it is.
    """

    of: str
    wrt: str
    sign: int
    note: str = ""


@dataclass(frozen=True)
class Drop:
    """The key is not an observable. ``reason`` names what owns it instead."""

    reason: str


Disposition = Union[Spec, Derivative, Drop]


#: Every key the two frozen model modules emit, and what it becomes.
DISPOSITIONS: dict[str, Disposition] = {
    # --- energies ----------------------------------------------------------
    "energy": Spec(irreps="0e"),
    "node_energy": Spec(irreps="0e"),
    "interaction_energy": Spec(irreps="0e"),
    "les_energy": Spec(irreps="0e"),
    "electrostatic_energy": Spec(irreps="0e"),
    "electron_energy": Spec(irreps="0e"),
    "fermi_level": Spec(irreps="0e"),
    "contributions": Spec(
        irreps_pattern="<2 + num_interactions>x0e",
        set_by="the number of energy terms the model sums",
        note=(
            "the energy decomposed into its terms. `energies = [e0, "
            "pair_energy]` and one entry is appended per interaction layer "
            "(mace/modules/models.py:361, :398), so the extent is the "
            "isolated-atom reference plus the ZBL pair term plus one per "
            "layer. Measured on the tiny_mace anchor, which has two layers: "
            "shape (1, 4)."
        ),
    ),
    # --- charges and potentials -------------------------------------------
    "charges": Spec(irreps="0e"),
    "latent_charges": Spec(irreps="0e"),
    "spins": Spec(
        irreps="0e",
        note=(
            "the per-atom spin population, a scalar. Not to be confused with a "
            "magnetic moment, which is an axial vector ('1e')."
        ),
    ),
    "electrostatic_potentials": Spec(irreps="0e"),
    # --- dipoles -----------------------------------------------------------
    "dipole": Spec(irreps="1o"),
    "atomic_dipoles": Spec(irreps="1o"),
    "latent_dipoles": Spec(irreps="1o"),
    # --- rank-2 quantities -------------------------------------------------
    "polarizability": Spec(irreps="0e+2e"),
    "polarizability_sh": Spec(
        irreps="0e+2e",
        note=(
            "already the spherical form, six components. The Cartesian "
            "`polarizability` above is the same quantity in the 3x3 layout, "
            "which is what makes the pair a useful check on the head."
        ),
    ),
    "virials": Spec(
        irreps="0e+2e",
        note=(
            "the NEGATED cell derivative, and the sign is not a detail: "
            "`compute_forces_virials` computes the stress from the raw "
            "gradient and negates the virial only in its return statement "
            "(mace/modules/utils.py:107-115), so the frozen tree reports "
            "`virials = -dE/dstrain` and `stress = +dE/dstrain / V`. Measured "
            "on the tiny_scaleshift anchor over an fcc cell: "
            "max|stress * V + virials| = 1.2e-35, exactly zero, while "
            "max|stress * V - virials| = 6.5e-3. The two therefore differ by a "
            "sign as well as by the volume. `stress` is the row that carries "
            "the derivative naming; `virials` stays an observable of its own "
            "because the spec has no way to say 'the same derivative in "
            "another normalization and the opposite sign', which is a question "
            "for the head that produces them."
        ),
    ),
    "atomic_virials": Spec(irreps="0e+2e"),
    "atomic_stresses": Spec(irreps="0e+2e"),
    # --- families whose shape is a model hyperparameter ---------------------
    "density_coefficients": Spec(
        irreps_pattern="two concatenated ladders 0e+1o+2e+...+<atomic_multipoles_max_l>",
        set_by="atomic_multipoles_max_l",
        note=(
            "the spin-summed electron density in the model's multipole basis. "
            "`self.charges_irreps = 2 * o3.Irreps.spherical_harmonics("
            "atomic_multipoles_max_l)` (mace/modules/extensions.py:789), and "
            "in e3nn that product concatenates the ladder twice rather than "
            "doubling each multiplicity, so the layout is two ladders end to "
            "end even though the content simplifies to 2x0e+2x1o+2x2e+... "
            "Dimension 8, 18, 32 for max_l 1, 2, 3."
        ),
    ),
    "spin_density": Spec(
        irreps_pattern="two concatenated ladders 0e+1o+2e+...+<atomic_multipoles_max_l>",
        set_by="atomic_multipoles_max_l",
        note=(
            "the alpha minus beta difference of the same basis as "
            "`density_coefficients` (mace/modules/extensions.py:1250), so it "
            "carries identical irreps."
        ),
    ),
    "spin_charge_density": Spec(
        irreps_pattern=(
            "<2 spin channels> x two concatenated ladders "
            "0e+1o+2e+...+<atomic_multipoles_max_l>"
        ),
        set_by="atomic_multipoles_max_l, and the fixed pair of spin channels",
        note=(
            "the density before the spin channels are summed or subtracted: "
            "`spin_charge_density.view(shape[0], 2, -1)` "
            "(mace/modules/extensions.py:1094). `density_coefficients` is its "
            "sum over that axis and `spin_density` its difference."
        ),
    ),
    "fukui_functions": Spec(
        irreps="2x0e",
        note=(
            "two scalars per atom, one per spin channel, and not model "
            "dependent at all: `self.fukui_source_map` is a readout whose "
            "output irreps are the literal `o3.Irreps(\"2x0e\")` "
            "(mace/modules/extensions.py:838-842). They are added to the l=0 "
            "component of each spin channel of the density."
        ),
    ),
    "BEC": Spec(
        irreps_pattern="<1 or 2>x(0e+1e+2e)",
        set_by=(
            "whether the model passes latent dipoles to LES alongside the "
            "latent charges, which adds a second channel"
        ),
        note=(
            "NOT the same quantity as `dmu_dr`, which is the tempting reading "
            "and the wrong one. LES builds a polarization from its own latent "
            "charges with the mean removed and an epsilon^(1/2) factor, takes "
            "it through a Berry phase under periodic boundary conditions, and "
            "differentiates *that* against the positions, dephasing and "
            "projecting the result by the cell "
            "(les/module/bec.py:56-93). `dmu_dr` differentiates the "
            "dielectric model's `dipole` readout, a different quantity in a "
            "different gauge and a different unit (e against Debye/Ang). "
            "Measured on the tiny_maceles anchor, the shape is "
            "(n_atoms, 2, 3, 3), not (n_atoms, 3, 3): the leading pair is the "
            "charge-derived tensor and the latent-dipole-derived one, and "
            "whether that axis is there at all depends on whether the model "
            "passes latent dipoles. Each 3x3 block is a general rank-2 tensor "
            "rather than a symmetric one (max|B - B^T| = 0.20 on that anchor), "
            "so a block is 0e+1e+2e and the channel axis is the multiplicity "
            "DEP-1a has to fix."
        ),
    ),
    "latent_kappas": Spec(
        irreps="0e",
        note=(
            "one scalar per atom. The LES signature documents it as "
            "[n_atoms, ] (les/les.py:89) and `les_kappa_readouts` is a scalar "
            "readout; measured (3,) on the tiny_maceles anchor."
        ),
    ),
    "latent_alphas": Spec(
        irreps_pattern="0e, or 0e+2e",
        set_by="whether the model declares the anisotropic alpha readouts",
        note=(
            "an atomic polarizability, isotropic by default and anisotropic "
            "when `use_induced_dipoles` brings in `les_alpha_2e_readouts`. "
            "That branch reads out a spherical 0e+2e and expands it through "
            "`spherical_to_cartesian` (mace/modules/extensions.py:485-494), "
            "then squares it as A A^T, which is symmetric, so the anisotropic "
            "form is 0e+2e and not the general 0e+1e+2e. LES accepts both "
            "shapes explicitly (les/les.py:133-136). Measured (3,) on the "
            "tiny_maceles anchor, which is the isotropic path."
        ),
    ),
    "latent_quads": Spec(
        irreps="2e",
        note=(
            "an atomic quadrupole. Stored as a Cartesian 3x3 "
            "(les/les.py:88), but the model subtracts the trace explicitly "
            "(mace/modules/extensions.py:549-552), so it is symmetric and "
            "traceless, which is exactly 2e and five numbers rather than nine. "
            "Measured on the tiny_maceles anchor: max|A - A^T| = 2.7e-20 and "
            "|trace| = 5.4e-20."
        ),
    ),
    # --- derivatives -------------------------------------------------------
    "forces": Derivative(
        of="energy",
        wrt="pos",
        sign=-1,
        note="mace/modules/utils.py:115 returns `-1 * forces`.",
    ),
    "stress": Derivative(
        of="energy",
        wrt="cell",
        sign=+1,
        note=(
            "positive, and only because the stress is built from the raw "
            "gradient before the virial is negated. See the `virials` row: the "
            "two are not the same sign."
        ),
    ),
    "magforces": Derivative(
        of="energy",
        wrt="magmom",
        sign=-1,
        note=(
            "mace/modules/utils.py returns `-mag_forces` from both "
            "`compute_forces_virials_magforces` and `compute_forces_magforces`."
        ),
    ),
    "dmu_dr": Derivative(
        of="dipole",
        wrt="pos",
        sign=+1,
        note=(
            "mace/modules/models.py:1169 differentiates the model's own "
            "`dipole` key, and `compute_dielectric_gradients` returns the "
            "gradient with no sign flip."
        ),
    ),
    "dalpha_dr": Derivative(
        of="polarizability",
        wrt="pos",
        sign=+1,
        note=(
            "the Cartesian polarizability flattened to nine components "
            "(mace/modules/models.py:1173-1176), differentiated the same way "
            "as `dmu_dr`."
        ),
    ),
    # --- not observables ---------------------------------------------------
    "hessian": Drop(
        reason=(
            "the second derivative of the energy with respect to the "
            "positions, and the derivative grammar is first order by design: "
            "it names d(quantity)/d(input) for a declared quantity and a "
            "declared input, and all three of its special cases are first "
            "order. Reading it instead as the first derivative of the forces "
            "does not rescue it, and the sign is how that shows: "
            "`compute_hessians_vmap` differentiates `-1 * forces` "
            "(mace/modules/utils.py:168), so the key holds +d2E/dpos2, which "
            "is MINUS d(forces)/d(pos). Measured on the tiny_scaleshift anchor "
            "against a central difference of the forces, "
            "max|hessian[:, 0] + dF/dx| = 1.8e-10 against "
            "max|hessian[:, 0] - dF/dx| = 0.32. It is a real output and it is "
            "not lost: the derivative engine owns second derivatives, the same "
            "way the export path owns `edge_forces`."
        )
    ),
    "displacement": Drop(
        reason=(
            "the symmetric strain handle the cell derivative is taken against. "
            "It is created as zeros to attach the cell to the autograd graph "
            "and nothing ever writes to it, so its value is identically zero "
            "on every structure. It belongs to the derivative engine, not to "
            "the output surface."
        )
    ),
    "edge_forces": Drop(
        reason=(
            "indexed by the neighbour list, so it is neither per-atom nor "
            "per-graph and the spec's classification cannot express it. It is "
            "the per-edge decomposition the LAMMPS pair style sums into its "
            "virial, and belongs to the export path."
        )
    ),
    "node_feats": Drop(
        reason=(
            "the backbone's node features: what every head reads, not what one "
            "produces. No dataset carries a target for it, so declaring it "
            "would make the head-creation check unsatisfiable. The evaluation "
            "CLI exposes it as `descriptors`, which is CLI-1's layer."
        )
    ),
    "external_field": Drop(
        reason=(
            "an input echoed back, which is how the golden harness classifies "
            "it too. It reaches v1 as a declared input feature, which is also "
            "what makes a derivative against it expressible."
        )
    ),
    "total_charge": Drop(
        reason=(
            "an input echoed back, like `external_field`. Declared as an input "
            "feature rather than as an observable."
        )
    ),
    "charges_history": Drop(
        reason=(
            "the iterates of the electrostatic self-consistency loop: how a "
            "fixed point was reached, not the fixed point. Run telemetry, and "
            "it belongs in the stage's log."
        )
    ),
    "scf_energy_history": Drop(
        reason="run telemetry of the self-consistent loop, like `charges_history`."
    ),
    "scf_steps": Drop(
        reason="run telemetry of the self-consistent loop, like `charges_history`."
    ),
    "equilibrated_magmom": Drop(
        reason=(
            "an input the self-consistent stage converged, exposed through the "
            "stage result rather than as a model observable. Note the golden "
            "harness classifies it as an output channel, because from where it "
            "sits it is one; the distinction is which mechanism owns it."
        )
    ),
}


def legacy_model_keys() -> set[str]:
    """The keys the two frozen model modules can return, read from the source."""
    return surface_scan.scan_model_surface(list(LEGACY_MODEL_SOURCES)).all_keys


def legacy_calculator_keys() -> set[str]:
    """The keys the ase calculators can write into ``results``."""
    return surface_scan.scan_calculator_surface(
        list(LEGACY_CALCULATOR_SOURCES)
    ).all_keys


def legacy_eval_keys() -> set[str]:
    """The names the evaluation CLI writes onto its structures, unprefixed."""
    scan, _stores = surface_scan.scan_eval_surface()
    return scan.all_keys


def channel_of(key: str) -> harness.Channel | None:
    """The golden harness's declaration for ``key`` on the model surface."""
    name = harness.resolve_channel(key, harness.SURFACE_MODEL)
    return None if name is None else harness.CHANNELS[name]


def per_atom_of(key: str) -> bool | None:
    """Whether ``key`` is per-atom, derived from its harness kind.

    ``None`` for the kinds that are neither -- a per-edge quantity, a hessian,
    a gradient whose atom axis is not the leading one. A key that lands here
    cannot be a plain observable, and the test asserts exactly that.
    """
    channel = channel_of(key)
    if channel is None:
        return None
    if channel.kind in PER_ATOM_KINDS:
        return True
    if channel.kind in PER_GRAPH_KINDS:
        return False
    return None
