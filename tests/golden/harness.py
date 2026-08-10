"""Framework-agnostic golden harness.

This module is the single machine both the legacy stack and the rewritten one
are allowed to share, so it deliberately depends on nothing but the standard
library, numpy and ase. It must never import the package under test: the
parity suites that consume it live outside the legacy tree and are forbidden
from importing it, and a single convenience import here would make the whole
comparison machinery unavailable to them. That constraint is checked by a
test in this directory (and by ``grep``), not merely documented.

What it provides:

* :func:`load_fixtures` -- the committed structures, keyed by name, with the
  neighbour-list regime each one exists to reach recorded in a manifest.
* :func:`snapshot_outputs` -- evaluate anything calculator-shaped over those
  structures and return a schema-checked snapshot dict.
* :func:`write_reference` / :func:`load_reference` -- JSON (de)serialisation
  carrying explicit dtype and unit fields.
* :func:`compare_to_reference` -- assert a fresh snapshot against a committed
  one at a named row of :data:`TOLERANCES`.

Units are eV and Angstrom throughout, as in ``ase.units``; every channel
carries its unit string in the snapshot, and a comparison fails if the units
disagree, because a silent unit change is exactly the class of regression a
golden exists to catch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from ase.atoms import Atoms
from ase.io import read as ase_read

GOLDEN_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = GOLDEN_ROOT / "fixtures"
REFERENCES_DIR = GOLDEN_ROOT / "references"
MODELS_DIR = GOLDEN_ROOT / "models"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"

#: Bumped only when the on-disk layout of a reference changes incompatibly.
#: Every committed reference records it, and a comparison refuses to run
#: across a bump rather than silently comparing two different layouts.
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Tolerance table -- the single source of truth.
#
# Every golden and parity test imports a row from here. Do not restate a
# number in a test file, and do not add a fourth table: two others already
# exist in this repository and this one is reconciled against both rather
# than competing with them.
#
#   * tests/backends/backend_parity.py uses 1e-6 (fp64) / 1e-4 (fp32) for
#     parameter-gradient parity between accelerated kernels and e3nn, in the
#     same process on the same device. That is a different measurement --
#     gradients, not outputs, and no cross-machine term -- so it is not
#     superseded by this table. Its fp64 number is nevertheless the same 1e-6
#     the reference row below uses, which is the reconciliation: the two
#     agree on what fp64 agreement between two implementations costs.
#
#   * tests/extensions/polar/test_polar_models.py ATOL_BY_DTYPE holds
#     1e-9 (fp64) / 5e-5 (fp32) for a committed-JSON regression of a
#     published model on downloaded structures -- the closest analogue to
#     this table, and it passes in CI. Its fp64 value is tighter than the row
#     below; a passing 1e-9 also passes at 1e-6, so the two do not conflict,
#     and the looser row is kept because these references are asserted across
#     four Python versions and any contributor's machine, where the polar
#     regression runs in one job. Its fp32 value is *adopted* here as the
#     absolute floor of the fp32 row: that file documents 5e-6 failing in CI
#     on single force components, which is measured evidence this table would
#     otherwise have to invent.
#
# Tolerances are edit-locked: changing one is a separate, justified, reviewed
# change, never a line inside a feature change. A test that needs a looser
# number is a test that found something.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """One row of the tolerance table.

    ``atol``/``rtol`` are used exactly as ``numpy.isclose`` does:
    ``|a - b| <= atol + rtol * |b|``, with ``b`` the committed reference.
    """

    name: str
    atol: float
    rtol: float
    rationale: str


#: fp64, legacy e3nn kernels, CPU, compared across machines and Python
#: versions. This is the row every committed CPU reference is asserted at.
FP64_CPU_REFERENCE = Tolerance(
    name="fp64_cpu_reference",
    atol=1e-6,
    rtol=0.0,
    rationale=(
        "fp64 e3nn on CPU, cross-machine and cross-Python: 1e-6 eV for "
        "energies, 1e-6 eV/Ang for forces. Absolute, not relative, so a "
        "near-zero force component is held to the same bound as a large one. "
        "Measured headroom on the committed anchors, one host, 1/4/6 threads: "
        "the deviation is exactly 0.0 on every channel, so the bound is "
        "carrying the cross-machine term alone."
    ),
)

#: fp64, accelerated kernels (cueq/oeq) on GPU, compared against the CPU
#: e3nn reference above. Looser by an order of magnitude because the
#: comparison crosses both a kernel implementation and a device.
FP64_ACCELERATED_BACKEND = Tolerance(
    name="fp64_accelerated_backend",
    atol=1e-5,
    rtol=0.0,
    rationale=(
        "Accelerated fp64 kernels evaluated on an accelerator against the "
        "committed CPU reference. In-process same-device parity already "
        "holds at 1e-6 (tests/backends/backend_parity.py); this row adds "
        "the cross-device and cross-machine term."
    ),
)

#: fp32, any backend. Relative, with an absolute floor, because a pure
#: relative bound is meaningless on the many force components that are
#: legitimately near zero.
FP32 = Tolerance(
    name="fp32",
    atol=5e-5,
    rtol=1e-3,
    rationale=(
        "fp32 sums in an order that depends on thread count and on which "
        "BLAS kernel is selected, so fp32 references are not reproducible "
        "bit for bit across machines. 1e-3 relative, floored at the 5e-5 "
        "absolute value measured in tests/extensions/polar (where 5e-6 was "
        "shown to fail in CI on single force components). Checked against "
        "the committed anchors evaluated in fp32 against their fp64 "
        "references: the tightest case clears the bound by about an order of "
        "magnitude (1.1e-5 against 1.0e-4, cluster forces)."
    ),
)

#: Bit-for-bit. Not selectable by a golden for its *outputs* -- no numerical
#: result reproduces exactly across machines, which is why the three rows
#: above exist. It is what the recorded **inputs** are always compared at, and
#: the reason is that an input is not computed: it is read verbatim off the
#: committed .xyz and handed to the model. Comparing it at an output row means
#: a perturbation smaller than the output tolerance is invisible -- a 2e-3
#: change to a magnetic moment passes at the fp32 row, and the snapshot then
#: claims to reproduce a reference it was not taken at. There is nothing to
#: trade off here: two reads of the same file either agree exactly or the
#: fixture changed.
EXACT = Tolerance(
    name="exact",
    atol=0.0,
    rtol=0.0,
    rationale=(
        "Bit-for-bit. Used for the recorded inputs, which are read from the "
        "committed fixture rather than computed, so any difference at all "
        "means the evaluation was fed something the reference was not."
    ),
)

TOLERANCES: Dict[str, Tolerance] = {
    row.name: row
    for row in (FP64_CPU_REFERENCE, FP64_ACCELERATED_BACKEND, FP32, EXACT)
}


def tolerance(row: str) -> Tolerance:
    """Return a tolerance row by name, with a message listing the rows."""
    try:
        return TOLERANCES[row]
    except KeyError:
        raise KeyError(
            f"unknown tolerance row {row!r}; the table has {sorted(TOLERANCES)}. "
            f"Add a row only in a dedicated, reviewed change -- never inline "
            f"a number in a test."
        ) from None


# ---------------------------------------------------------------------------
# Channel schema
#
# A channel is one named quantity a model can emit (or consume). Its "kind"
# fixes the expected array shape, which is what lets the harness reject a
# silently reshaped output instead of comparing nonsense. Per-atom vector and
# tensor kinds are first-class from the start: the families this schema has
# to serve emit per-atom vectors (magnetic forces, latent dipoles) and
# per-atom tensors (Born effective charges), and a scalar-only schema would
# force every later golden to widen it.
# ---------------------------------------------------------------------------

GRAPH_SCALAR = "graph_scalar"  # ()
GRAPH_VECTOR = "graph_vector"  # (3,)
GRAPH_TENSOR = "graph_tensor"  # (3, 3)
GRAPH_ARRAY = "graph_array"  # (k, ...) with the extent set by the channel
PER_ATOM_SCALAR = "per_atom_scalar"  # (n_atoms,)
PER_ATOM_VECTOR = "per_atom_vector"  # (n_atoms, 3)
PER_ATOM_TENSOR = "per_atom_tensor"  # (n_atoms, 3, 3)
PER_ATOM_MATRIX = "per_atom_matrix"  # (n_atoms, ...) with the rest free
PER_EDGE_VECTOR = "per_edge_vector"  # (n_edges, 3)
# (k, n_atoms, 3): the gradient of a k-component quantity with respect to
# every Cartesian coordinate. Dielectric models emit two of these and neither
# fits any per-atom kind, because the atom axis is in the middle rather than
# leading. `k` is left free -- it is a property of the differentiated
# quantity, not of the structure -- but the atom axis is pinned, which is the
# part a golden needs. The measured extents are recorded next to the channels
# that use this kind, in the registration modules.
POSITION_GRADIENT = "position_gradient"
HESSIAN = "hessian"  # (3 * n_atoms, n_atoms, 3)

KINDS = (
    GRAPH_SCALAR,
    GRAPH_VECTOR,
    GRAPH_TENSOR,
    GRAPH_ARRAY,
    PER_ATOM_SCALAR,
    PER_ATOM_VECTOR,
    PER_ATOM_TENSOR,
    PER_ATOM_MATRIX,
    PER_EDGE_VECTOR,
    POSITION_GRADIENT,
    HESSIAN,
)

#: Kinds whose leading axis is *not* the atom count. They exist because two
#: real outputs cannot be expressed by any per-atom kind, and a schema that
#: cannot express an output ends up dropping it:
#:
#:   * a per-edge quantity is indexed by the neighbour list, whose length
#:     depends on the cutoff -- a number this module cannot know, because it
#:     is a property of the model and this module never imports one. The
#:     leading extent is therefore recorded rather than predicted; a change
#:     in the edge count still fails a comparison, through the shape check.
#:   * a hessian is a square matrix over the 3N Cartesian degrees of freedom,
#:     stored as 3N rows of one gradient each.
_LEADING_AXIS_IS_FREE = (GRAPH_ARRAY, PER_EDGE_VECTOR)

#: Channels whose value is recorded for provenance but never asserted --
#: optimiser iterate counts, histories, anything that is an implementation
#: detail of how a fixed point was reached rather than the fixed point.
ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_METADATA = "metadata"
ROLES = (ROLE_INPUT, ROLE_OUTPUT, ROLE_METADATA)


# ---------------------------------------------------------------------------
# Surfaces
#
# A quantity reaches this module through one of two doors, and they do not
# use the same words. The model's ``forward`` returns one dict; the ase
# calculator that wraps it returns another, with four renames and -- worse --
# one *collision*: ``virials`` is the graph-level virial in the forward and
# the per-atom virial in the calculator's results. A single spelling->channel
# map cannot express that, and the version that tried resolved the model
# surface's ``virials`` onto a per-atom kind, which is a shape failure waiting
# for the first golden that takes the model route.
#
# So a registration may name the surface it describes. ``SURFACE_ANY`` is the
# default and means "both doors agree", which is the common case.
# ---------------------------------------------------------------------------

SURFACE_ANY = "any"
SURFACE_MODEL = "model"  # a forward() return dict, via golden_outputs()
SURFACE_CALCULATOR = "calculator"  # an ase calculator's results dict
SURFACES = (SURFACE_ANY, SURFACE_MODEL, SURFACE_CALCULATOR)


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str
    unit: str
    role: str = ROLE_OUTPUT


def _channel(name: str, kind: str, unit: str, role: str = ROLE_OUTPUT) -> Channel:
    assert kind in KINDS, kind
    assert role in ROLES, role
    return Channel(name=name, kind=kind, unit=unit, role=role)


CHANNELS: Dict[str, Channel] = {
    # --- energy family -----------------------------------------------------
    "energy": _channel("energy", GRAPH_SCALAR, "eV"),
    "free_energy": _channel("free_energy", GRAPH_SCALAR, "eV"),
    "node_energy": _channel("node_energy", PER_ATOM_SCALAR, "eV"),
    # Per-atom energies *including* the isolated-atom reference, as the ase
    # property of that name is defined. Distinct from node_energy, which has
    # the reference subtracted, and the pair is worth pinning separately:
    # their difference is exactly the E0 table.
    "energies": _channel("energies", PER_ATOM_SCALAR, "eV"),
    "forces": _channel("forces", PER_ATOM_VECTOR, "eV/Ang"),
    "edge_forces": _channel("edge_forces", PER_EDGE_VECTOR, "eV/Ang"),
    "stress": _channel("stress", GRAPH_TENSOR, "eV/Ang^3"),
    # Graph-level, as the model's forward returns it. The calculator writes
    # the same physics as Voigt-6 under the same name; that is a
    # representation change, handled by a calculator-surface conversion in
    # tests/golden/calculator_keys.py, not by a second channel.
    "virials": _channel("virials", GRAPH_TENSOR, "eV"),
    # The per-atom decompositions. Canonical layout is the full 3x3, which is
    # what the model emits (`atomic_stresses` / `atomic_virials`); the
    # calculator's `stresses` (Voigt-6) and `virials` (3x3, per atom) are
    # converted onto these on ingest. See calculator_keys.py for the
    # measurement that says the Voigt conversion is lossless here.
    "atomic_stresses": _channel("atomic_stresses", PER_ATOM_TENSOR, "eV/Ang^3"),
    "atomic_virials": _channel("atomic_virials", PER_ATOM_TENSOR, "eV"),
    "hessian": _channel("hessian", HESSIAN, "eV/Ang^2"),
    "interaction_energy": _channel("interaction_energy", GRAPH_SCALAR, "eV"),
    "electron_energy": _channel("electron_energy", GRAPH_SCALAR, "eV"),
    # Per-readout-layer energy terms, stacked per graph. The extent is a
    # property of the architecture -- how many readouts, and what is prepended
    # to them -- so it is recorded rather than predicted.
    "contributions": _channel("contributions", GRAPH_ARRAY, "eV"),
    # The concatenated latent node features the readouts see. Not a physical
    # observable, but it is the model's descriptor: anything reading features
    # rather than energies depends on exactly these numbers.
    "node_feats": _channel("node_feats", PER_ATOM_MATRIX, "arb"),
    # --- dipole / polarisability family -----------------------------------
    "dipole": _channel("dipole", GRAPH_VECTOR, "Debye"),
    # The per-atom terms the graph dipole is the sum of.
    "atomic_dipoles": _channel("atomic_dipoles", PER_ATOM_VECTOR, "Debye"),
    "polarizability": _channel("polarizability", GRAPH_TENSOR, "Debye*Ang/V"),
    # The same quantity in a spherical basis: six irreducible components
    # rather than a 3x3, so it is a graph array and not a graph tensor.
    "polarizability_sh": _channel("polarizability_sh", GRAPH_ARRAY, "Debye*Ang/V"),
    # Position derivatives of the two above, which is what a dielectric model
    # is trained against when infrared or Raman intensities are the target.
    "dmu_dr": _channel("dmu_dr", POSITION_GRADIENT, "Debye/Ang"),
    "dalpha_dr": _channel("dalpha_dr", POSITION_GRADIENT, "Debye/V"),
    "charges": _channel("charges", PER_ATOM_SCALAR, "e"),
    "spins": _channel("spins", PER_ATOM_SCALAR, "e"),
    "density_coefficients": _channel("density_coefficients", PER_ATOM_MATRIX, "arb"),
    "spin_charge_density": _channel("spin_charge_density", PER_ATOM_MATRIX, "arb"),
    # The alpha-minus-beta partner of density_coefficients, same layout: the
    # scalar `spins` channel above is only its leading column.
    "spin_density": _channel("spin_density", PER_ATOM_MATRIX, "arb"),
    # Normalised per graph, hence dimensionless.
    "fukui_functions": _channel("fukui_functions", PER_ATOM_MATRIX, "1"),
    # --- long-range electrostatics family ----------------------------------
    "les_energy": _channel("les_energy", GRAPH_SCALAR, "eV"),
    "electrostatic_energy": _channel("electrostatic_energy", GRAPH_SCALAR, "eV"),
    "latent_charges": _channel("latent_charges", PER_ATOM_SCALAR, "e"),
    "latent_dipoles": _channel("latent_dipoles", PER_ATOM_VECTOR, "e*Ang"),
    "latent_alphas": _channel("latent_alphas", PER_ATOM_MATRIX, "arb"),
    "latent_kappas": _channel("latent_kappas", PER_ATOM_MATRIX, "arb"),
    "latent_quads": _channel("latent_quads", PER_ATOM_MATRIX, "e*Ang^2"),
    "BEC": _channel("BEC", PER_ATOM_TENSOR, "e"),
    "electrostatic_potentials": _channel(
        "electrostatic_potentials", PER_ATOM_SCALAR, "V"
    ),
    # The chemical potential a fixed-point charge equilibration runs at,
    # carried as the zeroth component of the external potential.
    "fermi_level": _channel("fermi_level", GRAPH_SCALAR, "eV"),
    # --- magnetic family ----------------------------------------------------
    # magmom is an INPUT: the magnetic models consume per-atom moments and
    # differentiate the energy with respect to them, so a reference that did
    # not record the moments it was taken at would not be reproducible.
    "magmom": _channel("magmom", PER_ATOM_VECTOR, "muB", role=ROLE_INPUT),
    "magforces": _channel("magforces", PER_ATOM_VECTOR, "eV/muB"),
    "equilibrated_magmom": _channel("equilibrated_magmom", PER_ATOM_VECTOR, "muB"),
    # --- iteration histories --------------------------------------------------
    # All three are metadata, and for one reason: they describe *how* a fixed
    # point was reached, not the fixed point. Their extent is the iteration
    # count, so a converged run that takes one step fewer changes them without
    # changing a single physical number -- which is exactly the false failure
    # the metadata role exists to avoid. The fixed point itself is pinned:
    # `equilibrated_magmom` and `charges` are ordinary output channels.
    #
    # The three are one family and are declared together. Where each is
    # emitted, with file and line, is recorded in the registration modules.
    "scf_steps": _channel("scf_steps", GRAPH_SCALAR, "1", role=ROLE_METADATA),
    "scf_energy_history": _channel(
        "scf_energy_history", GRAPH_ARRAY, "eV", role=ROLE_METADATA
    ),
    # A charge-equilibration trajectory, whose trailing axis is the iteration
    # count.
    "charges_history": _channel(
        "charges_history", PER_ATOM_MATRIX, "e", role=ROLE_METADATA
    ),
    # --- graph-level inputs --------------------------------------------------
    "total_charge": _channel("total_charge", GRAPH_SCALAR, "e", role=ROLE_INPUT),
    "total_spin": _channel("total_spin", GRAPH_SCALAR, "1", role=ROLE_INPUT),
    "elec_temp": _channel("elec_temp", GRAPH_SCALAR, "K", role=ROLE_INPUT),
    "external_field": _channel("external_field", GRAPH_VECTOR, "V/Ang", ROLE_INPUT),
    # --- per-atom inputs -------------------------------------------------------
    # The reference charges a structure carries and the model reads
    # (`AtomicData.charges`, filled from `arrays_keys["charges"]`); the
    # fixed-charge dipole baseline is computed from them. Deliberately not the
    # `charges` channel above: that one is what a model *predicts*, and the
    # two are different numbers that happen to describe the same observable.
    "input_charges": _channel("input_charges", PER_ATOM_SCALAR, "e", ROLE_INPUT),
}


@dataclass(frozen=True)
class Alias:
    """One spelling, on one surface, and the channel it lands in.

    ``convert`` is what makes an alias able to cross a *representation*
    change and not only a rename. Without it there are exactly two options
    for a quantity the two surfaces store differently, and both are wrong: a
    plain alias fails on the shape, and two separate channels mean a golden
    taken through the model route and one taken through the calculator route
    never compare -- the split this whole registry exists to prevent. With
    it, both surfaces land on the same channel holding the same layout, so
    there is only ever one thing to compare.

    A converter must be lossless in the direction it is applied, and the
    registration is expected to say how that was measured, not assert it.
    """

    spelling: str
    channel: str
    surface: str = SURFACE_ANY
    convert: Optional[Callable[[np.ndarray], np.ndarray]] = None
    note: str = ""


#: Alternative spellings for a declared channel, as
#: ``spelling -> {surface: Alias}``. A model's forward and the calculator that
#: wraps it do not have to agree on a name, and when they disagree the
#: registry can only know one of the two. Every extra spelling resolves here
#: to the one channel, so a golden taken through either path records the same
#: key -- rather than recording nothing, which is how a reference ends up
#: claiming to pin a family it never saw.
#:
#: The harness declares none itself: which name a given implementation writes
#: is knowledge about that implementation, and this module has none. The
#: spellings this repository uses are registered in
#: ``tests/golden/calculator_keys.py`` and ``tests/golden/model_keys.py``,
#: which the package imports for their side effect, so any consumer of the
#: harness has them.
CHANNEL_ALIASES: Dict[str, Dict[str, Alias]] = {}

#: Keys an evaluation may return that are deliberately not snapshotted, each
#: with the reason it is not. There is no blanket "unknown key" escape: a key
#: that is neither declared, aliased nor listed here is an error, because
#: silently discarding an output the schema does not know is precisely the
#: regression a golden exists to catch.
IGNORED_KEYS: Dict[str, str] = {}


def register_alias(
    spelling: str,
    channel: str,
    *,
    surface: str = SURFACE_ANY,
    convert: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    note: str = "",
) -> None:
    """Map an alternative output name onto a declared channel.

    Args:
        spelling: the name the implementation writes.
        channel: the declared channel it means.
        surface: ``SURFACE_ANY`` when both doors use this spelling for this
            quantity; ``SURFACE_MODEL`` / ``SURFACE_CALCULATOR`` when only one
            does. A surface-scoped registration is the only way to say that a
            spelling means different things on the two surfaces.
        convert: applied to the value on ingest, to bring it into the
            channel's canonical layout. Must be lossless.
        note: mandatory when the spelling shadows a declared channel, because
            that is a collision and a collision has to be stated.
    """
    if surface not in SURFACES:
        raise ValueError(f"unknown surface {surface!r}; the surfaces are {SURFACES}")
    if channel not in CHANNELS:
        raise KeyError(
            f"cannot alias {spelling!r} onto {channel!r}: no such channel. "
            f"Declare it with register_channel() first."
        )
    if spelling in CHANNELS:
        if surface == SURFACE_ANY:
            raise ValueError(
                f"{spelling!r} is itself a declared channel and cannot also be "
                f"an alias for {channel!r}"
            )
        if not note.strip():
            raise ValueError(
                f"{spelling!r} is a declared channel and the {surface} surface "
                f"uses it for {channel!r} instead. That is a name collision "
                f"between the two surfaces, not a rename; say so in note=."
            )
    by_surface = CHANNEL_ALIASES.setdefault(spelling, {})
    existing = by_surface.get(surface)
    if existing is not None and existing.channel != channel:
        raise ValueError(
            f"{spelling!r} already resolves to {existing.channel!r} on the "
            f"{surface} surface and cannot also resolve to {channel!r}"
        )
    by_surface[surface] = Alias(
        spelling=spelling,
        channel=channel,
        surface=surface,
        convert=convert,
        note=note,
    )


def ignore_key(key: str, reason: str) -> None:
    """Declare that ``key`` is intentionally absent from every snapshot.

    ``reason`` is mandatory and is printed by the tooling that lists the
    allowlist, so that "we never pinned this" stays a decision on the record
    rather than an omission nobody can date.
    """
    if not reason.strip():
        raise ValueError(f"ignoring {key!r} requires a reason")
    if key in CHANNELS or key in CHANNEL_ALIASES:
        raise ValueError(f"{key!r} is a declared channel; it cannot be ignored")
    IGNORED_KEYS[key] = reason


def _lookup_alias(key: str, surface: str) -> Optional[Alias]:
    """The alias that governs ``key`` on ``surface``, if any.

    A surface-scoped registration beats the surface-agnostic one, which is
    what lets ``virials`` be the graph virial everywhere except through an
    ase calculator, where it is the per-atom one.
    """
    by_surface = CHANNEL_ALIASES.get(key)
    if not by_surface:
        return None
    if surface != SURFACE_ANY and surface in by_surface:
        return by_surface[surface]
    if SURFACE_ANY in by_surface:
        return by_surface[SURFACE_ANY]
    if surface == SURFACE_ANY and key not in CHANNELS and len(by_surface) == 1:
        # Only one surface writes this spelling at all, and it is not a
        # declared channel, so there is nothing it could be confused with.
        return next(iter(by_surface.values()))
    return None


def resolve_channel(key: str, surface: str = SURFACE_ANY) -> Optional[str]:
    """The channel ``key`` names on ``surface``, or ``None`` if allowlisted.

    Raises:
        KeyError: if the key is neither declared, aliased nor ignored. This
            is the harness refusing to drop an output it does not recognise.
    """
    alias = _lookup_alias(key, surface)
    if alias is not None:
        return alias.channel
    if key in CHANNELS:
        return key
    if key in IGNORED_KEYS:
        return None
    raise KeyError(
        f"the evaluation produced {key!r} on the {surface!r} surface, which "
        f"the schema does not know. An output that is silently dropped is an "
        f"output nothing pins, so this is an error rather than a skip. Fix it "
        f"in one of three ways: declare it with register_channel({key!r}, "
        f"<kind>, <unit>) if it is a new quantity; register_alias({key!r}, "
        f"<channel>) if it is another spelling of a channel that already "
        f"exists (this repository's spellings live in "
        f"tests/golden/calculator_keys.py and tests/golden/model_keys.py); or "
        f"ignore_key({key!r}, <reason>) if it genuinely must not be pinned. "
        f"Declared channels: {sorted(CHANNELS)}."
    )


def channel_conversion(
    key: str, surface: str = SURFACE_ANY
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """The layout conversion ``key`` needs on ``surface``, if any."""
    alias = _lookup_alias(key, surface)
    return None if alias is None else alias.convert


def voigt_6_to_full_3x3(voigt: Any) -> np.ndarray:
    """Expand a Voigt-6 tensor (or a stack of them) to full 3x3 form.

    Vectorised over any number of leading axes, so it serves both the
    graph-level stress ``(6,)`` and the per-atom stress ``(n_atoms, 6)``.

    Expansion is exact -- it re-materialises the off-diagonals a symmetric
    tensor already determines. The lossy step is the *other* direction, and
    whether this repository ever takes it lossily is a measurement, recorded
    where the conversion is registered.
    """
    arr = np.asarray(voigt, dtype=np.float64)
    if arr.shape[-1] != 6:
        raise ValueError(f"a Voigt-6 tensor has 6 components, got shape {arr.shape}")
    xx, yy, zz, yz, xz, xy = (arr[..., i] for i in range(6))
    return np.stack(
        [
            np.stack([xx, xy, xz], axis=-1),
            np.stack([xy, yy, yz], axis=-1),
            np.stack([xz, yz, zz], axis=-1),
        ],
        axis=-2,
    )


def register_channel(
    name: str, kind: str, unit: str, role: str = ROLE_OUTPUT
) -> Channel:
    """Declare a channel the shared registry does not know yet.

    Later golden work adds observables; each one has to say what shape and
    unit it is, once, here, rather than being inferred per test. Re-declaring
    an existing channel identically is a no-op; re-declaring it differently
    is an error, because two callers disagreeing about a channel's unit is
    the failure this registry exists to prevent.
    """
    new = _channel(name, kind, unit, role)
    existing = CHANNELS.get(name)
    if existing is not None and existing != new:
        raise ValueError(
            f"channel {name!r} is already registered as {existing} and cannot "
            f"be redefined as {new}"
        )
    CHANNELS[name] = new
    return new


def expected_shape(kind: str, n_atoms: int) -> Optional[tuple]:
    """The array shape a kind implies, or ``None`` when it is only partly
    constrained (``per_atom_matrix`` fixes the leading axis only;
    ``graph_array`` and ``per_edge_vector`` do not fix it at all)."""
    return {
        GRAPH_SCALAR: (),
        GRAPH_VECTOR: (3,),
        GRAPH_TENSOR: (3, 3),
        GRAPH_ARRAY: None,
        PER_ATOM_SCALAR: (n_atoms,),
        PER_ATOM_VECTOR: (n_atoms, 3),
        PER_ATOM_TENSOR: (n_atoms, 3, 3),
        PER_ATOM_MATRIX: None,
        PER_EDGE_VECTOR: None,
        POSITION_GRADIENT: None,
        HESSIAN: (3 * n_atoms, n_atoms, 3),
    }[kind]


def _check_shape(name: str, channel: Channel, arr: np.ndarray, n_atoms: int) -> None:
    """Reject an array that does not match the shape its kind promises."""
    want = expected_shape(channel.kind, n_atoms)
    if want is not None:
        if arr.shape != want:
            raise ValueError(
                f"channel {name!r} is declared {channel.kind} and should have "
                f"shape {want} for {n_atoms} atoms, got {arr.shape}"
            )
        return
    if channel.kind == PER_ATOM_MATRIX:
        if arr.ndim < 1 or arr.shape[0] != n_atoms:
            raise ValueError(
                f"channel {name!r} is declared {channel.kind} and should have "
                f"{n_atoms} leading rows, got shape {arr.shape}"
            )
        return
    if channel.kind == PER_EDGE_VECTOR:
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"channel {name!r} is declared {channel.kind} and should have "
                f"shape (n_edges, 3), got {arr.shape}"
            )
        return
    if channel.kind == POSITION_GRADIENT:
        if arr.ndim != 3 or arr.shape[1:] != (n_atoms, 3):
            raise ValueError(
                f"channel {name!r} is declared {channel.kind} and should have "
                f"shape (k, {n_atoms}, 3), got {arr.shape}"
            )
        return
    if channel.kind == GRAPH_ARRAY:
        if arr.ndim < 1:
            raise ValueError(
                f"channel {name!r} is declared {channel.kind} and should be an "
                f"array, got a scalar"
            )
        return
    raise AssertionError(f"kind {channel.kind!r} has no shape rule")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def load_manifest() -> Dict[str, dict]:
    """The fixture manifest: name -> {file, description, regime, tags, ...}."""
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)["fixtures"]


def load_fixtures(
    names: Optional[Sequence[str]] = None,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Atoms]:
    """Load the committed evaluation structures.

    Args:
        names: restrict to these fixture names, in this order.
        tags: keep only fixtures carrying every one of these manifest tags
            (``"periodic"``, ``"aperiodic"``, ``"molecular"``, ...). This is
            how a model family that only applies to part of the set -- an
            organic-chemistry model, say -- selects its subset without
            hard-coding names.

    Returns:
        An ordered mapping name -> ``ase.Atoms``. The returned objects are
        fresh copies; a caller may attach a calculator to them freely.
    """
    manifest = load_manifest()
    if names is None:
        names = list(manifest)
    else:
        unknown = [name for name in names if name not in manifest]
        if unknown:
            raise KeyError(
                f"unknown fixture(s) {unknown}; the manifest has {sorted(manifest)}"
            )
    wanted = set(tags or ())
    out: Dict[str, Atoms] = {}
    for name in names:
        entry = manifest[name]
        if wanted and not wanted.issubset(set(entry.get("tags", []))):
            continue
        atoms = ase_read(FIXTURES_DIR / entry["file"], index=0, format="extxyz")
        atoms.info["golden_name"] = name
        out[name] = atoms
    return out


def is_periodic(atoms: Atoms) -> bool:
    """Whether a stress is meaningful for this structure.

    Any periodic axis counts: a slab has a real volume along its periodic
    directions and the neighbour-list layer guarantees a non-degenerate cell
    even when the vacuum row is all zeros, so the stress is defined.
    """
    return bool(np.asarray(atoms.pbc).any())


# ---------------------------------------------------------------------------
# Snapshotting
# ---------------------------------------------------------------------------


def _as_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(f"cannot snapshot a ragged/object value: {value!r}")
    return arr.astype(np.float64) if arr.dtype.kind in "fiub" else arr


def _evaluate(calc_like: Any, atoms: Atoms) -> Tuple[Dict[str, Any], str]:
    """Run one structure through whatever the caller handed us.

    Three shapes are accepted, in this order:

    1. anything exposing ``golden_outputs(atoms) -> dict`` -- the explicit
       protocol, and the one a direct model wrapper should implement when the
       quantities it wants pinned are not reachable through a calculator;
    2. an ase calculator (``get_potential_energy`` and friends), whose
       ``results`` dict is scraped for any further known channels;
    3. a plain callable ``atoms -> dict``.

    Returns the raw dict and the surface it came off, because the two
    surfaces do not use the same words and one spelling (``virials``) means
    different things on each. An object may override the guess by setting a
    ``golden_surface`` attribute -- a callable that internally drives an ase
    calculator and hands back its results verbatim is a calculator surface,
    and only it knows that.
    """
    declared = getattr(calc_like, "golden_surface", None)
    if declared is not None and declared not in SURFACES:
        raise ValueError(
            f"golden_surface={declared!r} is not one of {SURFACES}"
        )

    hook = getattr(calc_like, "golden_outputs", None)
    if callable(hook):
        return dict(hook(atoms)), declared or SURFACE_MODEL

    if hasattr(calc_like, "get_potential_energy"):
        probe = atoms.copy()
        probe.calc = calc_like
        results: Dict[str, Any] = {
            "energy": float(probe.get_potential_energy()),
            "forces": np.asarray(probe.get_forces(), dtype=np.float64),
        }
        if is_periodic(probe):
            # Called for its effect, not its value: it is what forces the
            # stress to be computed and left in `results`. The Voigt-6 layout
            # ase stores it in is turned into the channel's 3x3 by the
            # registered calculator-surface conversion, not by hand here --
            # one representation change, one place.
            probe.get_stress(voigt=True)
        raw = getattr(calc_like, "results", None) or {}
        for name, value in raw.items():
            # Everything the calculator left behind is handed on, known or
            # not: deciding what the schema covers is the schema's job, and
            # a filter here would make an undeclared output disappear before
            # anything could complain about it. The one exception is a key
            # the ase accessors above already produced, because those are in
            # this module's own convention.
            if name in results:
                continue
            results[name] = value
        return results, declared or SURFACE_CALCULATOR

    if callable(calc_like):
        return dict(calc_like(atoms)), declared or SURFACE_MODEL

    raise TypeError(
        "snapshot_outputs needs an object with golden_outputs(atoms), an ase "
        f"calculator, or a callable; got {type(calc_like).__name__}"
    )


def _encode(name: str, value: Any, n_atoms: int) -> dict:
    channel = CHANNELS.get(name)
    if channel is None:
        raise KeyError(
            f"channel {name!r} is not in the schema; declare it with "
            f"register_channel() so its kind and unit are recorded once"
        )
    arr = _as_array(value)
    _check_shape(name, channel, arr, n_atoms)
    return {
        "kind": channel.kind,
        "unit": channel.unit,
        "shape": list(arr.shape),
        "value": arr.tolist(),
    }


#: Where each declared input actually lives on an ``ase.Atoms``, in the order
#: the spellings are tried. This has to track the reader, not the convention:
#: a model that takes its moments from ``atoms.arrays["REF_magmom"]`` is not
#: fed by ``set_initial_magnetic_moments``, so recording the latter would put
#: provenance in the reference that the evaluation never saw -- a reference
#: that looks reproducible and is not.
#:
#: These are only the *defaults*. Both of them are constructor arguments on
#: the calculators (``magmom_key``, ``charges_key``), so a table of literals
#: is a table of guesses: a structure whose moments live under a
#: non-default key matches nothing here, the snapshot records no magmom at
#: all, and the comparison passes -- silently, since there is nothing on
#: either side to disagree about. The configured value is recovered per
#: evaluation through :func:`register_input_probe`.
INPUT_ARRAY_KEYS: Dict[str, tuple] = {
    "magmom": ("REF_magmom",),
    "input_charges": ("REF_charges",),
}

#: The same, for graph-level inputs read off ``atoms.info``. Two spellings
#: are live for charge and spin -- the long one the data pipeline writes and
#: the short one the calculators default to -- so both are accepted, and
#: disagreement between them is an error rather than a coin toss.
INPUT_INFO_KEYS: Dict[str, tuple] = {
    "total_charge": ("total_charge", "charge"),
    "total_spin": ("total_spin", "spin"),
    "elec_temp": ("elec_temp",),
    "external_field": ("external_field",),
}


def register_input_source(
    channel: str,
    *,
    arrays: Sequence[str] = (),
    info: Sequence[str] = (),
) -> None:
    """Teach the harness where an input channel is read from."""
    if CHANNELS.get(channel) is None or CHANNELS[channel].role != ROLE_INPUT:
        raise KeyError(f"{channel!r} is not a declared input channel")
    if arrays:
        INPUT_ARRAY_KEYS[channel] = tuple(
            dict.fromkeys(INPUT_ARRAY_KEYS.get(channel, ()) + tuple(arrays))
        )
    if info:
        INPUT_INFO_KEYS[channel] = tuple(
            dict.fromkeys(INPUT_INFO_KEYS.get(channel, ()) + tuple(info))
        )


@dataclass(frozen=True)
class InputProbe:
    """Where an evaluation *says* it reads one of its inputs from.

    The spellings in :data:`INPUT_ARRAY_KEYS` are defaults, and a default is
    not what the evaluation used. Calculators in this repository take those
    names as constructor arguments and keep them on the instance, so the
    harness can ask the object it was handed rather than assume. That turns
    "the moments were somewhere else and nothing was recorded" from a silent
    pass into a recorded input. The attribute names are registered by the
    surface modules, not known here.
    """

    channel: str
    attribute: str
    store: str  # "arrays" or "info"


INPUT_PROBES: Dict[str, Tuple[InputProbe, ...]] = {}


def register_input_probe(channel: str, *, attribute: str, store: str) -> None:
    """Declare that ``calc_like.<attribute>`` names where ``channel`` lives."""
    if CHANNELS.get(channel) is None or CHANNELS[channel].role != ROLE_INPUT:
        raise KeyError(f"{channel!r} is not a declared input channel")
    if store not in ("arrays", "info"):
        raise ValueError(f"store must be 'arrays' or 'info', got {store!r}")
    probe = InputProbe(channel=channel, attribute=attribute, store=store)
    INPUT_PROBES[channel] = tuple(
        dict.fromkeys(INPUT_PROBES.get(channel, ()) + (probe,))
    )


def _effective_input_keys(
    calc_like: Any, overrides: Optional[Mapping[str, Sequence[str]]]
) -> Tuple[Dict[str, tuple], Dict[str, tuple]]:
    """The array/info spellings to read each input from, for this evaluation.

    A configured key **replaces** the defaults for its channel rather than
    joining them. The two are not synonyms: the entries in
    :data:`INPUT_ARRAY_KEYS` are alternative spellings of one input, and
    disagreement between them is an error (:func:`_first_present`), whereas
    an evaluation that says it reads its moments from ``spins_from_dft`` has
    settled the question -- a stale ``REF_magmom`` sitting on the same
    structure is not a second opinion, it is an array the evaluation ignored.
    Merging the two would turn that into a spurious "present under both with
    different values" failure.

    ``overrides`` wins over the probe, which wins over the defaults.
    """
    arrays = {name: tuple(keys) for name, keys in INPUT_ARRAY_KEYS.items()}
    info = {name: tuple(keys) for name, keys in INPUT_INFO_KEYS.items()}

    for probes in INPUT_PROBES.values():
        for probe in probes:
            configured = getattr(calc_like, probe.attribute, None)
            if isinstance(configured, str) and configured:
                target = arrays if probe.store == "arrays" else info
                target[probe.channel] = (configured,)
    for channel, keys in (overrides or {}).items():
        if CHANNELS.get(channel) is None or CHANNELS[channel].role != ROLE_INPUT:
            raise KeyError(
                f"input_keys names {channel!r}, which is not a declared input "
                f"channel; the inputs are "
                f"{sorted(n for n, c in CHANNELS.items() if c.role == ROLE_INPUT)}"
            )
        target = info if channel in INPUT_INFO_KEYS else arrays
        target[channel] = tuple(dict.fromkeys(keys))
    return arrays, info


def _first_present(channel: str, store: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """The value of the first spelling present, refusing a disagreement."""
    found = [key for key in keys if key in store]
    if not found:
        return None
    values = [np.asarray(store[key], dtype=np.float64) for key in found]
    for key, value in zip(found[1:], values[1:]):
        if value.shape != values[0].shape or not np.array_equal(value, values[0]):
            raise ValueError(
                f"input {channel!r} is present under both {found[0]!r} and "
                f"{key!r} with different values; the evaluation reads one of "
                f"them and the reference would record the other"
            )
    return values[0]


def _fixture_inputs(
    atoms: Atoms,
    array_keys: Mapping[str, tuple],
    info_keys: Mapping[str, tuple],
) -> Dict[str, dict]:
    """Record the non-geometric inputs a structure carries.

    Geometry lives in the committed .xyz; what has to travel with the
    reference is everything else a model reads off the structure, because a
    snapshot taken at different initial moments (or a different total charge)
    is a different number and would otherwise look like a regression.
    """
    n_atoms = len(atoms)
    inputs: Dict[str, dict] = {}
    for channel, keys in sorted(array_keys.items()):
        value = _first_present(channel, atoms.arrays, keys)
        if value is None:
            continue
        if channel == "magmom" and value.ndim == 1:
            # A collinear moment is stored along z, so the schema stays
            # vector-valued rather than being widened later.
            value = np.stack([np.zeros(n_atoms), np.zeros(n_atoms), value], axis=-1)
        inputs[channel] = _encode(channel, value, n_atoms)
    if "magmom" not in inputs:
        stray = np.asarray(atoms.get_initial_magnetic_moments(), dtype=np.float64)
        if stray.any():
            raise ValueError(
                "this structure carries ase initial magnetic moments but none "
                f"of the arrays {list(array_keys.get('magmom', ()))} the "
                "evaluation reads. The magnetic models read an array, not the "
                "ase attribute, so a reference taken here would record moments "
                "the evaluation never used. Put the moments in the array the "
                "model reads, or clear them."
            )
    for channel, keys in sorted(info_keys.items()):
        value = _first_present(channel, atoms.info, keys)
        if value is None:
            continue
        inputs[channel] = _encode(channel, value, n_atoms)
    return inputs


def snapshot_outputs(
    calc_like: Any,
    fixtures: Mapping[str, Atoms],
    *,
    dtype: str = "float64",
    device: str = "cpu",
    backend: str = "e3nn",
    channels: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    input_keys: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict:
    """Evaluate ``calc_like`` over ``fixtures`` and return a snapshot dict.

    Args:
        calc_like: see :func:`_evaluate` for the three accepted shapes.
        fixtures: name -> structure, as returned by :func:`load_fixtures`.
        dtype / device / backend: recorded verbatim in the snapshot. They are
            not inspected -- the caller is the only one who knows what it
            configured -- but a reference without them cannot be reproduced.
        channels: restrict the recorded output channels to these names. The
            default records every channel the evaluation returned that the
            schema knows, minus ``stress`` on aperiodic structures.
        metadata: free-form extra fields stored alongside the snapshot.
        input_keys: channel -> the array/info spellings this evaluation reads
            its inputs from, tried ahead of anything else. Needed only when
            the evaluated object does not expose them itself; the calculators
            do, through the probes in ``tests/golden/calculator_keys.py``.
    """
    wanted = (
        {resolve_channel(name) for name in channels} if channels is not None else None
    )
    if wanted is not None and None in wanted:
        raise KeyError(
            "channels= names an allowlisted key, which is by definition never "
            f"recorded: {sorted(set(channels) & set(IGNORED_KEYS))}"
        )
    array_keys, info_keys = _effective_input_keys(calc_like, input_keys)
    out_fixtures: Dict[str, dict] = {}
    for name, atoms in fixtures.items():
        n_atoms = len(atoms)
        periodic = is_periodic(atoms)
        raw, surface = _evaluate(calc_like, atoms)
        outputs: Dict[str, dict] = {}
        meta: Dict[str, Any] = {}
        for key in sorted(raw):
            # Unknown keys raise here rather than being skipped; see
            # resolve_channel for why, and for the three ways to fix it.
            channel = resolve_channel(key, surface)
            if channel is None:
                continue
            if raw[key] is None:
                # A forward returns its whole key set every call and fills in
                # None for anything this call did not ask for. Resolution has
                # already happened, so an unknown key still raises; what is
                # skipped here is a known channel with no value, and a
                # reference that pins it will fail on "vanished" rather than
                # on a confusing dtype error.
                continue
            if wanted is not None and channel not in wanted:
                continue
            if channel == "stress" and not periodic:
                continue
            if CHANNELS[channel].role == ROLE_INPUT:
                # Recorded from the structure by _fixture_inputs, which reads
                # the same place the model does; an echo in the results dict
                # would be the value after the reader, not the input.
                continue
            if CHANNELS[channel].role == ROLE_METADATA:
                meta[channel] = np.asarray(raw[key]).tolist()
                continue
            convert = channel_conversion(key, surface)
            value = raw[key] if convert is None else convert(_as_array(raw[key]))
            outputs[channel] = _encode(channel, value, n_atoms)
        if wanted is not None:
            missing = sorted(
                wanted
                - set(outputs)
                - set(meta)
                - ({"stress"} if not periodic else set())
            )
            if missing:
                raise KeyError(
                    f"fixture {name!r}: requested channel(s) {missing} were not "
                    f"produced; the evaluation returned {sorted(raw)}"
                )
        out_fixtures[name] = {
            "n_atoms": n_atoms,
            "periodic": periodic,
            "formula": atoms.get_chemical_formula(),
            "inputs": _fixture_inputs(atoms, array_keys, info_keys),
            "outputs": outputs,
            "metadata": meta,
        }
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "dtype": dtype,
        "device": device,
        "backend": backend,
        "units": {"length": "Ang", "energy": "eV"},
        "fixtures": out_fixtures,
    }
    if metadata:
        snapshot["metadata"] = dict(metadata)
    return snapshot


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def write_reference(
    path: Path,
    snapshot: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a snapshot as a committed reference.

    Refuses to overwrite an existing file unless told to. Goldens change only
    in dedicated, reviewed changes; the regeneration script is the only
    caller that passes ``allow_overwrite=True``, and it in turn refuses to
    run without an explicit acknowledgement flag.
    """
    path = Path(path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"{path} already exists. A committed golden is only rewritten by "
            f"tests/golden/regenerate.py, in its own reviewed change."
        )
    required = {"source", "recipe", "description"}
    missing = required - set(provenance)
    if missing:
        raise ValueError(
            f"provenance is missing {sorted(missing)}; a reference nobody can "
            f"regenerate is not a reference"
        )
    payload = dict(snapshot)
    payload["provenance"] = dict(provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_reference(path: Path) -> dict:
    """Read a committed reference and check its schema version."""
    with Path(path).open(encoding="utf-8") as handle:
        reference = json.load(handle)
    version = reference.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written with schema version {version}, this harness "
            f"speaks {SCHEMA_VERSION}; regenerate it rather than comparing "
            f"two different layouts"
        )
    return reference


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _values(entry: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(entry["value"], dtype=np.float64).reshape(entry["shape"])


def _compare_entry(
    label: str,
    got_entry: Mapping[str, Any],
    ref_entry: Mapping[str, Any],
    tol: Tolerance,
    problems: list,
) -> None:
    """Compare one encoded channel, appending any complaint to ``problems``."""
    if got_entry["unit"] != ref_entry["unit"]:
        problems.append(
            f"{label}: unit changed {ref_entry['unit']!r} -> {got_entry['unit']!r}"
        )
        return
    if list(got_entry["shape"]) != list(ref_entry["shape"]):
        problems.append(
            f"{label}: shape changed {ref_entry['shape']} -> {got_entry['shape']}"
        )
        return
    got = _values(got_entry)
    want = _values(ref_entry)
    bound = tol.atol + tol.rtol * np.abs(want)
    diff = np.abs(got - want)
    bad = diff > bound
    if not bad.any():
        return
    idx = np.unravel_index(int(np.argmax(diff - bound)), diff.shape)
    problems.append(
        f"{label} [{ref_entry['unit']}]: "
        f"{int(bad.sum())}/{diff.size} element(s) outside the "
        f"'{tol.name}' row (atol={tol.atol:g}, rtol={tol.rtol:g}); "
        f"worst at index {tuple(int(i) for i in idx)}: "
        f"got {float(got[idx]):.12g}, reference {float(want[idx]):.12g}, "
        f"|diff| {float(diff[idx]):.3g} > {float(bound[idx]):.3g}"
    )


def _compare_inputs(
    name: str,
    got_fix: Mapping[str, Any],
    ref_fix: Mapping[str, Any],
    problems: list,
) -> None:
    """Compare the recorded inputs, in both directions and unconditionally.

    Outputs are compared one way by default -- a model gaining an observable
    is not a regression of the ones already pinned. Inputs are not like that.
    A snapshot taken at moments, a total charge or a field the reference was
    not taken at is a different measurement wearing the reference's name, and
    an input the reference records but the snapshot does not means the
    evaluation was fed nothing where it used to be fed something. Both
    directions are therefore failures, and there is no flag to turn this off:
    the schema calls magmom a pinned input, and a comparison that ignored the
    block would make that claim decoration.

    They are compared at :data:`EXACT`, never at the row the outputs use.
    An input is read verbatim off the committed fixture rather than computed,
    so the numerical-agreement argument behind the output rows does not apply
    to it -- and applying one anyway means a perturbation below the output
    tolerance is invisible. A 2e-3 nudge to a magnetic moment sits inside the
    fp32 row, so the snapshot would pass while having been taken at moments
    the reference was not.
    """
    ref_in = ref_fix.get("inputs", {})
    got_in = got_fix.get("inputs", {})
    for channel in sorted(set(ref_in) | set(got_in)):
        if channel not in got_in:
            problems.append(
                f"{name}: input {channel!r} vanished; the reference was taken "
                f"with it and this snapshot was not"
            )
            continue
        if channel not in ref_in:
            problems.append(
                f"{name}: input {channel!r} appeared; this snapshot was fed "
                f"something the reference was not taken with"
            )
            continue
        _compare_entry(
            f"{name}/inputs/{channel}",
            got_in[channel],
            ref_in[channel],
            EXACT,
            problems,
        )


@dataclass
class Deviation:
    fixture: str
    channel: str
    max_abs: float
    max_rel: float
    worst_index: tuple
    got: float
    want: float


def deviations(
    snapshot: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    channels: Optional[Sequence[str]] = None,
) -> list:
    """Per-channel worst deviation between a snapshot and a reference.

    Separate from :func:`compare_to_reference` on purpose: measuring how much
    headroom a tolerance row actually has is a thing to do deliberately, and
    it must not be reachable by loosening an assertion.
    """
    wanted = set(channels) if channels is not None else None
    report = []
    for name, ref_fix in reference["fixtures"].items():
        got_fix = snapshot["fixtures"][name]
        for channel, ref_entry in ref_fix["outputs"].items():
            if wanted is not None and channel not in wanted:
                continue
            got = _values(got_fix["outputs"][channel])
            want = _values(ref_entry)
            diff = np.abs(got - want)
            scale = np.abs(want)
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape) if diff.size else ()
            report.append(
                Deviation(
                    fixture=name,
                    channel=channel,
                    max_abs=float(diff.max()) if diff.size else 0.0,
                    max_rel=(
                        float((diff / np.where(scale > 0, scale, np.inf)).max())
                        if diff.size
                        else 0.0
                    ),
                    worst_index=tuple(int(i) for i in idx),
                    got=float(got[idx]) if diff.size else 0.0,
                    want=float(want[idx]) if diff.size else 0.0,
                )
            )
    return report


def compare_to_reference(
    snapshot: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    row: str,
    channels: Optional[Sequence[str]] = None,
    strict_channels: bool = False,
) -> None:
    """Assert a fresh snapshot reproduces a committed reference.

    Args:
        snapshot: as returned by :func:`snapshot_outputs`.
        reference: as returned by :func:`load_reference`.
        row: the name of a row in :data:`TOLERANCES`.
        channels: compare only these channels (the default compares every
            channel the reference carries).
        strict_channels: also fail when the snapshot emits an output channel
            the reference does not carry. Off by default -- a model gaining
            an observable is not a regression of the ones already pinned --
            but the reverse is always a failure.

    Raises:
        AssertionError: naming the fixture, the channel, the element and both
            values, so the failure says what moved rather than that something
            did.
    """
    tol = tolerance(row)
    if snapshot.get("schema_version") != reference.get("schema_version"):
        raise AssertionError(
            f"schema version mismatch: snapshot "
            f"{snapshot.get('schema_version')} vs reference "
            f"{reference.get('schema_version')}"
        )
    missing_fixtures = sorted(set(reference["fixtures"]) - set(snapshot["fixtures"]))
    if missing_fixtures:
        raise AssertionError(
            f"the snapshot is missing fixture(s) {missing_fixtures} that the "
            f"reference pins"
        )
    problems = []
    wanted = set(channels) if channels is not None else None
    for name, ref_fix in reference["fixtures"].items():
        got_fix = snapshot["fixtures"][name]
        ref_out = ref_fix["outputs"]
        got_out = got_fix["outputs"]
        absent = sorted(
            (set(ref_out) if wanted is None else set(ref_out) & wanted) - set(got_out)
        )
        if absent:
            problems.append(f"{name}: channel(s) {absent} vanished from the output")
        if strict_channels:
            extra = sorted(set(got_out) - set(ref_out))
            if extra:
                problems.append(f"{name}: unpinned new channel(s) {extra}")
        _compare_inputs(name, got_fix, ref_fix, problems)
        for channel, ref_entry in sorted(ref_out.items()):
            if wanted is not None and channel not in wanted:
                continue
            if channel not in got_out:
                continue
            _compare_entry(
                f"{name}/{channel}", got_out[channel], ref_entry, tol, problems
            )
    if problems:
        raise AssertionError(
            "golden comparison failed against "
            f"{reference.get('provenance', {}).get('source', '<unknown source>')}:\n  "
            + "\n  ".join(problems)
            + f"\n\nTolerance row '{tol.name}': {tol.rationale}\n"
            "Goldens are edit-locked: a real physics change is regenerated in "
            "its own reviewed change, never by widening a tolerance here."
        )


__all__ = [
    "CHANNELS",
    "CHANNEL_ALIASES",
    "EXACT",
    "IGNORED_KEYS",
    "INPUT_ARRAY_KEYS",
    "INPUT_INFO_KEYS",
    "INPUT_PROBES",
    "FIXTURES_DIR",
    "FP32",
    "FP64_ACCELERATED_BACKEND",
    "FP64_CPU_REFERENCE",
    "GOLDEN_ROOT",
    "MODELS_DIR",
    "REFERENCES_DIR",
    "SCHEMA_VERSION",
    "SURFACES",
    "SURFACE_ANY",
    "SURFACE_CALCULATOR",
    "SURFACE_MODEL",
    "TOLERANCES",
    "Alias",
    "Channel",
    "Deviation",
    "InputProbe",
    "Tolerance",
    "channel_conversion",
    "compare_to_reference",
    "deviations",
    "expected_shape",
    "ignore_key",
    "is_periodic",
    "load_fixtures",
    "load_manifest",
    "load_reference",
    "register_alias",
    "register_channel",
    "register_input_probe",
    "register_input_source",
    "resolve_channel",
    "snapshot_outputs",
    "tolerance",
    "voigt_6_to_full_3x3",
    "write_reference",
]
