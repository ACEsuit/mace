"""The spellings the evaluation command line writes onto its structures.

The third door, and the one that had none of this. ``mace_eval_configs``
(``mace/cli/eval_configs.py``) does not return a dict: it writes its results
back onto the ``ase.Atoms`` it was given, into ``info`` for graph-level
quantities and ``arrays`` for per-atom ones, each under a caller-chosen
prefix (``--info_prefix``, default ``MACE_``), and then writes the lot out as
extxyz. That is the artefact a user compares against, and until now the
schema could not describe it.

Measured against the file: it writes **thirteen** names, and three of them
resolved to nothing, so a golden pinning the evaluation CLI would have died
on the first one it met -- at authoring time, which is the good case, but
only after the ticket had been scoped as if the harness supported it.

    written              this schema calls it     what differs
    -----------------    ---------------------    ----------------------
    BO_contributions     contributions            the name only
    node_energies        node_energy              the name only (and it is
                                                  not `energies`, which is a
                                                  different quantity)
    descriptors          node_feats               the name, and the shape is
                                                  a decision -- see below
    BEC                  BEC                      flattened to (n_atoms, 9)
                                                  or (n_atoms, 18)
    energy, forces,      the same                 nothing
    stress, magforces,
    latent_*

Two of the three renames are the divergence pattern already seen between the
other two surfaces (``bec``/``BEC``, ``LES_alphas``/``latent_alphas``): the
writer and the model disagree about a name for no reason anybody recorded.
The third is not a rename at all, and is settled here rather than by whoever
meets it first.

This module is imported by ``tests/golden/__init__.py`` for its side effect.
"""

from __future__ import annotations

import numpy as np

from tests.golden import harness

EVAL = harness.SURFACE_EVAL

#: mace/cli/eval_configs.py:106. Not part of any name -- the prefix is an
#: argument, so a schema that baked it in would need a spelling per
#: invocation. `harness.collect_prefixed_outputs` strips it.
DEFAULT_INFO_PREFIX = "MACE_"


# ---------------------------------------------------------------------------
# Plain renames
# ---------------------------------------------------------------------------

#: mace/cli/eval_configs.py:452 -- atoms.info[prefix + "BO_contributions"] is
#: `output["contributions"]` for this configuration, the per-body-order energy
#: terms. "BO" is the writer's abbreviation and appears nowhere else in the
#: tree.
harness.register_alias("BO_contributions", "contributions", surface=EVAL)

#: mace/cli/eval_configs.py:472 -- atoms.arrays[prefix + "node_energies"] is
#: `output["node_energy"]`, plural against the model's singular.
#:
#: And it lands on `energies`, not on `node_energy`, which is the opposite of
#: what this registration said when it was written. The near miss was spotted
#: and then resolved the wrong way round: the two channels do differ by the E0
#: table, and `node_energy` is the one with the reference *subtracted*
#: (mace/calculators/mace.py:792-795 makes `energies` a copy of the model's
#: node_energy and then subtracts node_e0 from `node_energy`) -- so the raw
#: `output["node_energy"]` this CLI writes, with no E0 arithmetic at all, is
#: `energies`. Landing it on `node_energy` meant an eval-route golden and a
#: calculator-route golden of the same model disagreed by the E0 table on a
#: channel whose shape and unit both matched, which is a comparison with
#: nothing to catch it.
#:
#: Measured on the committed tiny_magnetic anchor over all five magnetic
#: fixtures, float64: the CLI's node_energies equals the calculator's
#: `energies` to 5e-9 (extxyz writes per-atom columns as %16.8f) and differs
#: from its `node_energy` by the E0 table, up to 6.75 eV.
harness.register_alias(
    "node_energies",
    "energies",
    surface=EVAL,
    note=(
        "the eval CLI writes the model's raw node_energy, which includes the "
        "isolated-atom reference and is therefore the `energies` channel. The "
        "calculator's `node_energy` has E0 subtracted; the two differ by "
        "exactly the E0 table. Same collision as the model surface, see "
        "model_keys.py."
    ),
)


# ---------------------------------------------------------------------------
# One layout change: the Born effective charges arrive flat
#
# mace/cli/eval_configs.py:433-435 writes
# `atoms.arrays[prefix + "BEC"] = bec.reshape(bec.shape[0], -1)`, because an
# extxyz per-atom column set is two-dimensional and a (n_atoms, 3, 3) cannot
# be written as one. The channel is the model's (n_atoms, 3, 3), so the
# flattening is undone on ingest -- the same treatment the calculator's
# Voigt-6 per-atom stress gets, and for the same reason: one channel, or the
# two routes never compare.
#
# Losslessness is not an argument here, it is arithmetic: `reshape` on a
# C-contiguous array reorders nothing, and reshaping back with the same
# convention is its exact inverse. The round trip is asserted in
# tests/golden/test_harness.py rather than asserted here in prose.
#
# The measured extxyz round trip (ase 3.29) preserves both this and the
# (3, 3) info stress, so a golden may read the written file rather than the
# in-memory structures and meet the same shapes.
# ---------------------------------------------------------------------------


def unflatten_bec(value) -> np.ndarray:
    """Restore a flattened Born-charge block to the model's layout.

    Nine columns per atom is one 3x3; eighteen is the two-component form, and
    the leading pair is the charge and dipole contributions to the
    polarization derivative, in that order (``les/module/bec.py:100-104``
    returns ``torch.stack([result, result_u], dim=1)`` whenever the model
    predicts latent dipoles). This function had to make that call, because a
    ``MACELES`` with ``use_dipole`` on emits exactly it and the eval CLI
    flattens whatever it is given; refusing the wider block meant the only
    LES configuration the eval surface could describe was the one without
    dipoles.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim > 2:
        return arr
    if arr.ndim != 2 or arr.shape[-1] not in (9, 18):
        raise ValueError(
            "the BEC channel is a 3x3 tensor per atom, optionally with a "
            "leading charge/dipole pair, so the eval CLI's flattened form is "
            f"(n_atoms, 9) or (n_atoms, 18); got {arr.shape}. A different "
            "width means `les` changed the layout rather than that this needs "
            "a reshape."
        )
    trailing = (3, 3) if arr.shape[-1] == 9 else (2, 3, 3)
    return arr.reshape(arr.shape[0], *trailing)


harness.register_alias(
    "BEC",
    "BEC",
    surface=EVAL,
    convert=unflatten_bec,
    note=(
        "the eval CLI flattens each atom's Born charges to 9 or 18 columns so "
        "they fit an extxyz array (mace/cli/eval_configs.py:433-435); the "
        "channel is the model's (3, 3) or (2, 3, 3), and reshape is its exact "
        "inverse"
    ),
)


# ---------------------------------------------------------------------------
# The same flattening, twice more: the latent multipoles
#
# `BEC` is not the only thing this surface flattens. mace/cli/eval_configs.py
# :417-423 puts `latent_alphas` and `latent_quads` through the same
# `reshape(n_atoms, -1)`, for the same extxyz reason, and neither had a
# registration -- so an eval-surface LES snapshot resolved both onto their
# channels and then compared (n_atoms, 9) against the model's (n_atoms, 3, 3)
# and failed on the shape. Measured on the committed MACELES anchor: the CLI
# writes latent_quads as (n, 9) and latent_alphas as (n, 1) for the isotropic
# configuration, against (n, 3, 3) and (n,) from the forward.
#
# One wrinkle worth recording, because it makes the alphas look fine until
# they are not: ase drops a trailing axis of one on the extxyz round trip, so
# a golden that reads the *written file* sees (n,) and one that snapshots the
# in-memory structures sees (n, 1). Both land on the channel now.
# ---------------------------------------------------------------------------


def unflatten_latent_tensor(value) -> np.ndarray:
    """Undo the eval CLI's per-atom flattening of a latent multipole.

    Nine columns is a 3x3 per atom; one column is a scalar per atom that
    ``reshape(n, -1)`` widened. Anything else is `les` having changed the
    layout, which is a thing to look at rather than to reshape around.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1 or arr.ndim > 2:
        return arr
    if arr.shape[-1] == 1:
        return arr[:, 0]
    if arr.shape[-1] == 9:
        return arr.reshape(arr.shape[0], 3, 3)
    raise ValueError(
        "a flattened latent multipole is (n_atoms, 9) for a 3x3 per atom or "
        f"(n_atoms, 1) for a scalar per atom; got {arr.shape}"
    )


for _latent in ("latent_alphas", "latent_quads"):
    harness.register_alias(
        _latent,
        _latent,
        surface=EVAL,
        convert=unflatten_latent_tensor,
        note=(
            "the eval CLI flattens each atom's latent multipole to fit an "
            "extxyz array (mace/cli/eval_configs.py:443-449); the channel is "
            "the layout the forward returns"
        ),
    )


# ---------------------------------------------------------------------------
# The one that is a decision, not a rename: descriptors
#
# `--return_descriptors` writes the model's `node_feats`, optionally reduced
# to the invariant (l=0) part and truncated to the requested number of layers
# (mace/cli/eval_configs.py:316-348). Where it lands depends on
# `--descriptor_aggregation_method` (:428-443), and the three cases are three
# different shapes:
#
#   * None (the default): atoms.arrays[prefix + "descriptors"], one row per
#     atom -- (n_atoms, k);
#   * "mean": atoms.info[prefix + "descriptors"], a single (k,) vector, the
#     mean over atoms;
#   * "per_element_mean": atoms.info[prefix + "descriptors"], a *dict* keyed
#     by chemical symbol, whose members are lists. ase round-trips it through
#     extxyz as JSON, so it survives the file and is still a dict on the far
#     side.
#
# The decision, made here rather than by whichever ticket hits it first:
#
#   **the channel is `node_feats`, per atom, and only the per-atom form is
#   pinnable.**
#
# The two aggregations are pure functions of the per-atom array -- a mean over
# all rows, and a mean over the rows of one element. Pinning a mean adds no
# coverage that pinning the rows does not already give, and it costs a second
# channel holding the same physics, which is the silent split this registry
# exists to prevent. And `per_element_mean` is not expressible at all: its
# extent is the number of distinct elements in the structure and its axis is
# labelled by chemical symbol, so no kind describes it and no comparison could
# line two of them up across fixtures with different compositions.
#
# So an aggregated descriptor is refused with a message naming the flag,
# rather than accepted into a shape that means something else. A golden that
# genuinely needs the aggregate should pin the per-atom array and take the
# mean itself, where the reduction is visible.
# ---------------------------------------------------------------------------


def per_atom_descriptors(value) -> np.ndarray:
    """Accept the unaggregated descriptors; refuse the reduced forms."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "the descriptors channel is the per-atom node_feats block, shape "
            f"(n_atoms, k); got {arr.shape}. This is what "
            "--descriptor_aggregation_method produces when it is anything "
            "other than its default of None: 'mean' writes one (k,) vector "
            "per structure into atoms.info, and 'per_element_mean' writes a "
            "dict keyed by chemical symbol. Both are reductions of the "
            "per-atom array, so pin that and reduce it in the test, where the "
            "reduction is visible."
        )
    return arr


harness.register_alias(
    "descriptors",
    "node_feats",
    surface=EVAL,
    convert=per_atom_descriptors,
    note=(
        "the eval CLI's name for node_feats, optionally invariant-only and "
        "layer-truncated; only the unaggregated per-atom form is pinnable"
    ),
)


# ---------------------------------------------------------------------------
# Two things that are true of this surface and are not registrations
#
# 1. `stress` needs no alias here, and that is the interesting part. The
#    calculator converts it to Voigt-6 and the schema converts it back
#    (calculator_keys.py); the eval CLI writes the model's (3, 3) straight
#    into atoms.info (mace/cli/eval_configs.py:430, from a (n_graphs, 3, 3)
#    stack), so on this surface the spelling and the layout both already
#    match the channel. Because aliases are surface-scoped, the calculator's
#    Voigt conversion does not leak here -- which is exactly what a flat
#    spelling->channel map could not have expressed, and it would have
#    silently expanded an already-3x3 tensor.
#
# 2. The `latent_*` family keeps the model's own names on this surface
#    (mace/cli/eval_configs.py:437-449), unlike the calculator, which renames
#    four of them. Nothing to register, but it means the calculator's
#    `LES_alphas` alias is genuinely calculator-scoped rather than a global
#    rename waiting to be applied here too.
# ---------------------------------------------------------------------------
