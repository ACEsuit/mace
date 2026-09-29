"""Reading a legacy dipole or dielectric model's weights into the rewrite.

The backbone goes through :mod:`tests.parity.fm00_convert` unchanged. What is
added here is the readouts, and they are the part whose shape differs.

**One legacy readout becomes up to three heads.** The frozen tree reads the
charge, the polarizability's scalar, the dipole and the polarizability's
``2e`` out of one linear map to ``2x0e+1x1o+1x2e``, and slices the result. The
rewrite gives each observable its own head. A linear map joins each output
copy to input copies independently, so splitting it by output copy is exact:
each head takes the rows of the copies it reads out.

**The gated readout's middle is shared in the frozen tree and repeated here.**
Each head gets its own first map and gate, built from the same weights: the
scalars, the gates of the irreps that head reads out, and those irreps. The
outputs are the same numbers; what differs is that a model trained from
scratch would give each head its own middle.

Every weight is addressed by copy, one multiplicity of one term, as
:func:`~mace_core.kernels.canonical.linear_weight_table` addresses it.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import linear_weight_table

from tests.parity.fm00_convert import build_config, transfer_backbone_weights

__all__ = ["dipole_config", "transfer_dipole_weights"]

#: Which legacy output copy each head's output copy is.
_OUTPUT_COPIES = {
    "dipole": {"atomic_dipoles": [0]},
    "dielectric": {"charges": [0], "polarizability_sh": [1, 3], "atomic_dipoles": [2]},
}


def dipole_config(legacy) -> dict:
    """The model settings of a legacy dipole or dielectric model."""
    config = build_config(legacy)
    for key in ("pair_repulsion", "num_heads"):
        config.pop(key)
    config["cutoff_order"] = int(legacy.radial_embedding.cutoff_fn.p)
    config["readout_hidden"] = str(legacy.readouts[-1].hidden_irreps)
    return config


def _copies(irreps: str) -> list[tuple[str, int]]:
    """Each copy as its irrep and its channel within that irrep's terms."""
    seen: dict[str, int] = {}
    copies = []
    for mul, ir in Irreps.parse(irreps).terms:
        for _ in range(mul):
            name = str(ir)
            copies.append((name, seen.get(name, 0)))
            seen[name] = seen.get(name, 0) + 1
    return copies


def _legacy_entries(linear) -> dict[tuple[int, int], float]:
    """A legacy linear's weight for each (output copy, input copy) it joins,
    with its normalization folded in. A map with an empty side joins nothing."""
    if not str(linear.irreps_in) or not str(linear.irreps_out):
        return {}
    source = Irreps.parse(str(linear.irreps_in))
    target = Irreps.parse(str(linear.irreps_out))
    starts_in = np.cumsum([0] + [mul for mul, _ in source.terms])
    starts_out = np.cumsum([0] + [mul for mul, _ in target.terms])
    flat = linear.weight.detach().numpy().astype(float)
    entries, offset = {}, 0
    for instruction in linear.instructions:
        rows, columns = instruction.path_shape
        block = flat[offset : offset + rows * columns].reshape(rows, columns)
        offset += rows * columns
        for row in range(rows):
            for column in range(columns):
                key = (
                    int(starts_out[instruction.i_out]) + column,
                    int(starts_in[instruction.i_in]) + row,
                )
                entries[key] = float(block[row, column] * instruction.path_weight)
    return entries


def _load(destination, entries, v1_in, v1_out, in_map, out_map) -> None:
    """Write a rewrite linear from legacy entries, copy by copy.

    ``in_map`` and ``out_map`` give, for each copy of the rewrite's input and
    output, the legacy copy it is, or ``None`` where the legacy map has none,
    whose weight is then zero.
    """
    weights = torch.zeros_like(destination.weight)
    for (out_copy, in_copy), index in linear_weight_table(v1_in, v1_out).items():
        source = (out_map[out_copy], in_map[in_copy])
        if None not in source:
            weights[index] = entries.get(source, 0.0)
    destination.weight.copy_(weights)


def _by_role(copies, roles):
    """Map copies described by role to the legacy copies with the same role."""
    return [roles.get(copy) for copy in copies]


def _gated_maps(readout, destination):
    """The copy maps of one legacy gated readout onto one head's gated readout.

    The legacy middle is scalars, then one gate per gated channel, then the
    gated channels, sorted: every scalar first, the gates in the order of the
    gated terms. The rewrite's is the same sections for its own head.
    """
    nonlinear = readout.equivariant_nonlin
    scalars = (
        Irreps.parse(str(nonlinear.irreps_scalars))
        if len(nonlinear.irreps_scalars)
        else None
    )
    gated = (
        Irreps.parse(str(nonlinear.irreps_gated))
        if len(nonlinear.irreps_gated)
        else None
    )
    num_scalars = scalars.dimension if scalars else 0
    legacy_in: dict[tuple, int] = {}
    for channel in range(num_scalars):
        legacy_in[("scalar", channel)] = channel
    gate_offset = num_scalars
    gated_start = num_scalars + (sum(mul for mul, _ in gated.terms) if gated else 0)
    for mul, ir in gated.terms if gated else ():
        for channel in range(mul):
            legacy_in[("gate", str(ir), channel)] = gate_offset + channel
            legacy_in[("gated", str(ir), channel)] = gated_start + channel
        gate_offset += mul
        gated_start += mul
    # The second map reads the frozen tree's `MLP_irreps`, whose scalars and
    # higher irreps are exactly the gate's output.
    legacy_second: dict[tuple, int] = {}
    for position, (ir, channel) in enumerate(_copies(str(readout.linear_2.irreps_in))):
        role = ("scalar", channel) if ir == "0e" else ("gated", ir, channel)
        legacy_second[role] = position

    gate = destination.gate
    rewrite_in_roles = []
    for _, channel in _copies(str(gate.scalars)):
        rewrite_in_roles.append(("scalar", channel))
    gated_terms = _copies(gate.gated_declaration) if gate.gated_declaration else []
    for ir, channel in gated_terms:
        rewrite_in_roles.append(("gate", ir, channel))
    for ir, channel in gated_terms:
        rewrite_in_roles.append(("gated", ir, channel))
    rewrite_second_roles = [("scalar", c) for _, c in _copies(str(gate.scalars))] + [
        ("gated", ir, c) for ir, c in gated_terms
    ]
    return (
        _by_role(rewrite_in_roles, legacy_in),
        _by_role(rewrite_second_roles, legacy_second),
    )


def transfer_dipole_weights(legacy, model, family: str) -> None:
    """Copy every weight of a legacy ``AtomicDipolesMACE`` (``"dipole"``) or
    ``AtomicDielectricMACE`` (``"dielectric"``) into a rewrite model."""
    transfer_backbone_weights(legacy, model, build_config(legacy)["correlation"])
    with torch.no_grad():
        for name, legacy_copies in _OUTPUT_COPIES[family].items():
            head = model.outputs.heads[name]
            for readout_index, layer in enumerate(head.reachable):
                source = legacy.readouts[layer]
                destination = head.readouts[readout_index]
                if not hasattr(source, "linear_1"):
                    entries = _legacy_entries(source.linear)
                    v1_in, v1_out = head._grouped[readout_index], head.spec.irreps
                    _load(
                        destination,
                        entries,
                        v1_in,
                        v1_out,
                        list(range(len(_copies(v1_in)))),
                        legacy_copies,
                    )
                    continue
                first_in, second_in = _gated_maps(source, destination)
                v1_in = head._grouped[readout_index]
                _load(
                    destination.first,
                    _legacy_entries(source.linear_1),
                    v1_in,
                    destination.gate.irreps_in,
                    list(range(len(_copies(v1_in)))),
                    first_in,
                )
                _load(
                    destination.second,
                    _legacy_entries(source.linear_2),
                    destination.gate.irreps_out,
                    head.spec.irreps,
                    second_in,
                    legacy_copies,
                )
