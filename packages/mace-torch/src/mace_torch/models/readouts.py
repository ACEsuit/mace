"""One readout per declared observable or group of them, one copy per head.

A readout is built from :class:`~mace_core.observables.ObservableSpec`
declarations: their irreps and nothing else decide what the readout produces.
Observables read out together share every map, the frozen dielectric model's
charge, dipole and polarizability among them; :func:`group_irreps` lays them
out. A readout reads every layer's node features and returns one contribution
per layer, which the caller sums: the frozen tree's site energy is a sum over
layers, and a readout of the last layer alone is a different model.

The last layer of a model with more than one is read out through a gate: the
middle keeps the terms of the configured middle whose irrep the output has,
the scalars through a SiLU and every higher copy scaled by the SiLU of a
scalar of its own, as the frozen tree's gated readouts do.

**Each head has its own readout.** Heads share the backbone and nothing after
it. Every readout map produces the observable once per head, and each atom
takes its own head's copy. The copies of one term sit together, head-major,
which is the frozen tree's layout, so a multi-head checkpoint converts by
copying: the observable ``0e+1o`` for two heads is ``2x0e+2x1o``, head ``h``
owning copy ``h`` of each term. The gated readout's middle is laid out the
same way, and every other head's middle is zeroed before the second map, as
the frozen tree's ``mask_head`` does. The second map is dense across heads;
its cross-head blocks multiply zeros, in the frozen tree's weights as here.

One head's readout can be read and written on its own, in the canonical layout
of a one-head readout of the same observables: that is how a head starts from
another head's readout, or from a one-head model's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.kernels.canonical import (
    gate_order,
    linear_path_normalization,
    linear_weight_paths,
)
from mace_core.kernels.precision import Precision
from mace_core.kernels.protocol import CanonicalState
from mace_core.kernels.specs import LinearSpec
from mace_core.observables import ObservableSpec
from torch import Tensor, nn

from mace_torch.backends.reference import torch_dtype
from mace_torch.nn.gate import Gate, gate_input_irreps, scalars_and_gated

__all__ = ["ObservableReadout", "head_columns"]


def _per_head(irreps: str, num_heads: int) -> str:
    """``irreps`` once per head, each term's copies together."""
    return str(Irreps.parse(irreps).with_channels(num_heads))


def _copies(multiplicities: Sequence[int], num_heads: int, head: int) -> list[int]:
    """Of terms of ``multiplicities`` once per head, each term's copies
    together, the copies that are ``head``'s, in the order of one head's."""
    copies, offset = [], 0
    for mul in multiplicities:
        copies.extend(range(offset + head * mul, offset + (head + 1) * mul))
        offset += num_heads * mul
    return copies


def _head_copies(irreps: str, num_heads: int, head: int) -> list[int]:
    """The copies of ``irreps`` once per head that are ``head``'s, in the
    order of the copies of ``irreps``."""
    return _copies([mul for mul, _ in Irreps.parse(irreps).terms], num_heads, head)


def _gate_input_copies(middle: str, num_heads: int, head: int) -> list[int]:
    """The copies of the gate input of ``middle`` once per head that are
    ``head``'s, in the order of a one-head gate input: its scalars, then its
    gates, then each gated term."""
    scalars, gated = scalars_and_gated(middle)
    multiplicities = [mul for mul, _ in gated]
    gates_start = num_heads * scalars
    gated_start = gates_start + num_heads * sum(multiplicities)
    return (
        _copies([scalars], num_heads, head)
        + [gates_start + c for c in _copies(multiplicities, num_heads, head)]
        + [gated_start + c for c in _copies(multiplicities, num_heads, head)]
    )


def head_columns(irreps: str, num_heads: int) -> Tensor:
    """``[num_heads, dim]``: the columns of ``irreps`` once per head that hold
    each head's copy, in the order of ``irreps``."""
    slices = [part for part, _ in Irreps.parse(_per_head(irreps, num_heads)).slices()]
    return torch.tensor(
        [
            [
                column
                for copy in _head_copies(irreps, num_heads, head)
                for column in range(slices[copy].start, slices[copy].stop)
            ]
            for head in range(num_heads)
        ],
        dtype=torch.int64,
    )


def _head_weights(
    *,
    one_in: str,
    one_out: str,
    every_in: str,
    every_out: str,
    in_copies: Sequence[Sequence[int]] | None,
    out_copies: Sequence[Sequence[int]],
) -> tuple[Tensor, Tensor]:
    """Where each weight of a one-head linear map sits in the map of every
    head, and the factor between the two stored weights.

    Args:
        one_in: The one-head map's input declaration.
        one_out: Its output declaration.
        every_in: The map's own input declaration.
        every_out: Its output declaration.
        in_copies: Per head, the map's input copies that are that head's, in
            the order of the one-head map's; ``None`` when every head reads
            the same input.
        out_copies: Per head, its output copies, likewise.

    Returns:
        ``[num_heads, n]`` positions in the canonical weight, head by head,
        and ``[n]`` factors. A stored weight is multiplied by ``1 /
        sqrt(fan_in)`` in the forward, and a map reading every head's middle
        has a larger fan-in than a one-head map, so the same effect needs a
        larger stored weight. The factor is that ratio, the one the frozen
        tree applies when it copies a readout into a model with more heads.
    """
    single = linear_weight_paths(one_in, one_out)
    every = {
        path: weight
        for weight, path in enumerate(linear_weight_paths(every_in, every_out))
    }
    out_irreps = [irrep for _, irrep in Irreps.parse(one_out).slices()]
    factors = [
        linear_path_normalization(one_in, out_irreps[out_copy])
        / linear_path_normalization(every_in, out_irreps[out_copy])
        for out_copy, _ in single
    ]
    positions = [
        [
            every[outputs[out_copy], in_copy if inputs is None else inputs[in_copy]]
            for out_copy, in_copy in single
        ]
        for inputs, outputs in zip(
            in_copies or [None] * len(out_copies), out_copies, strict=True
        )
    ]
    return (
        torch.tensor(positions, dtype=torch.int64),
        torch.tensor(factors, dtype=torch.float64),
    )


def group_irreps(specs: Sequence[ObservableSpec]) -> tuple[str, dict[str, list[int]]]:
    """The declaration a readout of ``specs`` produces, and where each one's
    components sit in it.

    One term per irrep, sorted by degree, odd before even at one degree; within
    a term, the copies of the observables in the order of ``specs``. So
    ``charges`` (``0e``), ``polarizability`` (``0e+2e``) and ``dipole``
    (``1o``) read out together are ``2x0e+1x1o+1x2e``.

    Returns:
        The declaration, and per observable the columns of its components,
        in the order of its own declaration.
    """
    copies: dict[Irrep, int] = {}
    for spec in specs:
        for mul, ir in Irreps.parse(spec.irreps).terms:
            copies[ir] = copies.get(ir, 0) + mul
    order = sorted(copies, key=gate_order)
    start, offset = {}, 0
    for ir in order:
        start[ir] = offset
        offset += copies[ir] * ir.dimension
    taken = dict.fromkeys(order, 0)
    columns: dict[str, list[int]] = {}
    for spec in specs:
        own: list[int] = []
        for mul, ir in Irreps.parse(spec.irreps).terms:
            first = start[ir] + taken[ir] * ir.dimension
            own.extend(range(first, first + mul * ir.dimension))
            taken[ir] += mul
        columns[spec.name] = own
    return "+".join(f"{copies[ir]}x{ir}" for ir in order), columns


def gated_middle(mlp_irreps: str, irreps_out: str) -> str:
    """The terms of ``mlp_irreps`` whose irrep ``irreps_out`` has, as the
    frozen tree's gated readouts keep them; ``""`` when none is left."""
    produced = {ir for _, ir in Irreps.parse(irreps_out).terms}
    return "+".join(
        f"{mul}x{ir}" for mul, ir in Irreps.parse(mlp_irreps).terms if ir in produced
    )


class _GatedReadout(nn.Module):
    """A linear map to a gate's input per head, the gate, every other head's
    middle zeroed, and a linear map to the output."""

    mask: Tensor

    def __init__(
        self,
        backend,
        irreps_in: str,
        middle: str,
        irreps_out: str,
        num_heads: int,
        precision: Precision,
    ) -> None:
        super().__init__()
        #: The middle of every head, each term's copies together.
        self.every_middle = _per_head(middle, num_heads)
        self.linear_in = backend.make_linear(
            LinearSpec(
                irreps_in=irreps_in,
                irreps_out=gate_input_irreps(self.every_middle),
                precision=precision,
            )
        )
        self.activation = Gate(self.every_middle, gate_activation="silu")
        self.linear_out = backend.make_linear(
            LinearSpec(
                irreps_in=self.every_middle, irreps_out=irreps_out, precision=precision
            )
        )
        own = torch.zeros(
            num_heads,
            Irreps.parse(self.every_middle).dimension,
            dtype=torch_dtype(precision),
        )
        own.scatter_(1, head_columns(middle, num_heads), 1.0)
        self.register_buffer("mask", own, persistent=False)

    def forward(self, features: Tensor, node_head: Tensor | None) -> Tensor:
        middle = self.activation(self.linear_in(features))
        if node_head is not None:
            middle = middle * self.mask[node_head]
        return self.linear_out(middle)


class ObservableReadout(nn.Module):
    """The readout of a group of observables, one contribution per layer read.

    Args:
        backend: The kernel backend. Consulted at construction only.
        specs: The observables read out together; one for a readout of its own.
        layer_irreps: Each layer's node-feature declaration, whole, as the
            backbone reports it.
        num_heads: How many heads the model has.
        mlp_irreps: The gated middle per head.
        last_only: Read out only the last layer.
        precision: The dtype every op is built at.

    The last layer of a model with more than one is read out gated, every
    other layer linearly. A layer whose readout would be identically zero is
    not read out: a gated middle that keeps none of the output's irreps, or a
    gate without scalars in its input.

    Raises:
        ValueError: If a declared irrep is given by none of the layers read. A
            readout connects an irrep only to the same irrep, so it would be
            exactly zero there, and a loss on it could never go down.
    """

    columns: Tensor
    component_order: Tensor

    def __init__(
        self,
        backend,
        specs: Sequence[ObservableSpec],
        layer_irreps: Sequence[str],
        num_heads: int,
        mlp_irreps: str,
        last_only: bool = False,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        irreps, components = group_irreps(specs)
        names = tuple(spec.name for spec in specs)
        last = len(layer_irreps) - 1
        middle = gated_middle(mlp_irreps, irreps)
        gated_layer = last if last > 0 else None
        wanted = (last,) if last_only else tuple(range(last + 1))
        given = {
            layer: _given(
                layer_irreps[layer], irreps, middle, gated=layer == gated_layer
            )
            for layer in wanted
        }
        #: The layers read out, in order: those whose readout is not zero.
        self.layers = tuple(layer for layer in wanted if given[layer])
        _check_reachable(
            specs,
            irreps,
            set().union(*given.values()),
            [layer_irreps[layer] for layer in wanted],
            (
                f"{middle or 'nothing'!r} kept of mlp_irreps={mlp_irreps!r}"
                if gated_layer in wanted
                else None
            ),
        )
        self.num_heads = num_heads
        #: The observables, in the order their components are laid out.
        self.observables = names
        self.component_sizes = tuple(len(components[name]) for name in names)
        self.register_buffer(
            "component_order",
            torch.tensor([c for name in names for c in components[name]]),
            persistent=False,
        )
        #: Per op, by its path in this module, where each weight of a one-head
        #: readout's op sits in this one's, head by head, and the factor
        #: between the two stored weights.
        self.head_weights: dict[str, tuple[Tensor, Tensor]] = {}
        out_copies = [_head_copies(irreps, num_heads, h) for h in range(num_heads)]
        irreps_out = _per_head(irreps, num_heads)
        maps = []
        for layer in self.layers:
            path = f"maps.{len(maps)}"
            irreps_in = layer_irreps[layer]
            if layer == gated_layer:
                gated = _GatedReadout(
                    backend, irreps_in, middle, irreps_out, num_heads, precision
                )
                self.head_weights[f"{path}.linear_in"] = _head_weights(
                    one_in=irreps_in,
                    one_out=gate_input_irreps(middle),
                    every_in=irreps_in,
                    every_out=gate_input_irreps(gated.every_middle),
                    in_copies=None,
                    out_copies=[
                        _gate_input_copies(middle, num_heads, h)
                        for h in range(num_heads)
                    ],
                )
                self.head_weights[f"{path}.linear_out"] = _head_weights(
                    one_in=middle,
                    one_out=irreps,
                    every_in=gated.every_middle,
                    every_out=irreps_out,
                    in_copies=[
                        _head_copies(middle, num_heads, h) for h in range(num_heads)
                    ],
                    out_copies=out_copies,
                )
                maps.append(gated)
                continue
            maps.append(
                backend.make_linear(
                    LinearSpec(
                        irreps_in=irreps_in, irreps_out=irreps_out, precision=precision
                    )
                )
            )
            self.head_weights[path] = _head_weights(
                one_in=irreps_in,
                one_out=irreps,
                every_in=irreps_in,
                every_out=irreps_out,
                in_copies=None,
                out_copies=out_copies,
            )
        self.maps = nn.ModuleList(maps)
        self.register_buffer(
            "columns", head_columns(irreps, num_heads), persistent=False
        )

    def split(self, values: Tensor) -> dict[str, Tensor]:
        """One contribution's components, by observable."""
        if len(self.observables) == 1:
            return {self.observables[0]: values}
        pieces = values.index_select(1, self.component_order).split(
            self.component_sizes, dim=1
        )
        return dict(zip(self.observables, pieces, strict=True))

    def forward(
        self, layers: Sequence[Tensor], node_head: Tensor | None
    ) -> list[Tensor]:
        """``[n_nodes, dim]`` per layer read, each node its own head's copy,
        the observables side by side; :meth:`split` separates them.

        Args:
            layers: Every layer's node features, as the backbone returns them.
            node_head: ``[n_nodes]``, each node's head; ``None`` for a model
                with one head.
        """
        if (node_head is None) != (self.num_heads == 1):
            raise ValueError(
                f"the readout has {self.num_heads} head(s) and was "
                f"{'not ' if node_head is None else ''}given each node's head; "
                f"a model with more than one head reads it, one with one does not."
            )
        contributions = []
        for layer, readout in zip(self.layers, self.maps, strict=True):
            if isinstance(readout, _GatedReadout):
                values = readout(layers[layer], node_head)
            else:
                values = readout(layers[layer])
            if node_head is not None:
                values = values.gather(1, self.columns[node_head])
            contributions.append(values)
        return contributions

    def head_state(self, head: int) -> dict[str, dict[str, Tensor]]:
        """One head's readout, as a one-head readout of the same observables
        and layers saves it: by op path, each op's canonical tensors.

        Raises:
            IndexError: If the readout has no head ``head``.
        """
        self._check_head(head)
        state = {}
        for path, (positions, factors) in self.head_weights.items():
            weight = self._op(path).to_canonical()["weight"]
            state[path] = {
                "weight": weight[positions[head].to(weight.device)]
                / factors.to(weight),
                "bias": weight.new_zeros(0),
            }
        return state

    def load_head_state(
        self, head: int, state: Mapping[str, Mapping[str, Tensor]]
    ) -> None:
        """Write one head's readout from a :meth:`head_state`, of this
        readout or of another with the same observables and layers. The other
        heads are left as they are.

        Raises:
            IndexError: If the readout has no head ``head``.
            ValueError: If ``state`` is not a readout of this shape. Nothing
                is written then.
        """
        self._check_head(head)
        if set(state) != set(self.head_weights):
            raise ValueError(
                f"the state holds the ops {sorted(state)} and one head of this "
                f"readout has {sorted(self.head_weights)}."
            )
        for path, (positions, _) in self.head_weights.items():
            shape = tuple(state[path]["weight"].shape)
            if shape != (positions.shape[1],):
                raise ValueError(
                    f"{path}: the state's weight is {shape} and one head of "
                    f"this readout holds ({positions.shape[1]},)."
                )
        for path, (positions, factors) in self.head_weights.items():
            op = self._op(path)
            canonical = op.to_canonical()
            weight = canonical["weight"].clone()
            weight[positions[head].to(weight.device)] = state[path]["weight"].to(
                weight
            ) * factors.to(weight)
            op.load_canonical({"weight": weight, "bias": canonical["bias"]})

    def _op(self, path: str) -> CanonicalState:
        return cast(CanonicalState, self.get_submodule(path))

    def _check_head(self, head: int) -> None:
        if not 0 <= head < self.num_heads:
            raise IndexError(
                f"head {head} is not one of this readout's {self.num_heads}."
            )


def _given(irreps_in: str, irreps_out: str, middle: str, gated: bool) -> set[Irrep]:
    """The irreps of ``irreps_out`` a readout of ``irreps_in`` gives: those it
    has, by a linear map; through the gate, only those ``middle`` keeps too,
    and none without scalars to activate."""
    available = {ir for _, ir in Irreps.parse(irreps_in).terms}
    given = available & {ir for _, ir in Irreps.parse(irreps_out).terms}
    if not gated:
        return given
    if not middle or Irrep(0, 1) not in available:
        return set()
    return given & {ir for _, ir in Irreps.parse(middle).terms}


def _check_reachable(
    specs: Sequence[ObservableSpec],
    irreps: str,
    given: set[Irrep],
    read_irreps: Sequence[str],
    middle: str | None,
) -> None:
    """Refuse observables some irrep of which no layer read out gives.

    ``middle`` describes the gated middle of the last layer read, ``None``
    when no layer read is gated.
    """
    missing = [f"{ir}" for _, ir in Irreps.parse(irreps).terms if ir not in given]
    if not missing:
        return
    if len(specs) == 1:
        subject = f"the observable {specs[0].name!r} declares {specs[0].irreps!r},"
    else:
        declared = ", ".join(f"{spec.name!r} {spec.irreps!r}" for spec in specs)
        subject = f"the observables read out together, {declared},"
    through = f", the last through the gated middle {middle}" if middle else ""
    widen = "the node features or the middle" if middle else "the node features"
    raise ValueError(
        f"{subject} and no layer read out gives {missing}: the layers read "
        f"carry {list(read_irreps)}{through}. A readout cannot create an irrep "
        f"its input lacks, so those components would be zero. Widen {widen}, "
        f"or change the declaration."
    )
