"""One head per declared observable.

The frozen tree has ten readout blocks and three model classes that exist only
to select among them. Six of the ten are reachable by name; the dipole pair and
the latent-charge pair are chosen by the owning model class instead, which is
the coupling this replaces. Here a head is built from an
:class:`~mace_core.observables.ObservableSpec`: the declaration says the irreps
and whether the value is per atom, and that is enough to build the readout.

A head reads the node features of every layer, not only the last. That is the
frozen tree's behaviour and it is not incidental: the site energy is a sum of
per-layer contributions, so a readout that saw only the final layer would be a
different model.

**Each level of theory gets its own readout.** A multi-head model shares the
backbone and nothing after it: the readout of every layer produces one copy of
the observable per head, and each atom takes the copy of the head its structure
belongs to. That is what lets a replay head keep a foundation model's readout
while a new head learns a different level of theory on top of the same
features; with one readout shared between them, both would pull the same
weights towards two different targets and the only freedom left per head would
be an isolated-atom energy and an affine map.

The layout is the frozen tree's, so a multi-head legacy checkpoint converts by
copying. The copies of one term sit together, head-major: ``2x0e`` read out for
two heads is ``2x0e`` with head ``h`` at position ``h``, and a gated readout's
middle is ``(heads * width)x0e`` with head ``h`` owning channels
``[h * width, (h + 1) * width)``. The gated readout zeroes every other head's
middle before its second map, which is what ``mask_head`` does in the frozen
tree. The second map is therefore dense across heads with its cross-head
blocks multiplying zeros: they exist in the frozen tree's weights too, and
receive no gradient in either.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import linear_bias_table, linear_weight_table
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableSpec
from torch import Tensor, nn

from mace_torch.nn.layout import expanded_irreps
from mace_torch.nn.radial_mlp import SECOND_MOMENT_SCALE

__all__ = [
    "ObservableHead",
    "component_heads",
    "copy_heads",
    "head_columns",
    "per_head_irreps",
]


def per_head_irreps(irreps: str, num_heads: int) -> str:
    """``num_heads`` copies of a declaration, each term's copies together.

    With one head this is the declaration unchanged, spelling included, so a
    single-head model builds exactly the ops it built before heads existed.
    """
    if num_heads == 1:
        return irreps
    parsed = Irreps.parse(irreps)
    return "+".join(f"{num_heads * mul}x{ir}" for mul, ir in parsed.terms)


def head_columns(irreps: str, num_heads: int) -> np.ndarray:
    """Where each head's copy of ``irreps`` sits in :func:`per_head_irreps`.

    Returns:
        ``[num_heads, dim]`` of int64. Row ``h`` lists, in the single-head
        layout's order, the components of head ``h``'s copy.
    """
    parsed = Irreps.parse(irreps)
    width = parsed.dimension
    columns = np.empty((num_heads, width), dtype=np.int64)
    single_offset = 0
    offset = 0
    for mul, ir in parsed.terms:
        span = ir.dimension
        for head in range(num_heads):
            for copy in range(mul):
                source = offset + (head * mul + copy) * span
                target = single_offset + copy * span
                columns[head, target : target + span] = np.arange(source, source + span)
        single_offset += mul * span
        offset += num_heads * mul * span
    return columns


def component_heads(irreps: str, num_heads: int) -> np.ndarray:
    """Which head owns each component of :func:`per_head_irreps`.

    Returns:
        ``[dim * num_heads]`` of int64, the inverse of :func:`head_columns`.
    """
    columns = head_columns(irreps, num_heads)
    owner = np.empty(columns.size, dtype=np.int64)
    for head in range(num_heads):
        owner[columns[head]] = head
    return owner


def copy_heads(irreps: str, num_heads: int) -> list[int]:
    """Which head owns each copy of :func:`per_head_irreps`, in slice order.

    A copy is one multiplicity of one term. Within a term the copies are
    head-major, so copy ``m`` of a term of single-head multiplicity ``mul``
    belongs to head ``m // mul``.
    """
    owners: list[int] = []
    for mul, _ in Irreps.parse(irreps).terms:
        owners.extend(copy // mul for copy in range(num_heads * mul))
    return owners


def _ranked(owners: list[int], head: int) -> dict[int, int]:
    """Each of ``head``'s copies, by its position among that head's copies."""
    mine = [copy for copy, owner in enumerate(owners) if owner == head]
    return {copy: rank for rank, copy in enumerate(mine)}


def _head_indices(
    irreps_in: str,
    in_owners: list[int] | None,
    irreps_out: str,
    out_owners: list[int],
    head: int,
) -> tuple[list[int], list[int]]:
    """The canonical weight and bias entries one head's readout consists of.

    Ordered by the head's own copies rather than by where they sit, so the
    list for one head and the list for another pair up entry by entry, which
    is what copying a readout from one head to another needs.

    Args:
        in_owners: The head of each input copy, or ``None`` when the input is
            shared by every head, which the node features are.

    Returns:
        ``(weight indices, bias indices)``. A weight joining this head's output
        to another head's middle is left out: it multiplies a masked zero, so
        it is not part of what the head computes.
    """
    out_rank = _ranked(out_owners, head)
    if in_owners is None:
        count = sum(1 for _ in Irreps.parse(irreps_in).slices())
        in_rank = {copy: copy for copy in range(count)}
    else:
        in_rank = _ranked(in_owners, head)
    pairs = sorted(
        (out_rank[out_copy], in_rank[in_copy], index)
        for (out_copy, in_copy), index in linear_weight_table(
            irreps_in, irreps_out
        ).items()
        if out_copy in out_rank and in_copy in in_rank
    )
    biases = sorted(
        (out_rank[out_copy], index)
        for out_copy, index in linear_bias_table(irreps_out).items()
        if out_copy in out_rank
    )
    return [index for *_, index in pairs], [index for _, index in biases]


def _entries(
    canonical: dict[str, Tensor], indices: tuple[list[int], list[int]]
) -> dict[str, list[int]]:
    """The weight and bias entries of one head, for an op as it was built.

    An op built without a bias holds an empty bias, and then no head owns any
    of it, whatever its outputs could have carried.
    """
    weights, biases = indices
    return {
        "weight": weights,
        "bias": biases if canonical["bias"].numel() else [],
    }


def _is_reachable(spec: ObservableSpec, hidden_irreps: str) -> bool:
    """Whether a layer's features carry every irrep this observable declares."""
    available = {ir for _, ir in Irreps.parse(hidden_irreps).terms}
    return all(ir in available for _, ir in Irreps.parse(spec.irreps).terms)


def _check_reachable(spec: ObservableSpec, hidden_irreps: str) -> None:
    """Refuse an observable the node features cannot carry.

    An equivariant map connects an irrep only to the same irrep, so a readout
    cannot produce a degree the features do not already have. Built anyway, the
    head returns **exactly zero** on those components, with no error anywhere:
    a polarizability read out of scalar-and-vector features is a column of
    zeros that trains to a loss it can never reduce. The declaration is checked
    against the features instead.
    """
    available = {ir for _, ir in Irreps.parse(hidden_irreps).terms}
    missing = sorted(
        {str(ir) for _, ir in Irreps.parse(spec.irreps).terms if ir not in available}
    )
    if missing:
        raise ValueError(
            f"the observable {spec.name!r} declares {spec.irreps!r}, but the "
            f"node features are {hidden_irreps!r} and carry no {missing}. An "
            f"equivariant readout cannot create an irrep its input lacks, so "
            f"those components would be zero. Widen the node features or "
            f"change the declaration."
        )


class _Gate(nn.Module):
    """Scalars through a normalized SiLU, everything else scaled by a gate.

    The standard equivariant gate. A pointwise nonlinearity is only equivariant
    on scalars, so a higher irrep is instead multiplied by a scalar, which
    leaves its direction alone and changes only its length. The gates go
    through the same normalized SiLU as the scalars, not a sigmoid: that is the
    frozen tree's readout, whose gates take the model's one activation.
    """

    gate_repeat: Tensor

    def __init__(self, irreps_scalars: str, irreps_gated: str) -> None:
        super().__init__()
        self.scalars = Irreps.parse(irreps_scalars)
        # An empty string, not an empty `Irreps`: a scalar output has nothing
        # to gate, and the declaration grammar has no spelling for "nothing".
        self.gated_declaration = irreps_gated
        gated = Irreps.parse(irreps_gated) if irreps_gated else None
        self.num_gates = sum(mul for mul, _ in gated.terms) if gated else 0
        self.gated_dim = gated.dimension if gated else 0
        self.scalar_dim = self.scalars.dimension
        repeats = []
        if gated:
            for mul, ir in gated.terms:
                repeats.extend([ir.dimension] * mul)
        self.register_buffer(
            "gate_repeat", torch.tensor(repeats, dtype=torch.long), persistent=False
        )

    @property
    def irreps_in(self) -> str:
        pieces = [str(self.scalars)]
        if self.num_gates:
            pieces.append(f"{self.num_gates}x0e")
        if self.gated_declaration:
            pieces.append(self.gated_declaration)
        return "+".join(pieces)

    @property
    def irreps_out(self) -> str:
        pieces = [str(self.scalars)]
        if self.gated_declaration:
            pieces.append(self.gated_declaration)
        return "+".join(pieces)

    def forward(self, features: Tensor) -> Tensor:
        # The same unit-second-moment multiplier the radial network carries.
        # Trained readouts depend on it exactly as trained radial networks do.
        scalars = SECOND_MOMENT_SCALE * torch.nn.functional.silu(
            features[..., : self.scalar_dim]
        )
        if not self.num_gates:
            return scalars
        gates = SECOND_MOMENT_SCALE * torch.nn.functional.silu(
            features[..., self.scalar_dim : self.scalar_dim + self.num_gates]
        )
        gated = features[..., self.scalar_dim + self.num_gates :]
        return torch.cat(
            [scalars, gated * torch.repeat_interleave(gates, self.gate_repeat, dim=-1)],
            dim=-1,
        )


class ObservableHead(nn.Module):
    """The readout for one observable.

    Args:
        backend: The kernel backend. Consulted at construction only.
        spec: The declaration this head exists to satisfy.
        layer_irreps: One channel's node-feature declaration **per layer**.
            They differ: the last layer of a trained model carries only its
            invariants, which is why its width is not the others'.
        num_features: The channel width.
        nonlinear: Whether the last layer's readout carries a gate. The frozen
            tree makes exactly this choice, and only for the last layer.
        readout_irreps: The gated readout's middle, per head, as the frozen
            tree's ``MLP_irreps``: ``16`` is ``"16x0e"``. See
            :class:`_GatedReadout` for what of it the middle keeps.
        precision: The dtype every op is built at.
        num_heads: How many levels of theory read this observable out. Each
            gets its own readout weights; see the module docstring for how
            they are laid out.
    """

    columns: Tensor

    def __init__(
        self,
        backend,
        spec: ObservableSpec,
        layer_irreps: Sequence[str],
        num_features: int,
        nonlinear: bool = False,
        readout_irreps: int | str = 16,
        precision: Precision = "float64",
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(
                f"the observable {spec.name!r} is read out for {num_heads} "
                f"heads. A model has at least one, which is the single-head "
                f"case."
            )
        self.spec = spec
        self.per_atom = spec.per_atom
        self.dimension = spec.dimension
        self.num_heads = num_heads
        self.register_buffer(
            "columns",
            torch.tensor(head_columns(spec.irreps, num_heads)),
            persistent=False,
        )
        self.layer_irreps = list(layer_irreps)
        reachable = [
            index
            for index, irreps in enumerate(self.layer_irreps)
            if _is_reachable(spec, irreps)
        ]
        if not reachable:
            _check_reachable(spec, self.layer_irreps[-1])
        self.reachable = reachable

        readouts: list[nn.Module] = []
        gated: list[bool] = []
        self._grouped: list[str] = []
        for index in reachable:
            grouped = expanded_irreps(self.layer_irreps[index], num_features)
            last = index == len(self.layer_irreps) - 1
            if nonlinear and last:
                readouts.append(
                    _GatedReadout(
                        backend,
                        grouped,
                        spec.irreps,
                        readout_irreps,
                        precision,
                        num_heads=num_heads,
                    )
                )
            else:
                readouts.append(
                    backend.make_linear(
                        LinearDescriptor(
                            irreps_in=grouped,
                            irreps_out=per_head_irreps(spec.irreps, num_heads),
                            precision=precision,
                        )
                    )
                )
            gated.append(nonlinear and last)
            self._grouped.append(grouped)
        self.readouts = nn.ModuleList(readouts)
        # Decided here, as constants, so the forward reads a tuple of booleans
        # rather than asking each readout what it is.
        self._gated = tuple(gated)

    def per_layer(
        self, layers: list[Tensor], node_head: Tensor | None = None
    ) -> list[Tensor]:
        """This observable's contribution from each layer, unsummed.

        The energy path needs the terms apart, because the sum over layers and
        the sum over atoms are two different reductions and the energy head
        keeps them separate on purpose.

        Args:
            layers: One tensor per layer.
            node_head: ``[n_atoms]``, which head each atom's structure belongs
                to. Required with more than one head, since without it there is
                no saying whose readout an atom takes.

        Returns:
            One ``[n_atoms, dimension]`` tensor per reachable layer, each
            atom's value already the one its own head reads out.
        """
        if self.num_heads > 1 and node_head is None:
            raise ValueError(
                f"the observable {self.spec.name!r} has {self.num_heads} heads "
                f"and was given no per-atom head index, so there is no saying "
                f"which head's readout an atom takes."
            )
        # Only the layers whose declaration can carry this observable. A layer
        # that cannot contributes nothing, rather than a readout of zeros.
        values = []
        for index, readout, gated in zip(
            self.reachable, self.readouts, self._gated, strict=True
        ):
            features = layers[index]
            value = readout(features, node_head) if gated else readout(features)
            if self.num_heads > 1:
                assert node_head is not None
                value = value.gather(1, self.columns[node_head])
            values.append(value)
        return values

    def head_state(self, head: int) -> list[dict[str, dict[str, Tensor]]]:
        """One head's readout weights, in the canonical layout, per layer.

        The same shapes for every head of a model, and entry by entry in the
        same order, so the state of one head loads into another. That is how a
        fine-tune starts a new head from a foundation model's: it copies a
        readout, and nothing about which head it came from travels with it.

        Returns:
            One entry per layer, keyed by op (``"linear"``, or ``"first"`` and
            ``"second"`` for the gated readout), each ``{"weight", "bias"}``.
        """
        self._check_head(head)
        state = []
        for layer in range(len(self.readouts)):
            ops = {}
            for name, op, indices in self._layer_ops(layer, head):
                canonical = op.to_canonical()
                ops[name] = {
                    key: canonical[key][index].clone()
                    for key, index in _entries(canonical, indices).items()
                }
            state.append(ops)
        return state

    def load_head_state(
        self, head: int, state: list[dict[str, dict[str, Tensor]]]
    ) -> None:
        """Write one head's readout weights, leaving every other head's alone.

        Raises:
            ValueError: If the state has a different number of layers, ops or
                entries, which is a readout from a differently shaped model.
        """
        self._check_head(head)
        if len(state) != len(self.readouts):
            raise ValueError(
                f"the state has {len(state)} layers and this readout has "
                f"{len(self.readouts)}, so it came from a different model."
            )
        for layer, layer_state in enumerate(state):
            for name, op, indices in self._layer_ops(layer, head):
                if name not in layer_state:
                    raise ValueError(
                        f"layer {layer} of the state has no {name!r} op; it "
                        f"carries {sorted(layer_state)}."
                    )
                canonical = {
                    key: value.clone() for key, value in op.to_canonical().items()
                }
                for key, index in _entries(canonical, indices).items():
                    values = layer_state[name][key]
                    if values.shape[0] != len(index):
                        raise ValueError(
                            f"layer {layer}, {name!r} {key}: the state has "
                            f"{values.shape[0]} entries and one head here "
                            f"holds {len(index)}."
                        )
                    canonical[key][index] = values.to(canonical[key])
                op.load_canonical(canonical)

    def _check_head(self, head: int) -> None:
        if not 0 <= head < self.num_heads:
            raise IndexError(
                f"head {head} does not exist; the observable {self.spec.name!r} "
                f"is read out for {self.num_heads}."
            )

    def _layer_ops(self, layer: int, head: int):
        """The ops of one layer's readout, with one head's entries in each."""
        readout = self.readouts[layer]
        out_irreps = per_head_irreps(self.spec.irreps, self.num_heads)
        out_owners = copy_heads(self.spec.irreps, self.num_heads)
        if not self._gated[layer]:
            yield (
                "linear",
                readout,
                _head_indices(self._grouped[layer], None, out_irreps, out_owners, head),
            )
            return
        assert isinstance(readout, _GatedReadout)
        yield (
            "first",
            readout.first,
            _head_indices(
                self._grouped[layer],
                None,
                readout.gate.irreps_in,
                readout.middle_in_owners,
                head,
            ),
        )
        yield (
            "second",
            readout.second,
            _head_indices(
                readout.gate.irreps_out,
                readout.middle_out_owners,
                out_irreps,
                out_owners,
                head,
            ),
        )

    def forward(self, layers: list[Tensor], node_head: Tensor | None = None) -> Tensor:
        """The per-atom value of this observable.

        Args:
            layers: One ``[n_atoms, channels, width]`` tensor per layer.
            node_head: ``[n_atoms]``, required with more than one head.

        Returns:
            ``[n_atoms, dimension]``. Reducing to graph level is the output
            layer's job, since it is the same reduction for every observable.
        """
        terms = self.per_layer(layers, node_head)
        total = terms[0]
        for value in terms[1:]:
            total = total + value
        return total


class _GatedReadout(nn.Module):
    """Linear, gate, linear. The frozen tree's nonlinear readout, rebuilt.

    The middle is read off ``readout_irreps``, the frozen tree's
    ``MLP_irreps``, by its rule: every scalar it declares, and of its higher
    irreps those the output has, each with one gate per channel. An output
    irrep the middle does not carry has no path through the second map and
    reads out exactly zero, which is what the frozen tree's readout returns
    there too. With several heads every section of the middle and of the
    output carries one copy per head, and the middle is masked to the atom's
    own head before the second map.
    """

    hidden_heads: Tensor

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        readout_irreps: int | str,
        precision: Precision,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        out = Irreps.parse(irreps_out)
        if not out.terms:
            raise ValueError(f"{irreps_out!r} declares no irreps to read out.")
        hidden = Irreps.parse(
            f"{readout_irreps}x0e"
            if isinstance(readout_irreps, int)
            else readout_irreps
        )
        wanted = {ir for _, ir in out.terms}
        hidden_scalars = sum(
            mul for mul, ir in hidden.terms if ir.degree == 0 and ir.parity == 1
        )
        if not hidden_scalars:
            raise ValueError(
                f"the readout middle {readout_irreps!r} declares no scalars, and "
                f"a gated readout needs them: they are what the nonlinearity "
                f"acts on. Add a `0e` term."
            )
        # The middle always carries scalars, whether or not the output does:
        # they are what the nonlinearity acts on.
        gated = "+".join(
            f"{mul}x{ir}"
            for mul, ir in hidden.terms
            if not (ir.degree == 0 and ir.parity == 1) and ir in wanted
        )
        middle = f"{hidden_scalars}x0e" + (f"+{gated}" if gated else "")
        num_gates = sum(mul for mul, _ in Irreps.parse(gated).terms) if gated else 0
        # The head of each copy on either side of the gate. Going in there are
        # the scalars, one gate per gated channel, and the gated channels;
        # coming out, the gates have been spent.
        self.middle_in_owners = (
            copy_heads(f"{hidden_scalars}x0e", num_heads)
            + (copy_heads(f"{num_gates}x0e", num_heads) if num_gates else [])
            + (copy_heads(gated, num_heads) if gated else [])
        )
        self.middle_out_owners = copy_heads(middle, num_heads)
        self.gate = _Gate(
            f"{num_heads * hidden_scalars}x0e",
            per_head_irreps(gated, num_heads) if gated else "",
        )
        self.first = backend.make_linear(
            LinearDescriptor(
                irreps_in=irreps_in,
                irreps_out=self.gate.irreps_in,
                precision=precision,
            )
        )
        self.second = backend.make_linear(
            LinearDescriptor(
                irreps_in=self.gate.irreps_out,
                irreps_out=per_head_irreps(irreps_out, num_heads),
                precision=precision,
            )
        )
        # Which head owns each channel of the gate's output. The gate's output
        # is the per-head copies of `middle`, and that is the layout the owner
        # table is read off.
        self.register_buffer(
            "hidden_heads",
            torch.tensor(component_heads(middle, num_heads)),
            persistent=False,
        )

    def forward(self, features: Tensor, node_head: Tensor | None = None) -> Tensor:
        middle = self.gate(self.first(features))
        if self.num_heads > 1:
            assert node_head is not None
            keep = self.hidden_heads.unsqueeze(0) == node_head.unsqueeze(1)
            middle = middle * keep.to(middle.dtype)
        return self.second(middle)
