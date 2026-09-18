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
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.observables import ObservableSpec
from torch import Tensor, nn

__all__ = ["ObservableHead", "channel_layout_index", "expanded_irreps"]


def expanded_irreps(hidden: str, num_features: int) -> str:
    """The declaration of ``num_features`` channels of ``hidden``, grouped.

    The node features are held as ``[n, channel, component]``. Read as one flat
    vector that is channel-major, and an equivariant linear map wants its input
    grouped by irrep instead. This is the grouped declaration;
    :func:`channel_layout_index` is the permutation that gets there.
    """
    parsed = Irreps.parse(hidden)
    return "+".join(f"{num_features * mul}x{ir}" for mul, ir in parsed.terms)


def channel_layout_index(hidden: str, num_features: int) -> np.ndarray:
    """Where each entry of the grouped layout reads from the channel-major one.

    Args:
        hidden: One channel's declaration.
        num_features: How many channels.

    Returns:
        ``[dim]`` of int64, to be used as ``flat[..., index]``.
    """
    parsed = Irreps.parse(hidden)
    width = parsed.dimension
    index = np.empty(num_features * width, dtype=np.int64)
    target = 0
    source_offset = 0
    for mul, ir in parsed.terms:
        span = ir.dimension
        for channel in range(num_features):
            for copy in range(mul):
                start = channel * width + source_offset + copy * span
                index[target : target + span] = np.arange(start, start + span)
                target += span
        source_offset += mul * span
    return index


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
    """Scalars through SiLU, everything else scaled by a sigmoid of its own.

    The standard equivariant gate. A pointwise nonlinearity is only equivariant
    on scalars, so a higher irrep is instead multiplied by a scalar, which
    leaves its direction alone and changes only its length.
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
        scalars = torch.nn.functional.silu(features[..., : self.scalar_dim])
        if not self.num_gates:
            return scalars
        gates = torch.sigmoid(
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
        hidden_irreps: One channel's node-feature declaration.
        num_features: The channel width.
        num_layers: How many layers of node features it will be given.
        nonlinear: Whether the last layer's readout carries a gate. The frozen
            tree makes exactly this choice, and only for the last layer.
        hidden_scalars: The width of the gated readout's middle.
        precision: The dtype every op is built at.
    """

    layout: Tensor

    def __init__(
        self,
        backend,
        spec: ObservableSpec,
        hidden_irreps: str,
        num_features: int,
        num_layers: int,
        nonlinear: bool = False,
        hidden_scalars: int = 16,
        precision: str = "float64",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.per_atom = spec.per_atom
        self.dimension = spec.dimension
        grouped = expanded_irreps(hidden_irreps, num_features)
        _check_reachable(spec, hidden_irreps)
        self.register_buffer(
            "layout",
            torch.tensor(channel_layout_index(hidden_irreps, num_features)),
            persistent=False,
        )

        readouts: list[nn.Module] = []
        for layer in range(num_layers):
            last = layer == num_layers - 1
            if nonlinear and last:
                readouts.append(
                    _GatedReadout(
                        backend, grouped, spec.irreps, hidden_scalars, precision
                    )
                )
            else:
                readouts.append(
                    backend.make_linear(
                        LinearDescriptor(
                            irreps_in=grouped,
                            irreps_out=spec.irreps,
                            precision=precision,
                        )
                    )
                )
        self.readouts = nn.ModuleList(readouts)

    def per_layer(self, layers: list[Tensor]) -> list[Tensor]:
        """This observable's contribution from each layer, unsummed.

        The energy path needs the terms apart, because the sum over layers and
        the sum over atoms are two different reductions and the energy head
        keeps them separate on purpose.
        """
        values = []
        for features, readout in zip(layers, self.readouts, strict=True):
            flat = features.reshape(features.shape[0], -1)[..., self.layout]
            values.append(readout(flat))
        return values

    def forward(self, layers: list[Tensor]) -> Tensor:
        """The per-atom value of this observable.

        Args:
            layers: One ``[n_atoms, channels, width]`` tensor per layer.

        Returns:
            ``[n_atoms, dimension]``. Reducing to graph level is the output
            layer's job, since it is the same reduction for every observable.
        """
        terms = self.per_layer(layers)
        total = terms[0]
        for value in terms[1:]:
            total = total + value
        return total


class _GatedReadout(nn.Module):
    """Linear, gate, linear. The frozen tree's nonlinear readout, rebuilt.

    The middle is scalars plus, when the output is not a scalar, one gate per
    non-scalar channel and the non-scalar terms themselves.
    """

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        hidden_scalars: int,
        precision: str,
    ) -> None:
        super().__init__()
        out = Irreps.parse(irreps_out)
        scalars = "+".join(
            f"{mul}x{ir}" for mul, ir in out.terms if ir.degree == 0 and ir.parity == 1
        )
        gated = "+".join(
            f"{mul}x{ir}"
            for mul, ir in out.terms
            if not (ir.degree == 0 and ir.parity == 1)
        )
        # The middle always carries scalars, whether or not the output does:
        # they are what the nonlinearity acts on, and what the gates are cut
        # from. `scalars` is read only to check the output is not empty.
        if not scalars and not gated:
            raise ValueError(f"{irreps_out!r} declares no irreps to read out.")
        self.gate = _Gate(f"{hidden_scalars}x0e", gated)
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
                irreps_out=irreps_out,
                precision=precision,
            )
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.second(self.gate(self.first(features)))
