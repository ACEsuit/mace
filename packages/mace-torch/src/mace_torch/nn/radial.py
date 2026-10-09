"""Pair repulsion, distance transforms and the radial MLP.

Everything in this module is pure mathematics on interatomic distances: a
distance in Angstrom in, numbers out, with no graph beyond the element pair an
edge connects. The behavioural specification is the legacy characterization
suite ported to ``packages/mace-torch/tests/test_mace_torch_radial.py``; the
numbers there are the closed forms, committed as decimal literals.

Conventions shared by every class:

* distances are in Angstrom and enter as a column tensor ``[n_edges, 1]``;
* buffers are created in ``torch.get_default_dtype()`` at construction, so a
  module built under ``float64`` computes in ``float64``;
* per-edge element information arrives as ``node_atomic_numbers`` (``int64``,
  ``[n_nodes]``) plus ``edge_index`` (``int64``, ``[2, n_edges]``, row 0 the
  sender, row 1 the receiver). Nothing here reads a one-hot encoding.
"""

from __future__ import annotations

import ase.data
import torch

from mace_torch.backends.radial import polynomial_envelope

__all__ = [
    "AgnesiTransform",
    "RadialMLP",
    "SoftTransform",
    "ZBLBasis",
]


def _covalent_radii_buffer() -> torch.Tensor:
    """ASE's covalent radii in Angstrom, indexed by atomic number."""
    return torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype())


def _edge_atomic_numbers(
    node_atomic_numbers: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atomic numbers of every edge's sender and receiver, each ``[n_edges, 1]``."""
    sender_atomic_numbers = node_atomic_numbers[edge_index[0]].to(torch.int64)
    receiver_atomic_numbers = node_atomic_numbers[edge_index[1]].to(torch.int64)
    return sender_atomic_numbers.unsqueeze(-1), receiver_atomic_numbers.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Pair repulsion
# ---------------------------------------------------------------------------


class ZBLBasis(torch.nn.Module):
    """The Ziegler-Biersack-Littmark screened Coulomb repulsion, per receiving node.

    For a directed edge between atomic numbers ``Z_u`` (sender) and ``Z_v``
    (receiver) at distance ``r`` in Angstrom::

        a      = a_prefactor * 0.529 / (Z_u^a_exp + Z_v^a_exp)       screening length
        phi(x) = 0.1818 e^{-3.2x} + 0.5099 e^{-0.9423x}
               + 0.2802 e^{-0.4028x} + 0.02817 e^{-0.2016x}
        V(r)   = 1/2 * (14.3996 * Z_u * Z_v / r) * phi(r / a) * envelope(r)

    in eV, with ``14.3996 eV*Angstrom`` the Coulomb constant. The envelope is the
    polynomial cutoff with ``r_max`` the *pair's* covalent radii sum, not the
    model cutoff, so the term is exactly zero for ordinary bond lengths. The
    factor 1/2 is because the sum runs over directed edges, every pair twice.
    The return is scattered onto the receiver, ``[n_nodes]``.

    Where the term enters relative to scale and shift is the model's business,
    not this class's.
    """

    screening_coefficients: torch.Tensor
    screening_exponents: torch.Tensor
    covalent_radii: torch.Tensor
    screening_length_exponent: torch.Tensor
    screening_length_prefactor: torch.Tensor

    def __init__(self, polynomial_order: int = 6, trainable: bool = False):
        super().__init__()
        self.polynomial_order = int(polynomial_order)
        self.register_buffer(
            "screening_coefficients",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer(
            "screening_exponents",
            torch.tensor(
                [3.2, 0.9423, 0.4028, 0.2016], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("covalent_radii", _covalent_radii_buffer())
        screening_length_exponent = torch.tensor(0.300, dtype=torch.get_default_dtype())
        screening_length_prefactor = torch.tensor(
            0.4543, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.screening_length_exponent = torch.nn.Parameter(
                screening_length_exponent
            )
            self.screening_length_prefactor = torch.nn.Parameter(
                screening_length_prefactor
            )
        else:
            self.register_buffer("screening_length_exponent", screening_length_exponent)
            self.register_buffer(
                "screening_length_prefactor", screening_length_prefactor
            )

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """``[n_edges, 1]`` distances -> ``[n_nodes]`` repulsion energies in eV."""
        sender_atomic_numbers, receiver_atomic_numbers = _edge_atomic_numbers(
            node_atomic_numbers, edge_index
        )
        screening_length = (
            self.screening_length_prefactor
            * 0.529
            / (
                torch.pow(sender_atomic_numbers, self.screening_length_exponent)
                + torch.pow(receiver_atomic_numbers, self.screening_length_exponent)
            )
        )
        reduced_distance = edge_lengths / screening_length
        screening = torch.sum(
            self.screening_coefficients
            * torch.exp(-self.screening_exponents * reduced_distance),
            dim=-1,
            keepdim=True,
        )
        # An integer tensor times a Python float takes the default dtype, not
        # the model's: cast the charge product before the constant meets it.
        charge_product = (sender_atomic_numbers * receiver_atomic_numbers).to(
            edge_lengths.dtype
        )
        coulomb = 14.3996 * charge_product / edge_lengths
        pair_r_max = (
            self.covalent_radii[sender_atomic_numbers]
            + self.covalent_radii[receiver_atomic_numbers]
        )
        envelope = polynomial_envelope(edge_lengths, pair_r_max, self.polynomial_order)
        edge_energies = 0.5 * coulomb * screening * envelope  # [n_edges, 1]
        node_energies = torch.zeros(
            node_atomic_numbers.shape[0],
            dtype=edge_energies.dtype,
            device=edge_energies.device,
        )
        return node_energies.index_add(0, edge_index[1], edge_energies.squeeze(-1))

    def extra_repr(self) -> str:
        return (
            f"polynomial_order={self.polynomial_order}, "
            f"trainable={self.screening_length_exponent.requires_grad}"
        )


# ---------------------------------------------------------------------------
# Distance transforms
# ---------------------------------------------------------------------------


class AgnesiTransform(torch.nn.Module):
    """The Agnesi distance transform of ACEpotentials.jl (JCP 2023, 10.1063/5.0158783).

    With ``r_0`` half the covalent radii sum of the pair and ``y = r / r_0``::

        T(r) = 1 / (1 + a * y^q / (1 + y^(q - p)))

    Monotonically compressing: larger ``r`` maps to a smaller transformed value,
    which is why it is applied to the lengths *before* the basis and never to
    the cutoff. The three parameters are the paper's ``a``, ``q`` and ``p``.
    """

    exponent_q: torch.Tensor
    exponent_p: torch.Tensor
    amplitude: torch.Tensor
    covalent_radii: torch.Tensor

    def __init__(
        self,
        exponent_q: float = 0.9183,
        exponent_p: float = 4.5791,
        amplitude: float = 1.0805,
        trainable: bool = False,
    ):
        super().__init__()
        dtype = torch.get_default_dtype()
        parameters = {
            "exponent_q": torch.tensor(exponent_q, dtype=dtype),
            "exponent_p": torch.tensor(exponent_p, dtype=dtype),
            "amplitude": torch.tensor(amplitude, dtype=dtype),
        }
        for name, value in parameters.items():
            if trainable:
                setattr(self, name, torch.nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        self.register_buffer("covalent_radii", _covalent_radii_buffer())

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """``[n_edges, 1]`` distances -> ``[n_edges, 1]`` transformed distances."""
        sender_atomic_numbers, receiver_atomic_numbers = _edge_atomic_numbers(
            node_atomic_numbers, edge_index
        )
        pair_radius = 0.5 * (
            self.covalent_radii[sender_atomic_numbers]
            + self.covalent_radii[receiver_atomic_numbers]
        )
        scaled = edge_lengths / pair_radius
        return torch.reciprocal(
            1.0
            + self.amplitude
            * torch.pow(scaled, self.exponent_q)
            / (1.0 + torch.pow(scaled, self.exponent_q - self.exponent_p))
        )

    def extra_repr(self) -> str:
        return (
            f"amplitude={self.amplitude.item():.4f}, "
            f"exponent_q={self.exponent_q.item():.4f}, "
            f"exponent_p={self.exponent_p.item():.4f}"
        )


class SoftTransform(torch.nn.Module):
    """A tanh clamp: short distances flatten onto ``p_0``, long ones pass unchanged.

    With ``r_0`` the covalent radii sum of the pair, ``p_0 = 3/4 r_0``,
    ``p_1 = 4/3 r_0``, midpoint ``m = (p_0 + p_1) / 2`` and steepness
    ``alpha / (p_1 - p_0)``::

        T(r) = p_0 + (r - p_0) * 0.5 * (1 + tanh(alpha' * (r - m)))

    Not globally monotone: just below ``p_0`` it dips by about 5e-4 before
    recovering. That wrinkle is legacy behaviour and is pinned by test.
    """

    steepness: torch.Tensor
    covalent_radii: torch.Tensor

    def __init__(self, steepness: float = 4.0, trainable: bool = False):
        super().__init__()
        steepness_tensor = torch.tensor(steepness, dtype=torch.get_default_dtype())
        if trainable:
            self.steepness = torch.nn.Parameter(steepness_tensor)
        else:
            self.register_buffer("steepness", steepness_tensor)
        self.register_buffer("covalent_radii", _covalent_radii_buffer())

    def forward(
        self,
        edge_lengths: torch.Tensor,
        node_atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """``[n_edges, 1]`` distances -> ``[n_edges, 1]`` transformed distances."""
        sender_atomic_numbers, receiver_atomic_numbers = _edge_atomic_numbers(
            node_atomic_numbers, edge_index
        )
        pair_radius = (
            self.covalent_radii[sender_atomic_numbers]
            + self.covalent_radii[receiver_atomic_numbers]
        )
        lower_clamp = 0.75 * pair_radius
        upper_point = (4.0 / 3.0) * pair_radius
        midpoint = 0.5 * (lower_clamp + upper_point)
        steepness = self.steepness / (upper_point - lower_clamp)
        switch = 0.5 * (1.0 + torch.tanh(steepness * (edge_lengths - midpoint)))
        return lower_clamp + (edge_lengths - lower_clamp) * switch

    def extra_repr(self) -> str:
        return f"steepness={self.steepness.item():.4f}"


# ---------------------------------------------------------------------------
# Radial MLP
# ---------------------------------------------------------------------------


class RadialMLP(torch.nn.Module):
    """``Linear -> LayerNorm -> SiLU`` stacks over radial features; ``--radial_MLP``.

    ``channels`` lists the widths, input first. No normalisation or activation
    follows the last linear layer, so the output is unbounded.
    """

    def __init__(self, channels: list[int]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_channels = channels[0]
        for index, out_channels in enumerate(channels[1:], start=1):
            layers.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if index < len(channels) - 1:
                layers.append(torch.nn.LayerNorm(out_channels))
                layers.append(torch.nn.SiLU())
        self.layers = torch.nn.Sequential(*layers)
        self.channels = list(channels)

    def forward(self, radial_features: torch.Tensor) -> torch.Tensor:
        """``[n_edges, channels[0]]`` -> ``[n_edges, channels[-1]]``."""
        return self.layers(radial_features)
