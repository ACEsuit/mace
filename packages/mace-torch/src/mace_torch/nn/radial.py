"""Radial bases, cutoff envelopes, distance transforms and the radial MLP.

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

import math

import ase.data
import torch

__all__ = [
    "AgnesiTransform",
    "BesselBasis",
    "ChebyshevBasis",
    "GaussianBasis",
    "PolynomialCutoff",
    "RadialMLP",
    "SoftTransform",
    "ZBLBasis",
    "polynomial_envelope",
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
# Bases
# ---------------------------------------------------------------------------


class BesselBasis(torch.nn.Module):
    """Spherical Bessel functions of order zero, equation (7) of the MACE paper.

    ``f_n(r) = sqrt(2 / r_max) * sin(n * pi * r / r_max) / r`` for ``n = 1..num_basis``.

    The frequencies ``n * pi / r_max`` are a buffer, or a parameter when
    ``trainable`` is set.
    """

    frequencies: torch.Tensor
    r_max: torch.Tensor
    prefactor: torch.Tensor

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False):
        super().__init__()
        frequencies = (
            math.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(math.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    @property
    def num_basis(self) -> int:
        return len(self.frequencies)

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` distances in Angstrom -> ``[n_edges, num_basis]``."""
        numerator = torch.sin(self.frequencies * edge_lengths)
        return self.prefactor * (numerator / edge_lengths)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(r_max={self.r_max.item()}, "
            f"num_basis={self.num_basis}, trainable={self.frequencies.requires_grad})"
        )


class ChebyshevBasis(torch.nn.Module):
    """Chebyshev polynomials ``T_n`` of the raw input, ``--radial_type chebyshev``.

    ``T_0 = 1``, ``T_1 = x``, ``T_n = 2 x T_{n-1} - T_{n-2}``, evaluated by the
    three-term recurrence so the basis is differentiable to any order. Without
    the constant term the orders are ``1..num_basis``; with it, ``0..num_basis-1``.

    Both legacy classes collapse onto this one: the ``--radial_type chebyshev``
    basis is the default, and so is the magnetic family's moment basis, which
    legacy built as a second class with ``include_constant=False``. Nobody in
    the legacy tree uses the constant term.

    The input is not mapped into ``[-1, 1]``: legacy accepted and stored an
    ``r_max`` it never used, so a model trained with this basis sees the
    divergent ``cosh`` branch beyond 1 Angstrom. That is pinned by test.
    """

    def __init__(self, num_basis: int = 8, include_constant: bool = False):
        super().__init__()
        self.num_basis = num_basis
        self.include_constant = include_constant

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` -> ``[n_edges, num_basis]``.

        The magnetic family passes a transformed moment length in place of an
        edge length; the recurrence does not care which.
        """
        highest_order = self.num_basis - 1 if self.include_constant else self.num_basis
        previous = torch.ones_like(edge_lengths)
        current = edge_lengths
        polynomials = [previous, current]
        for _ in range(2, highest_order + 1):
            previous, current = current, 2.0 * edge_lengths * current - previous
            polynomials.append(current)
        first_order = 0 if self.include_constant else 1
        return torch.cat(
            polynomials[first_order : first_order + self.num_basis], dim=-1
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_basis={self.num_basis}, "
            f"include_constant={self.include_constant})"
        )


class GaussianBasis(torch.nn.Module):
    """Gaussians on evenly spaced centres in ``[0, r_max]``, ``--radial_type gaussian``.

    ``g_k(r) = exp(-0.5 * ((r - c_k) / w)^2)`` with centres
    ``c_k = linspace(0, r_max, num_basis)`` and width ``w = r_max / (num_basis - 1)``,
    folded into one coefficient ``-0.5 / w^2``. Never zero: only the cutoff envelope
    makes a long edge vanish.
    """

    centers: torch.Tensor

    def __init__(self, r_max: float, num_basis: int = 128, trainable: bool = False):
        super().__init__()
        centers = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.centers = torch.nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)
        self.exponent_coefficient = -0.5 / (r_max / (num_basis - 1)) ** 2
        self.num_basis = num_basis

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """``[n_edges, 1]`` -> ``[n_edges, num_basis]``."""
        offsets = edge_lengths - self.centers
        return torch.exp(self.exponent_coefficient * torch.pow(offsets, 2))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_basis={self.num_basis}, "
            f"trainable={self.centers.requires_grad})"
        )


# ---------------------------------------------------------------------------
# Cutoff envelope
# ---------------------------------------------------------------------------


def polynomial_envelope(
    edge_lengths: torch.Tensor, r_max: torch.Tensor, polynomial_order: int
) -> torch.Tensor:
    """The smooth envelope ``u(r)`` that is 1 at ``r = 0`` and 0 with two vanishing
    derivatives at ``r = r_max``, and exactly zero beyond.

    With ``x = r / r_max`` and ``p`` the order::

        u = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^(p+1) - p(p+1)/2 * x^(p+2)

    multiplied by the mask ``r < r_max``. The mask is what makes the padded-batch
    contract hold: a self-loop edge shifted by ``2 * r_max`` contributes exactly
    zero, not a small number. ``r_max`` may be a tensor broadcasting against
    ``edge_lengths``, which is how the ZBL term uses a per-pair radius.
    """
    order = float(polynomial_order)
    scaled = edge_lengths / r_max
    envelope = (
        1.0
        - ((order + 1.0) * (order + 2.0) / 2.0) * torch.pow(scaled, order)
        + order * (order + 2.0) * torch.pow(scaled, order + 1.0)
        - (order * (order + 1.0) / 2.0) * torch.pow(scaled, order + 2.0)
    )
    return envelope * (edge_lengths < r_max)


class PolynomialCutoff(torch.nn.Module):
    """The polynomial envelope as a module, sized by ``--num_cutoff_basis``."""

    r_max: torch.Tensor

    def __init__(self, r_max: float, polynomial_order: int = 6):
        super().__init__()
        self.polynomial_order = int(polynomial_order)
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:
        """Same shape as the input; values in ``[0, 1]``."""
        return polynomial_envelope(edge_lengths, self.r_max, self.polynomial_order)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(polynomial_order={self.polynomial_order}, "
            f"r_max={self.r_max.item()})"
        )


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
        coulomb = (
            14.3996 * sender_atomic_numbers * receiver_atomic_numbers
        ) / edge_lengths
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

    def __repr__(self) -> str:
        coefficients = ", ".join(
            f"{c:.5g}" for c in self.screening_coefficients.tolist()
        )
        return (
            f"{self.__class__.__name__}(screening_coefficients=[{coefficients}], "
            f"polynomial_order={self.polynomial_order})"
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(amplitude={self.amplitude.item():.4f}, "
            f"exponent_q={self.exponent_q.item():.4f}, "
            f"exponent_p={self.exponent_p.item():.4f})"
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(steepness={self.steepness.item():.4f})"


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
