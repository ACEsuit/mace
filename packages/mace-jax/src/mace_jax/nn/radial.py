"""Distances in Å; Bessel features and screened pair energies in eV."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def polynomial_envelope(lengths, cutoff, order: int):
    ratio = lengths / cutoff
    value = 1 - (order + 1) * (order + 2) / 2 * ratio**order
    value += order * (order + 2) * ratio ** (order + 1)
    value -= order * (order + 1) / 2 * ratio ** (order + 2)
    return jnp.where(lengths < cutoff, value, 0)


def bessel(lengths, frequencies, prefactor):
    safe = jnp.where(lengths == 0, 1, lengths)
    return prefactor * jnp.where(
        lengths == 0, frequencies, jnp.sin(lengths * frequencies) / safe
    )


def agnesi(lengths, atomic_numbers, edge_index, state):
    """ACEpotentials.jl Agnesi transform, JCP 2023, doi:10.1063/5.0158783."""
    radii = state("covalent_radii")[atomic_numbers[edge_index]]
    scaled = lengths / (0.5 * (radii[0] + radii[1]))[:, None]
    return 1 / (
        1
        + state("amplitude")
        * scaled ** state("exponent_q")
        / (1 + scaled ** (state("exponent_q") - state("exponent_p")))
    )


def zbl(lengths, atomic_numbers, edge_index, state, order: int):
    """Ziegler-Biersack-Littmark repulsion, summed per receiving atom, [N] eV."""
    left, right = atomic_numbers[edge_index].astype(lengths.dtype)
    radius = (
        state("screening_length_prefactor")
        * 0.529
        / (
            left ** state("screening_length_exponent")
            + right ** state("screening_length_exponent")
        )
    )
    screening = jnp.sum(
        state("screening_coefficients")
        * jnp.exp(
            -jnp.asarray([3.2, 0.9423, 0.4028, 0.2016], dtype=lengths.dtype)
            * lengths
            / radius[:, None]
        ),
        axis=-1,
    )
    radii = state("covalent_radii")[atomic_numbers[edge_index]]
    envelope = polynomial_envelope(lengths[:, 0], radii[0] + radii[1], order)
    energy = 0.5 * 14.3996 * left * right / lengths[:, 0] * screening * envelope
    return jax.ops.segment_sum(
        energy, edge_index[1], num_segments=atomic_numbers.shape[0]
    )
