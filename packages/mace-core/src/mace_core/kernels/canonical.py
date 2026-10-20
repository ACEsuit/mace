"""The canonical weight layout, which is what makes one checkpoint portable.

One layout means one checkpoint loads into any backend. The frozen tree has the
opposite: each backend stores weights its own way, and five conversion command
line tools exist to move between them.

The canonical form is stated here once:

* **Layout** is ``mul_ir``: the multiplicity index varies slowest.
* **Symmetric-contraction weights** are one flat ``[Z, A, mul]`` array over the
  reduced Clebsch-Gordan basis, in the path order
  :mod:`mace_core.clebsch_gordan` pins. ``Z`` is the element, ``A`` the path,
  ``mul`` the channel.
* **Conversion happens only at the checkpoint boundary.** ``to_canonical`` and
  ``load_canonical`` are called when saving and loading, never in ``forward``.
  A backend that wants another layout permutes once at build time and keeps the
  permuted copy.

The reference backend holds the canonical form directly, so for it both
functions are views.

**The canonical form holds raw weights, and the normalization lives in the
forward**, exactly where the frozen tree's ``e3nn`` operations keep it. A
canonical linear or skip-connection weight is the number ``e3nn`` stores,
drawn from a standard normal, and every backend multiplies it by a fixed
per-path factor when it computes. Folding the factor into the weight instead
gives the same forward and a gradient ``sqrt(fan_in)`` times larger, which is
a different effective learning rate for that weight and so a different
training run. The factors are part of the format, which is why they are stated
here and read by every backend:

* a **linear** map multiplies by ``1 / sqrt(fan_in)``, the total input
  multiplicity feeding the output irrep, which is ``e3nn``'s ``"element"``
  path normalization;
* the **skip connection's** fully connected tensor product multiplies by
  ``1 / sqrt(fan_in)`` where the fan-in is ``e3nn``'s: the sum of
  ``multiplicity_in1 * multiplicity_in2`` over every pair of input terms that
  couples to the output irrep, so every path landing on the same output shares
  one factor. The second input is scalars, the element attributes, so the
  ``sqrt(2l + 1)`` of ``"component"`` normalization cancels against the
  coupling of an irrep with a scalar and does not appear;
* the **symmetric contraction** multiplies by nothing.

A fresh draw follows the frozen tree too. Linear and skip-connection weights
are a standard normal and a bias is zero. A symmetric-contraction weight is a
standard normal divided by the number of paths in its ``(output irrep, body
order)`` block, which is a scale on the draw and not on the forward.

**The radial MLP** holds one dense ``[fan_in, fan_out]`` matrix per layer, the
orientation ``x @ W`` reads and the one ``e3nn`` stores; ``torch.nn.Linear``
holds the transpose. Its two variants keep their normalization where the frozen
tree keeps it:

* ``"normalized_silu"`` multiplies each layer's weight by ``1 / sqrt(fan_in)``
  in the forward, and each hidden activation by
  :data:`SILU_SECOND_MOMENT_SCALE`. A fresh weight is a standard normal. There
  is no bias.
* ``"layer_norm_silu"`` applies no factor. Each layer carries a bias, and each
  hidden layer a layer norm with epsilon :data:`LAYER_NORM_EPSILON`. A fresh
  weight and bias are uniform on ``[-1 / sqrt(fan_in), 1 / sqrt(fan_in)]``,
  ``torch.nn.Linear``'s draw, and a fresh layer norm is the identity.
"""

from __future__ import annotations

from mace_core.clebsch_gordan.irreps import Irrep, Irreps

__all__ = [
    "CANONICAL_LAYOUT",
    "KERNEL_SPEC_VERSION",
    "LAYER_NORM_EPSILON",
    "SILU_SECOND_MOMENT_SCALE",
    "canonical_weight_shape",
    "exact_silu_second_moment_scale",
    "fully_connected_tp_path_normalization",
    "linear_path_normalization",
    "radial_mlp_initial_bound",
    "radial_mlp_layer_normalization",
    "symmetric_contraction_initial_scale",
]

#: The version of this contract. A backend records it, and a checkpoint carries
#: it, so a format change is a loud mismatch rather than a silent misread.
KERNEL_SPEC_VERSION = "1.0"

#: The one layout a checkpoint is ever written in.
CANONICAL_LAYOUT = "mul_ir"

#: The multiplier that gives ``silu`` unit second moment on standard normal
#: input, as **every trained MACE artifact carries it**.
#:
#: It is not the exact value. ``e3nn`` estimates it from a million standard
#: normal samples under a fixed seed, and the exact value,
#: :func:`exact_silu_second_moment_scale`, is 1.6765324703310915: 0.16 per cent
#: away. That is far outside any tolerance these models are compared at, so the
#: estimate is part of the format. A model trained against it computes a
#: different function under the exact value.
SILU_SECOND_MOMENT_SCALE = 1.679176792398942

#: The epsilon of the ``"layer_norm_silu"`` radial MLP's layer norms, which is
#: ``torch.nn.LayerNorm``'s default and what the frozen tree trained with.
LAYER_NORM_EPSILON = 1e-5


def canonical_weight_shape(
    num_elements: int, path_count: int, num_features: int
) -> tuple[int, int, int]:
    """The shape of a symmetric-contraction weight array, ``[Z, A, mul]``.

    The per-``(output irrep, body order)`` blocks a backend computes with are
    contiguous slices of this one along the path axis, in the pinned order:
    body order outermost, output irrep within it. A backend that consumes them
    output irrep outermost, as the reference does, has to permute the blocks
    when it reads and writes this array; it is not a plain split.
    """
    return (num_elements, path_count, num_features)


def linear_path_normalization(irreps_in: str, irrep_out: Irrep) -> float:
    """The factor a linear map applies, in the forward, to a weight writing to
    ``irrep_out``.

    An equivariant linear map connects a term only to a term of the same irrep,
    so the fan-in of an output copy is the total multiplicity of the inputs
    that share its irrep, summed over every input term carrying it.

    Returns:
        ``1 / sqrt(fan_in)``, or ``1.0`` when nothing feeds the irrep, which is
        an output the map cannot produce and whose weights do not exist.
    """
    fan_in = sum(
        multiplicity
        for multiplicity, irrep in Irreps.parse(irreps_in)
        if irrep == irrep_out
    )
    return 1.0 if fan_in == 0 else float(fan_in) ** -0.5


def fully_connected_tp_path_normalization(
    irreps_in1: str, irreps_in2: str, irrep_out: Irrep
) -> float:
    """The factor the skip connection's tensor product applies, in the forward,
    to every path writing to ``irrep_out``.

    ``e3nn``'s ``"element"`` normalization: the fan-in sums
    ``multiplicity_in1 * multiplicity_in2`` over every pair of input terms whose
    product contains ``irrep_out``. A first input of ``4x0e+2x0e`` against
    ``3x0e`` therefore gives both paths into ``0e`` the factor ``1 / sqrt(18)``,
    not ``1 / sqrt(12)`` and ``1 / sqrt(6)`` each.

    Returns:
        ``1 / sqrt(fan_in)``, or ``1.0`` when no pair reaches the irrep.
    """
    second = Irreps.parse(irreps_in2)
    fan_in = sum(
        multiplicity_in1 * multiplicity_in2
        for multiplicity_in1, irrep_in1 in Irreps.parse(irreps_in1)
        for multiplicity_in2, irrep_in2 in second
        if irrep_out in set(irrep_in1.couple(irrep_in2))
    )
    return 1.0 if fan_in == 0 else float(fan_in) ** -0.5


def symmetric_contraction_initial_scale(path_count: int) -> float:
    """The factor a fresh symmetric-contraction weight block is drawn at.

    A standard normal divided by the number of paths in the ``(output irrep,
    body order)`` block, the frozen tree's draw. Unlike the two path
    normalizations above this is a scale on the draw, not on the forward.

    Returns:
        ``1 / path_count``, or ``1.0`` for a block with no paths, which has no
        weights for the scale to apply to.
    """
    return 1.0 if path_count == 0 else 1.0 / path_count


def exact_silu_second_moment_scale() -> float:
    """The multiplier that would give ``silu`` exactly unit second moment.

    Gauss-Hermite quadrature against the standard normal weight. It exists to
    be compared with :data:`SILU_SECOND_MOMENT_SCALE`, not to replace it.
    """
    import numpy as np

    nodes, weights = np.polynomial.hermite_e.hermegauss(200)
    weights = weights / np.sqrt(2.0 * np.pi)
    activated = nodes / (1.0 + np.exp(-nodes))
    return float(np.sum(weights * activated**2) ** -0.5)


def radial_mlp_layer_normalization(kind: str, fan_in: int) -> float:
    """The factor a radial MLP layer applies, in the forward, to its weight.

    Returns:
        ``1 / sqrt(fan_in)`` for ``"normalized_silu"``, and ``1.0`` for
        ``"layer_norm_silu"``, whose layer norms do that job instead.

    Raises:
        ValueError: For a kind that is neither, rather than a default factor.
    """
    if kind == "normalized_silu":
        return float(fan_in) ** -0.5
    if kind == "layer_norm_silu":
        return 1.0
    raise ValueError(f"{kind!r} is not a radial MLP kind")


def radial_mlp_initial_bound(fan_in: int) -> float:
    """The half-width of the uniform draw of a ``"layer_norm_silu"`` layer.

    ``1 / sqrt(fan_in)``, which is what ``torch.nn.Linear`` draws its weight
    and its bias from, and so what the frozen tree's layers started at.
    """
    return float(fan_in) ** -0.5
