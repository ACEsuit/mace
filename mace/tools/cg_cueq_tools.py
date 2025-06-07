# This coded was modified from the cuequivariance library: https://github.com/NVIDIA/cuEquivariance/blob/main/cuequivariance/cuequivariance/group_theory/experimental/mace/symmetric_contractions.py
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Optional

import numpy as np
from e3nn import o3

from mace.tools.cg import U_matrix_real

try:
    import cuequivariance as cue
    from cuequivariance.etc.linalg import round_to_sqrt_rational, triu_array

    CUEQQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQQ_AVAILABLE = False

    class DummyCueq:
        class EquivariantPolynomial:
            pass

        class Irreps:
            pass

    cue = DummyCueq()
    round_to_sqrt_rational = None
    triu_array = None


def symmetric_contraction_proj(
    irreps_in: cue.Irreps, irreps_out: cue.Irreps, degrees: tuple[int, ...]
) -> tuple[cue.EquivariantPolynomial, np.ndarray]:
    r"""
    subscripts: ``weights[u],input[u],output[u]``

    Example:

    .. code-block:: python

        e, p = symmetric_contraction(
            4 * cue.Irreps("SO3", "0+1+2"), 4 * cue.Irreps("SO3", "0+1"), [1, 2, 3]
        )
        assert p.shape == (62, 18)

        mul = e.inputs[1].irreps.muls[0]
        w = jax.random.normal(jax.random.key(0), (p.shape[0], mul))
        w = jnp.einsum("au,ab->bu", w, p).flatten()

        x = cuex.randn(jax.random.key(1), e.inputs[1])
        y = cuex.equivariant_polynomial(e, [w, x])
    """
    return symmetric_contraction_cached(irreps_in, irreps_out, tuple(degrees))


def symmetric_contraction_cached(
    irreps_in: cue.Irreps, irreps_out: cue.Irreps, degrees: tuple[int, ...]
) -> tuple[cue.EquivariantPolynomial, np.ndarray]:
    assert min(degrees) > 0

    # poly1 replicates the behavior of the original MACE implementation
    poly1 = cue.EquivariantPolynomial.stack(
        [
            cue.EquivariantPolynomial.stack(
                [
                    _symmetric_contraction(irreps_in, irreps_out[i : i + 1], deg)
                    for deg in reversed(degrees)
                ],
                [True, False, False],
            )
            for i in range(len(irreps_out))
        ],
        [True, False, True],
    )

    poly2 = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, degrees)
    a1, a2 = [
        np.concatenate(
            [
                _flatten(
                    _stp_to_matrix(d.symmetrize_operands(range(1, d.num_operands - 1))),
                    1,
                    None,
                )
                for _, d in pol.polynomial.operations
            ],
            axis=1,
        )
        for pol in [poly1, poly2]
    ]

    # This nonzeros selection is just for lightening the inversion
    nonzeros = np.nonzero(np.any(a1 != 0, axis=0) | np.any(a2 != 0, axis=0))[0]
    a1, a2 = a1[:, nonzeros], a2[:, nonzeros]
    projection_1_2 = a1 @ np.linalg.pinv(a2)
    # projection = np.linalg.lstsq(a2.T, a1.T, rcond=None)[0].T
    projection_1_2 = round_to_sqrt_rational(projection_1_2)
    np.testing.assert_allclose(a1, projection_1_2 @ a2, atol=1e-7)
    return poly2, projection_1_2


def _flatten(
    x: np.ndarray, axis_start: Optional[int] = None, axis_end: Optional[int] = None
) -> np.ndarray:
    x = np.asarray(x)
    if axis_start is None:
        axis_start = 0
    if axis_end is None:
        axis_end = x.ndim
    assert 0 <= axis_start <= axis_end <= x.ndim
    return x.reshape(
        x.shape[:axis_start]
        + (np.prod(x.shape[axis_start:axis_end]),)
        + x.shape[axis_end:]
    )


def _stp_to_matrix(
    d: cue.SegmentedTensorProduct,
) -> np.ndarray:
    m = np.zeros([ope.num_segments for ope in d.operands])
    for path in d.paths:
        m[path.indices] = path.coefficients
    return m


# This function is an adaptation of https://github.com/ACEsuit/mace/blob/bd412319b11c5f56c37cec6c4cfae74b2a49ff43/mace/modules/symmetric_contraction.py
def _symmetric_contraction(
    irreps_in: cue.Irreps, irreps_out: cue.Irreps, degree: int
) -> cue.EquivariantPolynomial:
    mul = irreps_in.muls[0]
    assert all(mul == m for m in irreps_in.muls)
    assert all(mul == m for m in irreps_out.muls)
    irreps_in = irreps_in.set_mul(1)
    irreps_out = irreps_out.set_mul(1)

    input_operands = range(1, degree + 1)
    output_operand = degree + 1

    abc = "abcdefgh"[:degree]
    d = cue.SegmentedTensorProduct.from_subscripts(
        f"u_{'_'.join(f'{a}' for a in abc)}_i+{abc}ui"
    )

    for i in input_operands:
        d.add_segment(i, (irreps_in.dim,))
    irreps_out_e3nn = o3.Irreps(str(irreps_out))
    irreps_in_e3nn = o3.Irreps(str(irreps_in))
    for _, ir in irreps_out:
        u = U_matrix_real(irreps_in_e3nn, irreps_out_e3nn, degree)[-1]
        if str(ir) == "0e" or str(ir) == "0o":
            u = u.unsqueeze(0)
        u = np.asarray(u)
        u = np.moveaxis(u, 0, -1)
        # u is shape (irreps_in.dim, ..., irreps_in.dim, u, ir_out.dim)

        if u.shape[-2] == 0:
            d.add_segment(output_operand, {"i": ir.dim})
        else:
            u = triu_array(u, degree)
            d.add_path(None, *(0,) * degree, None, c=u)

    d = d.flatten_coefficient_modes()
    d = d.append_modes_to_all_operands("u", {"u": mul})

    assert d.num_operands >= 3
    [w, x], y = d.operands[:2], d.operands[-1]

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in.new_scalars(w.size), cue.ir_mul),
            cue.IrrepsAndLayout(mul * irreps_in, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(mul * irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial(
            [w, x], [y], [(cue.Operation([0] + [1] * degree + [2]), d)]
        ),
    )
