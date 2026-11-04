###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import collections
import functools
import itertools
import os
from fractions import Fraction
from math import factorial
from typing import Iterator, List, Union

import numpy as np
import torch
from e3nn import o3

from mace.tools.deprecation import warn_env

try:
    import cuequivariance as cue

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

USE_CUEQ_CG = os.environ.get("MACE_USE_CUEQ_CG", "0").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

warn_env("MACE_USE_CUEQ_CG")

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")
# ---------------------------------------------------------------------------------------
# Wigner 3j extension beyond e3nn 0.4.4's cached table
#
# e3nn 0.4.4 ships precomputed Wigner 3j tables only up to l1+l2+l3 combinations
# covered by its constants.pt (wigner_3j raises NotImplementedError beyond). MACE's
# symmetric-contraction CG recursion for correlation 3 over l=4 spherical harmonics
# needs couplings such as (4, 8, 12). The computation below (real Clebsch-Gordan via
# exact rational SU(2) coefficients) is ported from e3nn
# (e3nn/o3/_wigner.py, MIT license); its Clebsch-Gordan coefficient routine
# originates from QuTiP (BSD 3-clause). It reproduces e3nn 0.4.4's cached entries
# and extends to arbitrary l.
# ---------------------------------------------------------------------------------------


def _change_basis_real_to_complex(l: int) -> torch.Tensor:
    """Change-of-basis matrix from e3nn's real basis to the complex basis."""
    q = torch.zeros((2 * l + 1, 2 * l + 1), dtype=torch.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2**0.5
        q[l + m, l - abs(m)] = -1j / 2**0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2**0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2**0.5
    return (-1j) ** l * q


def _su2_clebsch_gordan_coeff(j1, m1, j2, m2, j3, m3) -> float:
    """SU(2) Clebsch-Gordan coefficient (QuTiP implementation, exact rationals)."""
    if m3 != m1 + m2:
        return 0.0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n: int) -> int:
        assert n == round(n)
        return factorial(round(n))

    c = (
        (2.0 * j3 + 1.0)
        * Fraction(
            f(j3 + j1 - j2)
            * f(j3 - j1 + j2)
            * f(j1 + j2 - j3)
            * f(j3 + m3)
            * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2),
        )
    ) ** 0.5

    s = 0
    for v in range(vmin, vmax + 1):
        s += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3),
        )
    return float(c * s)


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j1: int, j2: int, j3: int) -> torch.Tensor:
    mat = torch.zeros((2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1), dtype=torch.float64)
    if 2 * j3 in range(2 * abs(j1 - j2), 2 * (j1 + j2) + 1, 2):
        for m1 in range(-j1, j1 + 1):
            for m2 in range(-j2, j2 + 1):
                if abs(m1 + m2) <= j3:
                    mat[j1 + m1, j2 + m2, j3 + m1 + m2] = _su2_clebsch_gordan_coeff(
                        j1, m1, j2, m2, j3, m1 + m2
                    )
    return mat


@functools.lru_cache(maxsize=None)
def _compute_wigner_3j(l1: int, l2: int, l3: int) -> torch.Tensor:
    """Wigner 3j in e3nn's real basis, computed for arbitrary l."""
    q1 = _change_basis_real_to_complex(l1)
    q2 = _change_basis_real_to_complex(l2)
    q3 = _change_basis_real_to_complex(l3)
    c = _su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
    c = torch.einsum("ij,kl,mn,ikn->jlm", q1, q2, torch.conj(q3.T), c)
    assert torch.all(torch.abs(torch.imag(c)) < 1e-5)
    return torch.real(c) / torch.norm(c)


def _wigner_3j(l1: int, l2: int, l3: int, dtype=None):
    """o3.wigner_3j with an on-demand fallback beyond e3nn 0.4.4's cached table."""
    try:
        return o3.wigner_3j(l1, l2, l3, dtype=dtype)
    except NotImplementedError:
        cached = _compute_wigner_3j(l1, l2, l3)
        out = cached.to(dtype=torch.get_default_dtype() if dtype is None else dtype)
        if out is cached:
            # Same dtype: .to() hands back the lru_cache-shared tensor. Never
            # let callers mutate the process-global cache in place.
            out = cached.clone()
        return out


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = _wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C = C * ir_out.dim**0.5
                if normalization == "norm":
                    C = C * ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim
                )
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
    use_cueq_cg=True,
    use_nonsymmetric_product=False,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation

    if use_cueq_cg is None:
        use_cueq_cg = USE_CUEQ_CG
    if correlation == 4 and not use_cueq_cg:
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]
    if use_cueq_cg and CUET_AVAILABLE:
        return compute_U_cueq(  # pylint: disable=possibly-used-before-assignment
            irreps_in, irreps_out=irreps_out, correlation=correlation, dtype=dtype
        )

    try:
        wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    except NotImplementedError as e:
        if CUET_AVAILABLE:
            return compute_U_cueq(  # pylint: disable=possibly-used-before-assignment
                irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                use_nonsymmetric_product=use_nonsymmetric_product,
                dtype=dtype,
            )
        raise NotImplementedError(
            "The requested Clebsch-Gordan coefficients are not implemented, please install cuequivariance; pip install cuequivariance"
        ) from e

    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    try:
        out += [last_ir, stack]
    except:  # pylint: disable=bare-except
        first_dim = irreps_out.dim
        if first_dim != 1:
            size = [first_dim] + [o3.Irreps(irreps_in).dim] * correlation + [1]
        else:
            size = [o3.Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
    return out


if CUET_AVAILABLE:

    def compute_U_cueq(
        irreps_in, irreps_out, correlation=2, use_nonsymmetric_product=False, dtype=None
    ):
        if dtype is None:
            dtype = torch.get_default_dtype()
        U = []
        irreps_in = cue.Irreps(O3_e3nn, str(irreps_in))
        irreps_out = cue.Irreps(O3_e3nn, str(irreps_out))
        for _, ir in irreps_out:
            try:
                U_matrix_full_symm = cue.reduced_symmetric_tensor_product_basis(
                    irreps_in,
                    correlation,
                    keep_ir=ir,
                    layout=cue.ir_mul,
                )
                U_matrix_full_symm = U_matrix_full_symm.array
                if use_nonsymmetric_product:
                    try:
                        U_matrix_full_antisymmetric = (
                            cue.reduced_antisymmetric_tensor_product_basis(
                                irreps_in,
                                correlation,
                                keep_ir=ir,
                                layout=cue.ir_mul,
                            ).array
                        )
                        U_matrix_full = torch.cat(
                            (U_matrix_full_symm, U_matrix_full_antisymmetric), dim=-1
                        )
                    except ValueError:
                        continue
                else:
                    U_matrix_full = U_matrix_full_symm

            except ValueError:
                if ir.dim == 1:
                    out_shape = (*([irreps_in.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in.dim] * correlation), 1)
                return [
                    torch.zeros(
                        out_shape,
                        dtype=torch.get_default_dtype(),
                    )
                ]
            if U_matrix_full.shape[-1] == 0:
                if ir.dim == 1:
                    out_shape = (*([irreps_in.dim] * correlation), 1)
                else:
                    out_shape = (ir.dim, *([irreps_in.dim] * correlation), 1)
                return [
                    torch.zeros(
                        out_shape,
                        dtype=torch.get_default_dtype(),
                    )
                ]
            ir_str = str(ir)
            U.append(ir_str)
            U_matrix_full = torch.tensor(
                U_matrix_full.reshape(*([irreps_in.dim] * correlation), ir.dim, -1),
                dtype=dtype,
            )
            U_matrix_full = torch.moveaxis(U_matrix_full, -2, 0)
            if ir.dim == 1:
                U_matrix_full = U_matrix_full[0]
            U.append(U_matrix_full)
        return U

    class O3_e3nn(cue.O3):
        def __mul__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> Iterator["O3_e3nn"]:
            return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

        @classmethod
        def clebsch_gordan(
            cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
        ) -> np.ndarray:
            rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

            if rep1.p * rep2.p == rep3.p:
                return _wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
                    rep3.dim
                )
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

        def __lt__(  # pylint: disable=no-self-argument
            rep1: "O3_e3nn", rep2: "O3_e3nn"
        ) -> bool:
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)

else:

    class O3_e3nn:
        pass

    print(
        "cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled."
    )
