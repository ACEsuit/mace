###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import collections
import itertools
import os
from typing import Iterator, List, Union

import numpy as np
import torch
from e3nn import o3

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

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


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

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

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
                return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(
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
