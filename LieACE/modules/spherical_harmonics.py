import math
from sympy import re
import torch
from numpy import pi as np_pi


class SphericalHarmonics(torch.nn.Module):
    def __init__(self, lmax, dtype=torch.float64, device="cpu"):
        super().__init__()
        self._lmax = lmax
        self.device = device
        # self.rhat = rhat
        self.prec = Precision(dtype)
        self.alm, self.blm = self.pre_compute()
        self.pi = self.float_tensor(np_pi)

    def lm1d(self, l, m):
        return m + l * (l + 1) // 2

    def lmsh(self, l, m):
        return l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2

    def lm_m(self, l, m):
        return l * (l + 1) + m

    def float_tensor(self, x):
        return torch.as_tensor(x, dtype=self.prec.float, device=self.device)

    def int_tensor(self, x):
        return torch.as_tensor(x, dtype=self.prec.int, device=self.device)

    def pre_compute(self):
        alm = [self.float_tensor(0.0)]
        blm = [self.float_tensor(0.0)]
        lindex = torch.arange(
            0, self._lmax + 1, dtype=self.prec.float, device=self.device
        )
        for i in range(1, self._lmax + 1):
            l = lindex[i]
            lsq = l * l
            ld = 2 * l
            l1 = 4 * lsq - 1
            l2 = lsq - ld + 1
            for j in range(0, i + 1):
                m = lindex[j]
                msq = m * m
                a = torch.sqrt(l1 / (lsq - msq))
                b = -torch.sqrt((l2 - msq) / (4 * l2 - 1))
                if i == j:
                    cl = -torch.sqrt(1.0 + 0.5 / m)
                    alm += [cl]
                    blm += [torch.tensor(0, device=self.device)]  # placeholder
                else:
                    alm += [a]
                    blm += [b]

        return torch.stack(alm), torch.stack(blm)

    def legendre(self, x):
        x = self.float_tensor(x)

        y00 = 1.0 * torch.sqrt(1.0 / (4.0 * self.pi))
        plm = [x * 0 + y00]
        if self._lmax > 0:
            sq3o4pi = torch.sqrt(3.0 / (4.0 * self.pi))
            sq3o8pi = torch.sqrt(3.0 / (8.0 * self.pi))

            plm += [sq3o4pi * x]  # (1,0)
            plm += [x * 0 - sq3o8pi]  # (1,1)

            for l in range(2, self._lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = torch.sqrt(2.0 * m + self.float_tensor(3.0))
                        plm += [x * dl * plm[self.lm1d(l - 1, l - 1)]]
                    elif m == l:
                        plm += [
                            self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)]
                        ]
                    else:
                        # plm += [0.]
                        plm += [
                            self.alm[self.lm1d(l, m)]
                            * (
                                x * plm[self.lm1d(l - 1, m)]
                                + self.blm[self.lm1d(l, m)] * plm[self.lm1d(l - 2, m)]
                            )
                        ]

        plm = torch.stack(plm)
        return plm

    def compute_ylm_sqr(self, rhat):
        rhat = self.float_tensor(rhat)

        e_x = self.float_tensor([[1.0], [0.0], [0.0]])
        e_y = self.float_tensor([[0.0], [1.0], [0.0]])
        e_z = self.float_tensor([[0.0], [0.0], [1.0]])

        rx = torch.matmul(rhat, e_x)
        ry = torch.matmul(rhat, e_y)
        rz = torch.matmul(rhat, e_z)

        phase_r = rx
        phase_i = ry

        ylm_r = []
        ylm_i = []
        plm = self.legendre(rz)

        m = 0
        for l in range(0, self._lmax + 1):
            ylm_r += [plm[self.lm1d(l, m)]]
            ylm_i += [self.float_tensor(torch.zeros_like(plm[self.lm1d(l, m)]))]

        m = 1
        for l in range(1, self._lmax + 1):
            ylm_r += [phase_r * plm[self.lm1d(l, m)]]
            ylm_i += [phase_i * plm[self.lm1d(l, m)]]

        phasem_r = phase_r
        phasem_i = phase_i
        for m in range(2, self._lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            for l in range(m, self._lmax + 1):
                ylm_r += [phasem_r * plm[self.lm1d(l, m)]]
                ylm_i += [phasem_i * plm[self.lm1d(l, m)]]

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self._lmax + 1):
            for m in range(-self._lmax, self._lmax + 1):
                ph = torch.where(
                    torch.equal(torch.abs(m) % 2, 0),
                    torch.tensor(1, dtype=self.prec.float),
                    torch.tensor(-1, dtype=self.prec.float),
                )
                # ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
                ph_r = ph
                ph_i = torch.zeros_like(
                    ph, dtype=self.prec.float
                )  # torch.constant(0, dtype=self.prec.float)
                if m < 0 and abs(m) > l:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]
                elif m < 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    # sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [torch.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]

        ylm_r = torch.transpose(
            torch.stack(sqr_ylm_r), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = torch.transpose(
            torch.stack(sqr_ylm_i), [1, 0, 2]
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]

        # return torch.reshape(ylm, [-1, self._lmax + 1, 2 * self._lmax + 1])  ##[None, nYlm, 1] -> [None, l, m]
        return (
            torch.reshape(ylm_r, [-1, self._lmax + 1, 2 * self._lmax + 1]),
            torch.reshape(ylm_i, [-1, self._lmax + 1, 2 * self._lmax + 1]),
        )  ##[None, nYlm, 1] -> [None, l, m]

    def forward(self, rhat):
        rhat = self.float_tensor(rhat)
        self.l_tile = []
        e_x = self.float_tensor([[1.0], [0.0], [0.0]])
        e_y = self.float_tensor([[0.0], [1.0], [0.0]])
        e_z = self.float_tensor([[0.0], [0.0], [1.0]])

        rx = torch.matmul(rhat, e_x)
        ry = torch.matmul(rhat, e_y)
        rz = torch.matmul(rhat, e_z)

        phase_r = rx
        phase_i = ry

        ylm_r = []
        ylm_i = []
        plm = self.legendre(rz)

        m = 0
        for l in range(0, self._lmax + 1):
            ylm_r += [plm[self.lm1d(l, m)]]
            ylm_i += [self.float_tensor(torch.zeros_like(plm[self.lm1d(l, m)]))]

        m = 1
        for l in range(1, self._lmax + 1):
            ylm_r += [phase_r * plm[self.lm1d(l, m)]]
            ylm_i += [phase_i * plm[self.lm1d(l, m)]]

        phasem_r = phase_r
        phasem_i = phase_i
        for m in range(2, self._lmax + 1):
            pr_tmp = phasem_r
            phasem_r = phasem_r * phase_r - phasem_i * phase_i
            phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            for l in range(m, self._lmax + 1):
                ylm_r += [phasem_r * plm[self.lm1d(l, m)]]
                ylm_i += [phasem_i * plm[self.lm1d(l, m)]]

        sqr_ylm_r = []
        sqr_ylm_i = []
        for l in range(0, self._lmax + 1):
            for m in range(-self._lmax, self._lmax + 1):
                ph = torch.where(
                    torch.tensor(
                        torch.equal(torch.abs(torch.tensor(m)) % 2, torch.tensor(0))
                    ),
                    torch.tensor(1, dtype=self.prec.float),
                    torch.tensor(-1, dtype=self.prec.float),
                )
                # ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
                ph_r = ph
                if m < 0 and abs(m) > l:
                    pass
                elif m < 0 and abs(m) <= l:
                    self.l_tile += [torch.tensor(l)]
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    # sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [torch.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    self.l_tile += [torch.tensor(l)]
                    ind = self.lmsh(
                        l, m
                    )  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    pass
        self.l_tile = torch.stack(self.l_tile)
        # ylm_r = torch.transpose(torch.stack(sqr_ylm_r), [1, 0, 2])
        ylm_r = torch.stack(sqr_ylm_r).permute(
            1, 0, 2
        )  ##[nYlm, None, 1] -> [None, nYlm, 1]
        # ylm_i = torch.transpose(torch.stack(sqr_ylm_i), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = torch.stack(sqr_ylm_i).permute(1, 0, 2)

        ylm_r = torch.as_tensor(ylm_r).flatten(1, -1)
        ylm_i = torch.as_tensor(ylm_i).flatten(1, -1)
        ylm = torch.stack((ylm_r, ylm_i), dim=-1)
        return torch.view_as_complex(ylm)

    def compute_ylm(self, l, m, rhat):
        rhat = self.float_tensor(rhat)
        self.l_tile = []
        self.l = l
        self.m = m
        e_x = self.float_tensor([[1.0], [0.0], [0.0]])
        e_y = self.float_tensor([[0.0], [1.0], [0.0]])
        e_z = self.float_tensor([[0.0], [0.0], [1.0]])

        rx = torch.matmul(rhat, e_x)
        ry = torch.matmul(rhat, e_y)
        rz = torch.matmul(rhat, e_z)

        phase_r = rx
        phase_i = ry

        plm = self.legendre(rz)

        if self.m == 0:

            ylm_r = plm[self.lm1d(l, m)]
            ylm_i = self.float_tensor(torch.zeros_like(plm[self.lm1d(l, m)]))

        elif abs(self.m) == 1:

            m1 = abs(self.m)

            ylm_r = phase_r * plm[self.lm1d(l, m1)]
            ylm_i = phase_i * plm[self.lm1d(l, m1)]

        else:
            phasem_r = phase_r
            phasem_i = phase_i
            m1 = abs(self.m)
            for n in range(2, m1 + 1):
                pr_tmp = phasem_r
                phasem_r = phasem_r * phase_r - phasem_i * phase_i
                phasem_i = pr_tmp * phase_i + phasem_i * phase_r

            ylm_r = phasem_r * plm[self.lm1d(l, m1)]
            ylm_i = phasem_i * plm[self.lm1d(l, m1)]

        ph = torch.where(
            torch.tensor(torch.equal(torch.abs(torch.tensor(m)) % 2, torch.tensor(0))),
            torch.tensor(1, dtype=self.prec.float),
            torch.tensor(-1, dtype=self.prec.float),
        )
        # ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
        ph_r = ph
        if m < 0 and abs(m) > l:
            pass
        elif m < 0 and abs(m) <= l:
            self.l_tile = torch.tensor(l)
            # ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
            # sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]

            sqr_ylm_r = ylm_r * ph_r
            sqr_ylm_i = torch.negative(ylm_i) * ph_r

        elif m >= 0 and abs(m) <= l:
            self.l_tile = torch.tensor(l)
            # ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
            sqr_ylm_r = ylm_r
            sqr_ylm_i = ylm_i
        else:
            pass

        # self.l_tile = torch.stack(self.l_tile)
        # ylm_r = torch.transpose(torch.stack(sqr_ylm_r), [1, 0, 2])
        ylm_r = sqr_ylm_r  ##[nYlm, None, 1] -> [None, nYlm, 1]
        # ylm_i = torch.transpose(torch.stack(sqr_ylm_i), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = sqr_ylm_i
        return (
            torch.as_tensor(ylm_r),
            torch.as_tensor(ylm_i),
        )  # [None, nYlm, 1] -> [None, nYlm]


class Precision:
    def __init__(self, prec):

        if prec == torch.float64:
            self.float = torch.float64
            self.int = torch.int64
        else:
            self.float = torch.float32
            self.int = torch.int32


class sphericalharmonics(torch.nn.Module):
    def __init__(self, lmax: int, normalize: bool) -> None:
        super().__init__()
        self._lmax = lmax
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = torch.nn.functional.normalize(
                x, dim=-1
            )  # forward 0's instead of nan for zero-radius
        sh = _spherical_harmonics(
            self._lmax, x[..., 0], x[..., 1], x[..., 2]
        ) * torch.sqrt(4 * torch.tensor(math.pi, dtype=x.dtype, device=x.device))

        return sh


@torch.jit.script
def _spherical_harmonics(
    lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    sh_0_0_r = (1 / 2) * math.sqrt(1 / math.pi) * torch.ones_like(x)
    sh_0_0 = torch.complex(sh_0_0_r, torch.zeros_like(sh_0_0_r))
    if lmax == 0:
        return torch.stack([sh_0_0,], dim=-1)
    xmiy = torch.complex(x, -y)
    xpiy = torch.complex(x, y)
    z = torch.complex(z, torch.zeros_like(z))
    sh_1_0 = (1 / 2) * math.sqrt(3 / (2 * math.pi)) * (xmiy)
    sh_1_1 = (1 / 2) * math.sqrt(3 / (math.pi)) * z
    sh_1_2 = -(1 / 2) * math.sqrt(3 / (2 * math.pi)) * (xpiy)

    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

    xmiy2 = (xmiy).pow(2)
    xpiy2 = (xpiy).pow(2)
    z2 = z.pow(2)
    sh_2_0 = (1 / 4) * math.sqrt(15 / (2 * math.pi)) * xmiy2
    sh_2_1 = (1 / 2) * math.sqrt(15 / (2 * math.pi)) * (xmiy) * z
    sh_2_2 = (1 / 4) * math.sqrt(5 / (math.pi)) * (3 * z2 - 1)
    sh_2_3 = -(1 / 2) * math.sqrt(15 / (2 * math.pi)) * (xpiy) * z
    sh_2_4 = (1 / 4) * math.sqrt(15 / (2 * math.pi)) * xpiy2

    if lmax == 2:
        return torch.stack(
            [sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
            dim=-1,
        )

    xmiy3 = xmiy2 * xmiy
    xpiy3 = xpiy2 * xpiy
    z3 = z2 * z
    sh_3_0 = (1 / 8) * math.sqrt(35 / math.pi) * xmiy3
    sh_3_1 = (1 / 4) * math.sqrt(105 / (2 * math.pi)) * xmiy2 * z
    sh_3_2 = (1 / 8) * math.sqrt(21 / math.pi) * (xmiy) * (5 * z2 - 1)
    sh_3_3 = (1 / 4) * math.sqrt(7 / math.pi) * (5 * z3 - 3 * z)
    sh_3_4 = -(1 / 8) * math.sqrt(21 / math.pi) * (xpiy) * (5 * z2 - 1)
    sh_3_5 = (1 / 4) * math.sqrt(105 / (2 * math.pi)) * (xpiy2) * z
    sh_3_6 = -(1 / 8) * math.sqrt(35 / math.pi) * xpiy3

    if lmax == 3:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ],
            dim=-1,
        )

    sh_4_0 = 0.935414346693485 * sh_3_0 * z + 0.935414346693485 * sh_3_6 * x
    sh_4_1 = (
        0.661437827766148 * sh_3_0 * y
        + 0.810092587300982 * sh_3_1 * z
        + 0.810092587300983 * sh_3_5 * x
    )
    sh_4_2 = (
        -0.176776695296637 * sh_3_0 * z
        + 0.866025403784439 * sh_3_1 * y
        + 0.684653196881458 * sh_3_2 * z
        + 0.684653196881457 * sh_3_4 * x
        + 0.176776695296637 * sh_3_6 * x
    )
    sh_4_3 = (
        -0.306186217847897 * sh_3_1 * z
        + 0.968245836551855 * sh_3_2 * y
        + 0.790569415042095 * sh_3_3 * x
        + 0.306186217847897 * sh_3_5 * x
    )
    sh_4_4 = (
        -0.612372435695795 * sh_3_2 * x + sh_3_3 * y - 0.612372435695795 * sh_3_4 * z
    )
    sh_4_5 = (
        -0.306186217847897 * sh_3_1 * x
        + 0.790569415042096 * sh_3_3 * z
        + 0.968245836551854 * sh_3_4 * y
        - 0.306186217847897 * sh_3_5 * z
    )
    sh_4_6 = (
        -0.176776695296637 * sh_3_0 * x
        - 0.684653196881457 * sh_3_2 * x
        + 0.684653196881457 * sh_3_4 * z
        + 0.866025403784439 * sh_3_5 * y
        - 0.176776695296637 * sh_3_6 * z
    )
    sh_4_7 = (
        -0.810092587300982 * sh_3_1 * x
        + 0.810092587300982 * sh_3_5 * z
        + 0.661437827766148 * sh_3_6 * y
    )
    sh_4_8 = -0.935414346693485 * sh_3_0 * x + 0.935414346693486 * sh_3_6 * z
    return torch.stack(
        [
            sh_0_0,
            sh_1_0,
            sh_1_1,
            sh_1_2,
            sh_2_0,
            sh_2_1,
            sh_2_2,
            sh_2_3,
            sh_2_4,
            sh_3_0,
            sh_3_1,
            sh_3_2,
            sh_3_3,
            sh_3_4,
            sh_3_5,
            sh_3_6,
            sh_4_0,
            sh_4_1,
            sh_4_2,
            sh_4_3,
            sh_4_4,
            sh_4_5,
            sh_4_6,
            sh_4_7,
            sh_4_8,
        ],
        dim=-1,
    )

