import torch
from numpy import pi as np_pi


class SphericalHarmonics():
    def __init__(self, lmax, prec='DOUBLE'):
        self._lmax = lmax
        # self.rhat = rhat
        self.prec = Precision(prec)
        self.alm, self.blm = self.pre_compute()
        self.pi = self.float_tensor(np_pi)

    def lm1d(self, l, m):
        return m + l * (l + 1) // 2

    def lmsh(self, l, m):
        return l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2

    def lm_m(self, l, m):
        return l * (l + 1) + m

    def float_tensor(self, x):
        return torch.as_tensor(x, dtype=self.prec.float)

    def int_tensor(self, x):
        return torch.as_tensor(x, dtype=self.prec.int)

    def pre_compute(self):
        alm = [self.float_tensor(0.)]
        blm = [self.float_tensor(0.)]
        lindex = torch.arange(0, self._lmax + 1, dtype=self.prec.float)
        for i in range(1, self._lmax + 1):
            l = lindex[i]
            lsq = l * l
            ld = 2 * l
            l1 = (4 * lsq - 1)
            l2 = lsq - ld + 1
            for j in range(0, i + 1):
                m = lindex[j]
                msq = m * m
                a = torch.sqrt(l1 / (lsq - msq))
                b = -torch.sqrt((l2 - msq) / (4 * l2 - 1))
                if i == j:
                    cl = -torch.sqrt(1.0 + 0.5 / m)
                    alm += [cl]
                    blm += [torch.tensor(0)]  # placeholder
                else:
                    alm += [a]
                    blm += [b]

        return torch.stack(alm), torch.stack(blm)

    def legendre(self, x):
        x = self.float_tensor(x)

        y00 = 1. * torch.sqrt(1. / (4. * self.pi))
        plm = [x * 0 + y00]
        if self._lmax > 0:
            sq3o4pi = torch.sqrt(3. / (4. * self.pi))
            sq3o8pi = torch.sqrt(3. / (8. * self.pi))

            plm += [sq3o4pi * x]  # (1,0)
            plm += [x * 0 - sq3o8pi]  # (1,1)

            for l in range(2, self._lmax + 1):
                for m in range(0, l + 1):
                    if m == l - 1:
                        dl = torch.sqrt(2. * m + self.float_tensor(3.))
                        plm += [x * dl * plm[self.lm1d(l - 1, l - 1)]]
                    elif m == l:
                        plm += [self.alm[self.lm1d(l, l)] * plm[self.lm1d(l - 1, l - 1)]]
                    else:
                        # plm += [0.]
                        plm += [
                            self.alm[self.lm1d(l, m)] *
                            (x * plm[self.lm1d(l - 1, m)] + self.blm[self.lm1d(l, m)] * plm[self.lm1d(l - 2, m)])
                        ]

        plm = torch.stack(plm)
        return plm

    def compute_ylm_sqr(self, rhat):
        rhat = self.float_tensor(rhat)

        e_x = self.float_tensor([[1.], [0.], [0.]])
        e_y = self.float_tensor([[0.], [1.], [0.]])
        e_z = self.float_tensor([[0.], [0.], [1.]])

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
                ph = torch.where(torch.equal(torch.abs(m) % 2, 0), torch.tensor(1, dtype=self.prec.float),
                                 torch.tensor(-1, dtype=self.prec.float))
                #ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
                ph_r = ph
                ph_i = torch.zeros_like(ph, dtype=self.prec.float)  #torch.constant(0, dtype=self.prec.float)
                if m < 0 and abs(m) > l:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]
                elif m < 0 and abs(m) <= l:
                    ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    #sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [torch.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    sqr_ylm_r += [0 * ylm_r[0]]
                    sqr_ylm_i += [0 * ylm_i[0]]

        ylm_r = torch.transpose(torch.stack(sqr_ylm_r), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = torch.transpose(torch.stack(sqr_ylm_i), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]

        #return torch.reshape(ylm, [-1, self._lmax + 1, 2 * self._lmax + 1])  ##[None, nYlm, 1] -> [None, l, m]
        return torch.reshape(ylm_r, [-1, self._lmax + 1, 2 * self._lmax + 1]),\
                torch.reshape(ylm_i, [-1, self._lmax + 1, 2 * self._lmax + 1])  ##[None, nYlm, 1] -> [None, l, m]

    def compute_ylm_all(self, rhat):
        rhat = self.float_tensor(rhat)
        self.l_tile = []
        e_x = self.float_tensor([[1.], [0.], [0.]])
        e_y = self.float_tensor([[0.], [1.], [0.]])
        e_z = self.float_tensor([[0.], [0.], [1.]])

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
                ph = torch.where(torch.tensor(torch.equal(torch.abs(torch.tensor(m)) % 2, torch.tensor(0))),
                                 torch.tensor(1, dtype=self.prec.float), torch.tensor(-1, dtype=self.prec.float))
                #ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
                ph_r = ph
                if m < 0 and abs(m) > l:
                    pass
                elif m < 0 and abs(m) <= l:
                    self.l_tile += [torch.tensor(l)]
                    ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    #sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]
                    sqr_ylm_r += [ylm_r[ind] * ph_r]
                    sqr_ylm_i += [torch.negative(ylm_i[ind]) * ph_r]
                elif m >= 0 and abs(m) <= l:
                    self.l_tile += [torch.tensor(l)]
                    ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
                    sqr_ylm_r += [ylm_r[ind]]
                    sqr_ylm_i += [ylm_i[ind]]
                else:
                    pass
        self.l_tile = torch.stack(self.l_tile)
        #ylm_r = torch.transpose(torch.stack(sqr_ylm_r), [1, 0, 2])
        ylm_r = torch.stack(sqr_ylm_r).permute(1, 0, 2)  ##[nYlm, None, 1] -> [None, nYlm, 1]
        #ylm_i = torch.transpose(torch.stack(sqr_ylm_i), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = torch.stack(sqr_ylm_i).permute(1, 0, 2)
        return torch.as_tensor(ylm_r).flatten(1,
                                              -1), torch.as_tensor(ylm_i).flatten(1,
                                                                                  -1)  # [None, nYlm, 1] -> [None, nYlm]

    def compute_ylm(self, l, m, rhat):
        rhat = self.float_tensor(rhat)
        self.l_tile = []
        self.l = l
        self.m = m
        e_x = self.float_tensor([[1.], [0.], [0.]])
        e_y = self.float_tensor([[0.], [1.], [0.]])
        e_z = self.float_tensor([[0.], [0.], [1.]])

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

        ph = torch.where(torch.tensor(torch.equal(torch.abs(torch.tensor(m)) % 2, torch.tensor(0))),
                         torch.tensor(1, dtype=self.prec.float), torch.tensor(-1, dtype=self.prec.float))
        #ph = torch.complex(ph, torch.constant(0, dtype=self.prec.float))
        ph_r = ph
        if m < 0 and abs(m) > l:
            pass
        elif m < 0 and abs(m) <= l:
            self.l_tile = torch.tensor(l)
            #ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
            #sqr_ylm_r += [ylm_r[ind] * ph_r - ylm_i[ind] * ph_i]

            sqr_ylm_r = ylm_r * ph_r
            sqr_ylm_i = torch.negative(ylm_i) * ph_r

        elif m >= 0 and abs(m) <= l:
            self.l_tile = torch.tensor(l)
            #ind = self.lmsh(l, m)  # l + abs(m) * self._lmax - abs(m) * (abs(m) - 1) // 2
            sqr_ylm_r = ylm_r
            sqr_ylm_i = ylm_i
        else:
            pass

        #self.l_tile = torch.stack(self.l_tile)
        #ylm_r = torch.transpose(torch.stack(sqr_ylm_r), [1, 0, 2])
        ylm_r = sqr_ylm_r  ##[nYlm, None, 1] -> [None, nYlm, 1]
        #ylm_i = torch.transpose(torch.stack(sqr_ylm_i), [1, 0, 2])  ##[nYlm, None, 1] -> [None, nYlm, 1]
        ylm_i = sqr_ylm_i
        return torch.as_tensor(ylm_r), torch.as_tensor(ylm_i)  # [None, nYlm, 1] -> [None, nYlm]


class Precision:
    def __init__(self, prec):

        if prec == "DOUBLE":
            self.float = torch.float64
            self.int = torch.int64
        else:
            self.float = torch.float32
            self.int = torch.int32
