import torch
from math import inf

from lgn.cg_lib import CGModule, cg_product_tau


class CGProduct(CGModule):
    r"""
    Create new CGproduct object. Inherits from CGModule, and has access
    to the CGDict related features.
    Takes two lists of type
    .. math::
        [\tau^1_{\text{min} [l_1]}, \tau^1_{\text{min} [l_1]+1}, ..., \tau^1_{\text{max} [l_1]}],
        [\tau^2_{\text{min}[l_2]}, \tau^2_{\text{min}[l_2]+1}, ..., \tau^2_{\text{max} [l_2]}],
    and outputs a new G vector of type:
    .. math::
        [\tau_{\text{min} [l]}, \tau_{\text{min} [l]+1}, ..., \tau_{\text{max_l}}]
    Each part can have an arbitrary number of batch dimensions. These batch
    dimensions must be broadcastable, unless the option :aggregate=True: is used.
    Parameters
    ----------
    tau : :class:`list` of :class:`int`, :class:`GTau`, or object with `.tau` property.
        Multiplicity of the first G vector.
    tau : :class:`list` of :class:`int`, :class:`GTau`, or object with `.tau` property.
        Multiplicity of the second G vector.
    maxdim : :class:`int`, optional
        Maximum weight to include in CG Product
    aggregate : :class:`bool`, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a :class:`GVec` as a filter.
    cg_dict : :class:`CGDict`, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxdim.
    device : :class:`torch.torch.device`, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : :class:`torch.torch.dtype`, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.
    """

    def __init__(
        self,
        tau1=None,
        tau2=None,
        aggregate=False,
        maxdim=inf,
        cg_dict=None,
        dtype=None,
        device=None,
    ):

        self.aggregate = aggregate

        if (maxdim == inf) and cg_dict:
            maxdim = cg_dict.maxdim
        elif (maxdim == inf) and (tau1 and tau2):
            maxdim = max(len(tau1), len(tau2))
        elif maxdim == inf:
            raise ValueError(
                "maxdim is not defined, and was unable to retrieve get maxdim from cg_dict or tau1 and tau2"
            )

        super().__init__(cg_dict=cg_dict, maxdim=maxdim, device=device, dtype=dtype)

        self.set_taus(tau1, tau2)

    def forward(self, rep1, rep2):
        """
        Performs the Clebsch-Gordan product.
        Parameters
        ----------
        rep1 : :class:`GVec`
            First :class:`GVec` in the CG product
        rep2 : :class:`GVec`
            Second :class:`GVec` in the CG product
        """
        if self.tau1 and self.tau1 != rep1.tau:
            raise ValueError("Input rep1 does not match predefined tau!")

        if self.tau2 and self.tau2 != rep2.tau:
            raise ValueError("Input rep2 does not match predefined tau!")

        return cg_product(
            self.cg_dict, rep1, rep2, maxdim=self.maxdim, aggregate=self.aggregate
        )

    @property
    def tau_out(self):
        if not (self.tau1) or not (self.tau2):
            raise ValueError("Module not intialized with input type!")
        tau1 = {key: 1 if val > 0 else 0 for key, val in self.tau1.items()}
        tau2 = {key: 1 if val > 0 else 0 for key, val in self.tau2.items()}
        nchan = set(
            [t for t in self.tau1.values() if t > 0]
            + [t for t in self.tau2.values() if t > 0]
        ).pop()
        tau_out = cg_product_tau(tau1, tau2, maxdim=self.maxdim)
        tau_out = {key: nchan * t for key, t in tau_out.items()}
        return tau_out

    tau = tau_out

    @property
    def tau1(self):
        return self._tau1

    @property
    def tau2(self):
        return self._tau2

    def set_taus(self, tau1=None, tau2=None):
        self._tau1 = GTau(tau1) if tau1 else None
        self._tau2 = GTau(tau2) if tau2 else None

        if self._tau1 and self._tau2:
            if not self.tau1.channels or (self.tau1.channels != self.tau2.channels):
                raise ValueError(
                    "The number of fragments must be same for each part! "
                    "{} {}".format(self.tau1, self.tau2)
                )


def cg_product(cg_dict, rep1, rep2, maxdim=inf, aggregate=False, ignore_check=False):
    """
    Explicit function to calculate the Clebsch-Gordan product.
    See the documentation for CGProduct for more information.
    rep1 : list of :obj:`torch.Tensors`
        First :obj:`GVector` in the CG product
    rep2 : list of :obj:`torch.Tensors`
        First :obj:`GVector` in the CG product
    maxdim : :obj:`int`, optional
        Minimum weight to include in CG Product
    aggregate : :obj:`bool`, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a :obj:`GVector` as a filter.
    cg_dict : :obj:`CGDict`, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxdim.
    ignore_check : :obj:`bool`
        Ignore GVec initialization check. Necessary for current implementation
        of :obj:`zonal_functions`. Use with caution.
    """
    tau1 = GTau.from_rep(rep1)
    tau2 = GTau.from_rep(rep2)

    assert tau1.channels and (
        tau1.channels == tau2.channels
    ), "The number of fragments must be same for each part! {} {}".format(tau1, tau2)

    maxk1 = max({key[0] for key in rep1.keys()})
    maxn1 = max({key[1] for key in rep1.keys()})
    maxk2 = max({key[0] for key in rep2.keys()})
    maxn2 = max({key[1] for key in rep2.keys()})
    maxDim = min(max(maxk1 + maxk2, maxn1 + maxn2) + 1, maxdim)

    if (cg_dict.maxdim < maxDim) or (cg_dict.maxdim < max(maxk1, maxn1, maxk2, maxn2)):
        raise ValueError(
            "CG Dictionary maxdim ({}) not sufficiently large for (maxdim, L1, L2) = ({} {} {})".format(
                cg_dict.maxdim, maxdim, max1, max2
            )
        )
    assert cg_dict.transpose, "This operation uses transposed CG coefficients!"

    new_rep = {}

    for (k1, n1), irrep1 in rep1.items():
        for (k2, n2), irrep2 in rep2.items():
            if (
                max(k1, n1, k2, n2) > maxDim - 1
                or irrep1.shape[-2] == 0
                or irrep2.shape[-2] == 0
            ):
                continue
            # cg_mat, aka H, is initially a dictionary {(k,n):rectangular matrix},
            # which when flattened/stacked over keys becomes an orthogonal square matrix
            # we create a sorted list of keys first and then stack the rectangular matrices over keys
            cg_mat_keys = [
                (k, n)
                for k in range(abs(k1 - k2), min(maxdim, k1 + k2 + 1), 2)
                for n in range(abs(n1 - n2), min(maxdim, n1 + n2 + 1), 2)
            ]
            cg_mat = torch.cat(
                [cg_dict[((k1, n1), (k2, n2))][key] for key in cg_mat_keys], -2
            )
            # Pairwise tensor multiply parts, loop over atom parts accumulating each.
            irrep_prod = complex_kron_product(irrep1, irrep2, aggregate=aggregate)
            # Multiply by the CG matrix, effectively turning the product into stacked irreps. Channels are preserved
            # Have to add a dummy index because matmul acts over the last two dimensions, so the vector dimension on the right needs to be -2
            cg_decomp = torch.squeeze(
                torch.matmul(cg_mat, torch.unsqueeze(irrep_prod, -1)), -1
            )
            # Split the result into a list of separate irreps
            split = [(k + 1) * (n + 1) for (k, n) in cg_mat_keys]
            cg_decomp = torch.split(cg_decomp, split, dim=-1)
            # Add the irreps to the dictionary entries, first keeping the channel dimension as a list

            for idx, key in enumerate(cg_mat_keys):
                new_rep.setdefault(key, [])
                new_rep[key].append(cg_decomp[idx])
    # at the end concatenate over the channel dimension back into torch tensors

    new_rep = {key: torch.cat(val, dim=-2) for key, val in new_rep.items()}

    # TODO: Rewrite so ignore_check not necessary
    return GVec(new_rep, ignore_check=ignore_check)


def complex_kron_product(z1, z2, aggregate=False):
    """
    Take two complex matrix tensors z1 and z2, and take their tensor product.
    Parameters
    ----------
    z1 : :class:`torch.Tensor`
        Tensor of shape batch1 x M1 x N1 x 2.
        The last dimension is the complex dimension.
    z1 : :class:`torch.Tensor`
        Tensor of shape batch2 x M2 x N2 x 2.
    aggregate: :class:`bool`
        Apply aggregation/point-wise convolutional filter. Must have batch1 = B x A x A, batch2 = B x A
    Returns
    -------
    z1 : :class:`torch.Tensor`
        Tensor of shape batch x (M1 x M2) x (N1 x N2) x 2
    """
    s1 = z1.shape
    s2 = z2.shape
    assert len(s1) >= 3, "Must have batch dimension!"
    assert len(s2) >= 3, "Must have batch dimension!"

    b1, b2 = (
        s1[1:-2],
        s2[1:-2],
    )  # b can contantain batch and atom dimensions, not channel/multiplicity
    s1, s2 = (
        s1[-2:],
        s2[-2:],
    )  # s contains the channel dimension and the actual vector dimension
    if not aggregate:
        assert b1 == b2, "Batch sizes must be equal! {} {}".format(b1, b2)
        b = b1
    else:
        if (len(b1) == 3) and (len(b2) == 2):
            assert b1[0] == b2[0], "Batch sizes must be equal! {} {}".format(b1, b2)
            assert b1[2] == b2[1], "Neighborhood sizes must be equal! {} {}".format(
                b1, b2
            )

            z2 = z2.unsqueeze(2)
            b2 = z2.shape[1:-2]
            b = b1

            agg_sum_dim = 3

        elif (len(b1) == 2) and (len(b2) == 3):
            assert b2[0] == b1[0], "Batch sizes must be equal! {} {}".format(b1, b2)
            assert b2[2] == b1[1], "Neighborhood sizes must be equal! {} {}".format(
                b1, b2
            )

            z1 = z1.unsqueeze(2)
            b1 = z1.shape[1:-2]
            b = b2

            agg_sum_dim = 3

        else:
            raise ValueError("Batch size error! {} {}".format(b1, b2))

    # Treat the channel index like a "batch index".
    assert s1[0] == s2[0], "Number of channels must match! {} {}".format(s1[0], s2[0])

    s12 = (4,) + b + (s1[0], s1[1] * s2[1])

    # here we add extra empty dimensions to construct a tensor product
    s10 = (2, 1) + b1 + (s1[0],) + torch.Size([s1[1], 1])
    s20 = (1, 2) + b2 + (s1[0],) + torch.Size([1, s2[1]])

    z = z1.view(s10) * z2.view(s20)
    z = z.contiguous().view(s12)
    if aggregate:
        # Aggregation is sum over aggregation sum dimension defined above
        z = z.sum(agg_sum_dim, keepdim=False)

    # convert the tensor product of the two complex dimensions into an actual multiplication of complex numbers
    zrot = torch.tensor(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 1.0, 0.0]], dtype=z.dtype, device=z.device
    )
    z = torch.einsum("ab,b...->a...", zrot, z)
    return z


############## UNUSED #################


def cg_product_tau(tau1, tau2, maxdim=inf):
    tau = {}

    for (k1, n1) in tau1.keys():
        for (k2, n2) in tau2.keys():
            if max(k1, n1, k2, n2) >= maxdim:
                continue
            kmin, kmax = abs(k1 - k2), min(k1 + k2, maxdim - 1)
            nmin, nmax = abs(n1 - n2), min(n1 + n2, maxdim - 1)
            for k in range(kmin, kmax + 1, 2):
                for n in range(nmin, nmax + 1, 2):
                    tau.setdefault((k, n), 0)
                    tau[(k, n)] += tau1[(k1, n1)] * tau2[(k2, n2)]

    return tau


def cg_power_tau(tau, maxdim=inf, cg_sym=True):
    if cg_sym:
        return cg_power_sym_tau(tau, maxdim=maxdim)
    else:
        return cg_product_tau(tau, tau, maxdim=maxdim)


def cg_power_sym_tau(tau, maxdim=inf):
    # Easier than rewriting
    tausq = {}

    for ((k1, n1), (k2, n2)) in itertools.combinations(tau.keys(), 2):
        kmin, kmax = abs(k1 - k2), min(k1 + k2, 2 * maxdim)
        nmin, nmax = abs(n1 - n2), min(n1 + n2, 2 * maxdim)
        for k in range(kmin, kmax + 1, 2):
            for n in range(nmin, nmax + 1, 2):
                tausq.setdefault((k, n), 0)
                tausq[(k, n)] += tau[(k1, n1)] * tau[(k2, n2)]

    # Take upper tiangular part of "tau square".
    # By symmetry of the CG coefficients, the diagonal vanishes for odd-l
    for (k1, n1) in tau:
        for k in range(0, 2 * k1 + 1, 2):
            for n in range(0, 2 * n1 + 1, 2):
                tausq.setdefault((k, n), 0)
                tausq[(k, n)] += (
                    tau[(k1, n1)]
                    * (
                        tau[(k1, n1)]
                        + (-1) ** ((((k1 - k // 2) % 2) + ((n1 - n // 2) % 2)) % 2)
                    )
                    // 2
                )

    return tausq

