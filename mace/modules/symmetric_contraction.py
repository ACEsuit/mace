###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import math
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace.tools.cg import U_matrix_real
from .irreps_tools import reshape_irreps, inverse_reshape_irreps

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
CHANNEL_ALPHANET = ["a", "d", "f", "g", "h", "q"]

@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
        agnostic: Optional[bool] = False,
        tensor_format: str = "symmetric_cp",
        flexible_feats_L: bool = False,
        gaussian_prior: bool = False,
    ) -> None:
        super().__init__()

        self.gaussian_prior = gaussian_prior
        
        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        if flexible_feats_L:
            print("=== using flexible channel for different L levels === ")
        if gaussian_prior: 
            print("=== using trainable gaussian prior === ")

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]
        assert tensor_format in ["symmetric_cp", "symemtric_tucker", "non_symmetric_cp", \
                                 "non_symmetric_tucker", "flexible_symmetric_tucker", "flexible_non_symmetric_tucker"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.tensor_format = tensor_format
        
        self.contractions = torch.nn.ModuleList()
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    irrep_out_withmul = irrep_out,
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                    tensor_format=tensor_format,
                    flexible_feats_L=flexible_feats_L,
                    gaussian_prior=gaussian_prior
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor): #, learned_radials: torch.Tensor):
        # different contract is differnt level of L = 0, 1, 2...
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        irrep_out_withmul: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        tensor_format: str = "symmetric_cp",
        flexible_feats_L: bool = False,
        gaussian_prior: bool = False,
        
    ) -> None:
        super().__init__()
        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        self.irreps_in = irreps_in
        self.irrep_out = irrep_out
        self.flexible_feats_L = flexible_feats_L
        self.gaussian_prior = gaussian_prior
        
        self.irrep_out_withmul = irrep_out_withmul
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
                tensor_format=tensor_format
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        self.tensor_format = tensor_format
        
        # if this is tucker format, in order to allow
        # more A_klm basis to be formed and prevent the k^nu
        # scaling we need this further contraction
        self.irreps_mid = o3.Irreps()
        
        # control dimension flexibly for each nu
        # TODO: generalize to allow more flexible dimension
        self.irreps_nu = []
        self.linear_nu = torch.nn.ModuleList([])
        self.linear_nu_reshape = torch.nn.ModuleList([])
        
        if self.tensor_format in ["flexible_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            
            
            for irrep_in in self.irreps_in:
                self.irreps_mid += o3.Irreps(f"{irrep_out_withmul.mul}x{irrep_in.ir}")
            
            # allowing ax0e + bx1o + cx2e etc. instead of only K * (1x0e + 1x1o + 1x2e)
            if self.flexible_feats_L:
                self.linear = o3.Linear(self.irreps_in, 
                                    self.irreps_mid,
                                    internal_weights=True,
                                    shared_weights=True,)
            else:
                # if not allowing flexible dimension
                assert self.irreps_mid == self.irreps_in

            # update num_features too
            self.num_features = self.irreps_mid.count((0, 1))
            
            for nu in range(correlation, 0, -1):
                tmp_irreps = o3.Irreps()
                for irrep_mid in self.irreps_mid:
                    tmp_irreps += o3.Irreps(f"{int((irrep_mid.mul) ** (1 / nu))}x{irrep_mid.ir}")
                self.irreps_nu.append(tmp_irreps)
                self.linear_nu.append(o3.Linear(self.irreps_mid,
                                                tmp_irreps,
                                                internal_weights=True,
                                                shared_weights=True
                                                ))
                self.linear_nu_reshape.append(reshape_irreps(tmp_irreps))
        
        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])
        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]
            
            if i == correlation:
                if tensor_format in ["symmetric_cp", "non_symmetric_cp"]:
                    channel_idx = "c"
                    sample_x = torch.randn((BATCH_EXAMPLE, self.num_features, num_ell))
                    w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                    )
                elif tensor_format == "symmetric_tucker":
                    w = torch.nn.Parameter(
                    torch.randn([num_elements, num_params,] + [self.num_features,])
                    / num_params
                    )
                elif tensor_format == "non_symmetric_tucker":
                    w = torch.nn.Parameter(
                    torch.randn([num_elements, num_params,] + [self.num_features,] * self.correlation)
                    / num_params
                    )
                elif tensor_format == "flexible_symmetric_tucker":
                    w = torch.nn.Parameter(
                    torch.randn([num_elements, num_params,] + [int(self.num_features ** (1 / correlation)),])
                    / num_params
                    )
                elif tensor_format == "flexible_non_symmetric_tucker":
                    w = torch.nn.Parameter(
                    torch.randn([num_elements, num_params,] + [int(self.num_features ** (1 / correlation)),] * self.correlation)
                    / num_params
                    )

                # optimize contraction only implemented for cp
                if "cp" in self.tensor_format:
                    parse_subscript_main = (
                            [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                            + [f"ik,ek{channel_idx},b{channel_idx}i,be -> b{channel_idx}"]
                            + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                        )
                    graph_module_main = torch.fx.symbolic_trace(
                        lambda x, y, w, z: torch.einsum(
                            "".join(parse_subscript_main), x, y, w, z
                        )
                    )

                    # Optimizing the contractions
                    self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_main,
                        example_inputs=(
                            torch.randn(
                                [num_equivariance] + [num_ell] * i + [num_params]
                            ).squeeze(0),
                            torch.randn(w.shape),
                            sample_x,
                            torch.randn((BATCH_EXAMPLE, num_elements)),
                        ),
                    )
                # Parameters for the product basis
                self.weights_max = w
            else:
                
                if tensor_format in ["symmetric_cp", "non_symmetric_cp"]:
                    channel_idx = "c"
                    sample_x = torch.randn((BATCH_EXAMPLE, self.num_features, num_ell))
                    sample_x2 = torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                            ).squeeze(2)
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, self.num_features))
                            / num_params
                            )
                elif tensor_format == "symmetric_tucker":
                    # to be outer produced in model.forward to form symemtrized parameter tensor
                    # this can be improved
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, self.num_features))
                            / num_params
                            )
                elif tensor_format == "non_symmetric_tucker":
                    # size of channel of the weight tensor is the current correlation order : (i)
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, *([self.num_features] * i)))
                            / num_params
                            )
                # for tucker we only need to initialize weights and all the 
                # computational details are in model.forward()
                elif tensor_format == "flexible_symmetric_tucker":
                    # to be outer produced in model.forward to form symemtrized parameter tensor
                    # this can be improved
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, int(self.num_features ** (1 / i))))
                            / num_params
                            )
                elif tensor_format == "flexible_non_symmetric_tucker":
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, *([int(self.num_features ** (1 / i))] * i) ))
                            / num_params
                            )
                # === most of tucker related init ends here except putting weight in nn.ModuleList and some reshape weights ===
                
                # optimized contraction implemented for cp only 
                if "cp" in tensor_format:
                    # Generate optimized contractions equations
                    parse_subscript_weighting = (
                        [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                        + [f"k,ekc,be->bc"]
                        + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    )
                    parse_subscript_features = (
                        [f"bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                        + [f"i,bci->bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    )

                    # Symbolic tracing of contractions
                    graph_module_weighting = torch.fx.symbolic_trace(
                        lambda x, y, z: torch.einsum(
                            "".join(parse_subscript_weighting), x, y, z
                        )
                    )
                    graph_module_features = torch.fx.symbolic_trace(
                        lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                    )

                    # Optimizing the contractions
                    graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_weighting,
                        example_inputs=(
                            torch.randn(
                                [num_equivariance] + [num_ell] * i + [num_params]
                            ).squeeze(0),
                            #torch.randn((num_elements, num_params, self.num_features)),
                            torch.randn(w.shape),
                            torch.randn((BATCH_EXAMPLE, num_elements)),
                        ),
                    )
                    graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                        model=graph_module_features,
                        example_inputs=(
                            sample_x2,
                            sample_x,
                        ),
                    )
                    self.contractions_weighting.append(graph_opt_weighting)
                    self.contractions_features.append(graph_opt_features)

                # coefficients for product basis
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]
            
        # # add gaussian prior    
        # if gaussian_prior: 
        #     # given by exp(-wK * ( Rk_fn(Rk) )^2 - wL * sum_t l^2_t)
        #     # For CP, this can be done by two steps:
        #     # - multiply A_klm alog lm channel by exp(-wL * l^2_t) during the t th contraction
        #     # - multiply the final feature along k channel by exp(-wK * ( Rk_fn(Rk) )^2 )
        #     # normalization weighting over the lms
        #     self.wL = torch.nn.Parameter(torch.tensor(1.5, requires_grad = True))
        #     # smoothness prior for normalization
        #     self.register_buffer("l2vec", torch.tensor([irrep.l for _, irrep in self.coupling_irreps for _ in range((2 * irrep.l + 1))]))
        #     # normalization weighting over radials
        #     #self.wK = torch.nn.Parameter(torch.tensor(1.5, requires_grad = True))
        #     # # learnable smoothness of learned radial
        #     # self.Rk_fn = nn.FullyConnectedNet(
        #     #                                 [self.num_features]
        #     #                                 + [
        #     #                                     self.num_features,
        #     #                                 ],
        #     #                                 torch.nn.functional.silu,
        #     #                             )
        # else:
        #     self.wL = 0
        #     self.register_buffer("l2vec", torch.zeros_like(torch.tensor([irrep.l for _, irrep in self.coupling_irreps for _ in range((2 * irrep.l + 1))])))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        irrep_out = self.irrep_out
        num_equivariance = 2 * irrep_out.lmax + 1
        if "tucker" in self.tensor_format:
            # this allow generalization for different num_feats for 
            # different level of L
            if self.flexible_feats_L:
                x = self.linear(x) #self.reshape(self.linear(x))
            
            outs = dict()
            out_channel_idx = "".join([CHANNEL_ALPHANET[j] for j in range(self.correlation)])
            idx = 0
            for nu in range(self.correlation, 0, -1):
                num_params = self.U_tensors(nu).size()[-1]
                num_ell = self.U_tensors(nu).size()[-2]
                # "stick" x_nu to basis of all orders
                x_nu = x # init for scope
                # all symmetric formats requires just the same x for each nu
                # but note that the x from interaction layer from "flexible" formats
                # are of shape [n_nodes, klm] instead of [n_nodes, k, lm]
                if self.tensor_format in ["flexible_symmetric_tuckcer", "symmetric_tucker"]:
                    x_nu = x # no-op
                # non_symmmetric_tucker: interaction layer gives [n_nodes, k, lm, nu]
                elif self.tensor_format in ["non_symmetric_tucker", ]:
                    x_nu = x[:, :, :, nu - 1]
                # flexible_non_symmetric_tucker: intearaction layer gives [n_nnodes, klm, nu]
                # to be reshaped later with nu dependent layer
                elif self.tensor_format in ["flexible_non_symmetric_tucker", ]:
                    x_nu = x[:, :, nu - 1]

                # channel index of the final message
                # in symmetric tucker we cannot sum over the basis of different order
                # and will just produce "m_\tilde{k}LM" with \tilde{k} as tuple 
                if nu == self.correlation:
                    x_nu_trans = self.linear_nu_reshape[idx](self.linear_nu[idx](x_nu)) if "flexible" in self.tensor_format else x_nu
                    
                    outs[nu] = torch.einsum("".join((
                                            [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                                            + [f"ik,b{out_channel_idx[-1]}i-> b{out_channel_idx[-1]}"]
                                            +  [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                                            + ["k"]
                                            )),
                                            self.U_tensors(nu), 
                                            x_nu_trans)
                else:
                    # contractions to be done for U_tensors(nu)
                    idx2 = 0
                    for nu2 in range(self.correlation, nu - 1, -1):
                        # contraction for current nu
                        # [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                        # denotes indices that are preserved for further contractions
                        if nu2 == nu:
                            x_nu_trans = self.linear_nu_reshape[idx](self.linear_nu[idx](x_nu)) \
                                            if "flexible" in self.tensor_format else x_nu
                            outs[nu2] = torch.einsum("".join((
                                                        [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                                                        + [f"ik,b{out_channel_idx[nu]}i-> b{out_channel_idx[nu]}"]
                                                        +  [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)] 
                                                        + ["k"]
                                                        ))
                                                    ,
                                                    self.U_tensors(nu),
                                                    x_nu_trans
                                                    )          
                        # also contract previous nu and expand the tensor product basis
                        else:
                            x_nu_trans = self.linear_nu_reshape[idx2](self.linear_nu[idx2](x_nu)) \
                                                if "flexible" in self.tensor_format else x_nu
                            outs[nu2] = torch.einsum(
                                "".join([f"b{out_channel_idx[-(nu2 - nu):]}"]
                                + [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                                + ["ik,"]
                                + [f"b{out_channel_idx[-(nu2 - nu) - 1]}i -> b{out_channel_idx[-(nu2 - nu) - 1:]}"]
                                + [ALPHABET[j] for j in range(nu + min(irrep_out.lmax, 1) - 1)]
                                + ["k"]
                                ),
                                outs[nu2],
                                x_nu_trans)
                        idx2 += 1 # for each nu2 

                idx += 1 # for each nu
            
            # product basis coeffcients layer
            for nu in range(self.correlation, 0, -1):
                if nu == self.correlation:
                    if self.tensor_format in ["non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
                        c_tensor = torch.einsum(f"ek{out_channel_idx[:nu]},be->bk{out_channel_idx[:nu]}", self.weights_max, y)
                    elif self.tensor_format in ["symmetric_tucker", "flexible_symmetric_tucker"]:
                        c_tensor = torch.einsum("ekc,be->bkc", self.weights_max, y)
                        # outer product to symmetrize tensor
                        c_tensor = torch.einsum("".join([f"bk{out_channel_idx[i]}," for i in range(nu-1)]
                                +[f"bk{out_channel_idx[nu-1]}"]
                                +[f"->bk{out_channel_idx[:nu]}"]
                                ),
                        *[c_tensor for _ in range(nu)]) / torch.jit._builtins.math.factorial(nu)
                else:
                    if self.tensor_format in ["non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
                        c_tensor = torch.einsum(f"ek{out_channel_idx[:nu]},be->bk{out_channel_idx[:nu]}", self.weights[self.correlation - nu - 1], y)
                    elif self.tensor_format in ["symmetric_tucker", "flexible_symmetric_tucker"]:
                        c_tensor = torch.einsum("ekc,be->bkc", self.weights[self.correlation - nu - 1], y)
                        # outer product to symmetrize tensor
                        c_tensor = torch.einsum("".join([f"bk{out_channel_idx[i]}," for i in range(nu-1)]
                                +[f"bk{out_channel_idx[nu-1]}"]
                                +[f"->bk{out_channel_idx[:nu]}"]
                                ),
                        *[c_tensor for _ in range(nu)]) / torch.jit._builtins.math.factorial(nu)
                
                outs[nu] = torch.einsum(
                    "".join(
                        [f"b{out_channel_idx[:nu]}"]
                        +[ALPHABET[j] for j in range(min(irrep_out.lmax, 1))]
                        +["k"]
                        +[f",bk{out_channel_idx[:nu]}->b{out_channel_idx[:nu]}"]
                        +[ALPHABET[j] for j in range(min(irrep_out.lmax, 1))]
                    ),
                    outs[nu],
                    c_tensor,
                )
            for nu in range(self.correlation, 0, -1):
                if self.tensor_format in ["flexible_symmetric_tucker", "flexible_non_symmetric_tucker"]:
                    shape_outnu = [outs[nu].shape[0]] + [int(self.num_features ** (1 / nu))] * nu
                elif self.tensor_format in ["symmetric_tucker", "non_symmetric_tucker"]:
                    shape_outnu = [outs[nu].shape[0]] + [self.num_features] * nu
                if irrep_out.lmax > 0:
                    shape_outnu += [num_equivariance]
                # combine all the features channels
                outs[nu] = outs[nu].reshape(*shape_outnu)
                # reshape kLM
                outs[nu] = outs[nu].reshape(outs[nu].shape[0], -1)
            return torch.cat([outs[nu] for nu in range(self.correlation, 0, -1)], dim = 1)
        
        ## previous CP implementation
        elif "cp" in self.tensor_format:
            if self.tensor_format == "symmetric_cp":
                out = self.graph_opt_main(
                    self.U_tensors(self.correlation),
                    self.weights_max, # shape = (num_elements, num_paras, channel)
                    #x * torch.exp(-self.wL * (self.l2vec ** 2) ) if self.gaussian_prior else x, # [nnodes, channel, num_paras]
                    x,
                    y,
                )
            elif self.tensor_format == "non_symmetric_cp":
                out = self.graph_opt_main(
                    self.U_tensors(self.correlation),
                    self.weights_max, # shape = (num_elements, num_paras, channel)
                    #x[:, :, :, self.correlation - 1] * torch.exp(-self.wL * (self.l2vec ** 2) ) if self.gaussian_prior else x[:, :, :, self.correlation - 1],
                    x[:, :, :, self.correlation - 1],
                    y,
                )
            for i, (weight, contract_weights, contract_features) in enumerate(
                zip(self.weights, self.contractions_weighting, self.contractions_features)
            ):
                c_tensor = contract_weights(
                    self.U_tensors(self.correlation - i - 1),
                    weight,
                    y,
                )
                c_tensor = c_tensor + out
                if self.tensor_format == "symmetric_cp":
                    #out = contract_features(c_tensor, x * torch.exp(-self.wL * (self.l2vec ** 2) if self.gaussian_prior else x))
                    out = contract_features(c_tensor, x)
                elif self.tensor_format == "non_symmetric_cp":
                    #out = contract_features(c_tensor, x[:, :, :, self.correlation - i - 1] * torch.exp(-self.wL * (self.l2vec ** 2) if self.gaussian_prior else x[:, :, :, self.correlation - i - 1] ))
                    out = contract_features(c_tensor, x[:, :, :, self.correlation - i - 1])
            return out.reshape(out.shape[0], -1)


    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]