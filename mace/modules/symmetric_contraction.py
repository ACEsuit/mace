###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace.tools.cg import U_matrix_real

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
        tensor_format: str = "symmetric_cp"
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

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
            print(irrep_out)
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                    tensor_format=tensor_format
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        tensor_format: str = "symmetric_cp",
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
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
        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])
        # for tucker decomposition
        for i in range(correlation, 0, -1):

            print(f"initing i = {i}")
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]
            
            if i == correlation:
                
                if tensor_format == "symmetric_cp":
                    channel_idx = "c"
                    sample_x = torch.randn((BATCH_EXAMPLE, self.num_features, num_ell))
                    w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                    )
                elif tensor_format == "symmetric_tucker":
                    #channel_idx = "".join([CHANNEL_ALPHANET[j] for j in range(self.correlation)])
                    channel_idx = "c"
                    sample_x = torch.randn([BATCH_EXAMPLE, ] + [self.num_features, ] + [num_ell, ])
                    w = torch.nn.Parameter(
                    torch.randn([num_elements, num_params,] + [self.num_features,])
                    / num_params
                    )
                    
                parse_subscript_main = (
                        [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                        + [f"ik,ek{channel_idx},b{channel_idx}i,be -> b{channel_idx}"]
                        + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    )
                print("".join(parse_subscript_main))
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
                        #torch.randn((num_elements, num_params, self.num_features)),
                        #torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn(w.shape),
                        sample_x,
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                # w = torch.nn.Parameter(
                #     torch.randn((num_elements, num_params, self.num_features))
                #     / num_params
                # )
                self.weights_max = w
            else:
                
                if tensor_format == "symmetric_cp":
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
                    out_channel_idx = "".join([CHANNEL_ALPHANET[j] for j in range(self.correlation - i + 1)])
                    #in_channel_idx = out_channel_idx[:-1]
                    #channel_idx = "c"
                    sample_x = torch.randn([BATCH_EXAMPLE, self.num_features, num_ell])
                    # order of features is of length out_channel_idx - 1
                    sample_x2 = torch.randn(
                            [BATCH_EXAMPLE,] + [self.num_features,] * (self.correlation - i) + [num_equivariance, ]
                            + [num_ell] * i
                            ).squeeze(self.correlation - i + 1)
                    print("sample_x.shape: ", sample_x.shape)
                    print("sample_x2.shap: ", sample_x2.shape)
                    w = torch.nn.Parameter(
                            torch.randn((num_elements, num_params, self.num_features))
                            / num_params
                            )
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + [f"k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                if tensor_format == "symmetric_cp":
                    parse_subscript_features = (
                        [f"bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                        + [f"i,bci->bc"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    )
                elif tensor_format == "symmetric_tucker":
                    parse_subscript_features = (
                        [f"b{out_channel_idx[:-1]}"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                        + [f"i,b{out_channel_idx[-1]}i->b{out_channel_idx}"]
                        + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    )
                print("weighting: ", "".join(parse_subscript_weighting))
                print("features: ", "".join(parse_subscript_features))

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
                        # torch.randn(
                        #     [BATCH_EXAMPLE, self.num_features, num_equivariance]
                        #     + [num_ell] * i
                        # ).squeeze(2),
                        #torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        sample_x2,
                        sample_x,
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                # w = torch.nn.Parameter(
                #     torch.randn((num_elements, num_params, self.num_features))
                #     / num_params
                # )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        print(" === graph opt main ===")
        print("self.U_tensors(self.correlation).shape: ", self.U_tensors(self.correlation).shape)
        print("self.weights_max.shape", self.weights_max.shape)
        print("x.shape: ", x.shape)
        print("y.shape :", y.shape)
            
        out = self.graph_opt_main(
            self.U_tensors(self.correlation),
            self.weights_max, # shape = (num_elements, num_paras, channel)
            x,
            y,
        )
        # if self.tensor_format == "symmetric_tucker":
        #     for i in range(self.correlation - 1):
        #         out = out.unsqueeze(2)
        print("out.shape :", out.shape)
        print(' === ')
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            print(f"=== correlation - i - 1: {self.correlation - i - 1} === ")
            print("self.U_tensors(self.correlation - i - 1).shape", self.U_tensors(self.correlation - i - 1).shape)
            print("weight.shape :", weight.shape)
            print("y.shape", y.shape)
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            ) # this step combine the cf coefficients and the weight tensor together
            # non-symmetric would require different tensor for each order slice??
            # unsqueze make it automatically boardcast to each channel when doing c_tensor+out
            if self.tensor_format == "symmetric_tucker":
                for _ in range(i):
                    c_tensor = c_tensor.unsqueeze(2)
            print("c_tensor.shape: ", c_tensor.shape)
            print("out.shape before contract: ", out.shape)
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
            print("out.shape: ", out.shape)
            print("===")
        #print("out before reshape: ", out.shape)
        #print("out after reshape: ", out.reshape(out.shape[0], -1))
        return out.reshape(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]
