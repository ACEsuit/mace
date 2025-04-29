from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import math
import torch.fx
from e3nn import nn, o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace.tools.cg import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]
CHANNEL_ALPHANET = ["a", "d", "f", "g", "h", "q"]

import torch
from e3nn.nn import FullyConnectedNet

class InvariantMLPReducingWeight(torch.nn.Module):
    def __init__(self, feature_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        # Query and Key MLPs (compute attention scores)
        self.q_mlp = FullyConnectedNet(
            [feature_dim] + [feature_dim // 4] + [feature_dim // 8] + [output_dim],  # MLP for Queries
            torch.nn.functional.silu,
        )
        self.k_mlp = FullyConnectedNet(
            [feature_dim] + [feature_dim // 4] + [feature_dim // 8] + [output_dim],  # MLP for Keys
            torch.nn.functional.silu,
        )
        

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        # mixing k channels to 2
        q = self.q_mlp(x)
        # mixing k channels to 2
        k = self.k_mlp(x)
        # product to correlate the k channels
        attn_weights = torch.einsum("bf,bf->bf", q, k)  # [batch, output_dim]
        # softmax to produce 
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return attn_weights
    
@compile_mode("script")
class SymmetricCPRing(CodeGenMixin, torch.nn.Module):
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
                )
            )
            
        
        self.ring_contract_weight = torch.nn.ModuleList([InvariantMLPReducingWeight(
            irrep_out.mul * 4,
            4
            ) for _ in self.contractions])

    def forward(self, x: torch.Tensor, y: torch.Tensor): #, learned_radials: torch.Tensor):
        outs_all_L = [contraction(x, y) for contraction in self.contractions]
        
        # getting invariant many body features
        inv_mb_feats = outs_all_L[0]
        
        # reshape for later processing
        for idx in range(len(outs_all_L)):
            extra_dim = "" if str(self.contractions[idx].irrep_out) == "1x0e" else "i"
            # obtain coefficient for closing the ring via attension-like mechnism 
            # which should be of shape baf
            inv_mb_contraction_weight = self.ring_contract_weight[idx](inv_mb_feats).reshape(x.shape[0], 2, 2)
            # and then contract correspdongingly with inv_contraction_weight for each order
            # inject the invariant mb information to each order
            outs_all_L[idx] = torch.einsum(f"bacf{extra_dim},baf->bc{extra_dim}", outs_all_L[idx], inv_mb_contraction_weight)
            outs_all_L[idx] = outs_all_L[idx].reshape(outs_all_L[idx].shape[0], -1)
        return torch.cat(outs_all_L, dim=-1)
        

def _clean_tucker_rank(flexi_tucker_inner_rank, correlation):
    if flexi_tucker_inner_rank == -1:
        cleaned_flexi_tucker_inner_rank = [-1] * correlation
    else:
        cleaned_flexi_tucker_inner_rank = flexi_tucker_inner_rank    
    assert len(cleaned_flexi_tucker_inner_rank) == correlation    
    
    return cleaned_flexi_tucker_inner_rank

def initialize_weight_tensor_max_order(tensor_format, num_elements, num_params, num_features, correlation, flexi_tucker_inner_rank):
    """
    Initializes the weight tensor based on the given tensor format.

    Args:
        tensor_format (str): Type of tensor format ("symmetric_cp", "non_symmetric_cp", etc.).
        num_elements (int): Number of elements in the weight tensor.
        num_params (int): Number of parameters.
        num_features (int): Feature dimension.
        correlation (int): Correlation order.
        flexi_tucker_inner_rank (list or int): Inner rank for Tucker decompositions.

    Returns:
        torch.nn.Parameter: Initialized weight tensor.
    """
    if tensor_format in ["symmetric_tensor_ring_fast",]:
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features)) / num_params)
    elif tensor_format in ["symmetric_cp", "non_symmetric_cp", "symmetric_tensor_ring_cp", "symmetric_tensor_ring_cp2", "symmetric_tensor_ring_cp3", ]:
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features)) / num_params)

    elif tensor_format == "symmetric_tucker":
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features)) / num_params)

    elif tensor_format == "non_symmetric_tucker":
        return torch.nn.Parameter(torch.randn([num_elements, num_params] + [num_features] * correlation) / num_params)

    elif tensor_format == "flexible_symmetric_tucker":
        rank = int(num_features ** (1 / correlation)) if flexi_tucker_inner_rank[-1] == -1 else flexi_tucker_inner_rank[-1]
        return torch.nn.Parameter(torch.randn([num_elements, num_params, rank]) / num_params)

    elif tensor_format == "flexible_non_symmetric_tucker":
        rank = int(num_features ** (1 / correlation)) if flexi_tucker_inner_rank[-1] == -1 else flexi_tucker_inner_rank[-1]
        return torch.nn.Parameter(torch.randn([num_elements, num_params] + [rank] * correlation) / num_params)

    else:
        raise ValueError(f"Unsupported tensor format: {tensor_format}")

def initialize_weight_tensor(tensor_format, num_elements, num_params, num_features, i, flexi_tucker_inner_rank):
    """
    Initializes the weight tensor based on the given tensor format.

    Args:
        tensor_format (str): Type of tensor format ("symmetric_cp", "non_symmetric_cp", etc.).
        num_elements (int): Number of elements in the weight tensor.
        num_params (int): Number of parameters.
        num_features (int): Feature dimension.
        correlation (int): Correlation order.
        flexi_tucker_inner_rank (list or int): Inner rank for Tucker decompositions.

    Returns:
        torch.nn.Parameter: Initialized weight tensor.
    """
    if tensor_format in ["symmetric_tensor_ring_fast",]:
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features)) / num_params)
    if tensor_format in ["symmetric_cp", "non_symmetric_cp", "symmetric_tensor_ring_cp", "symmetric_tensor_ring_cp2", "symmetric_tensor_ring_cp3", ]:         
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features))/ num_params)
    elif tensor_format == "symmetric_tucker":
        # to be outer produced in model.forward to form symemtrized parameter tensor
        # this can be improved
        return torch.nn.Parameter(torch.randn((num_elements, num_params, num_features))/ num_params)
    elif tensor_format == "non_symmetric_tucker":
        # size of channel of the weight tensor is the current correlation order : (i)
        return torch.nn.Parameter(
                torch.randn((num_elements, num_params, *([num_features] * i)))
                / num_params
                )
    # for tucker we only need to initialize weights and all the 
    # computational details are in model.forward()
    elif tensor_format == "flexible_symmetric_tucker":
        # to be outer produced in model.forward to form symemtrized parameter tensor
        # this can be improved
        if flexi_tucker_inner_rank[i] == -1:
            return torch.nn.Parameter(
                    torch.randn((num_elements, num_params, int(num_features ** (1 / i))))
                    / num_params
                    )
        else:
            return torch.nn.Parameter(
                    torch.randn((num_elements, num_params, flexi_tucker_inner_rank[i]))
                    / num_params
                    )
    elif tensor_format == "flexible_non_symmetric_tucker":
        if flexi_tucker_inner_rank[i] == -1:
            return torch.nn.Parameter(
                    torch.randn((num_elements, num_params, *([int(num_features ** (1 / i))] * i) ))
                    / num_params
                    )
        else:
            return torch.nn.Parameter(
                    torch.randn((num_elements, num_params, *([flexi_tucker_inner_rank[i - 1]] * i)))
                    / num_params
                    )
            
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
    ) -> None:
        super().__init__()
        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        self.irreps_in = irreps_in
        self.irrep_out = irrep_out
        
        self.irrep_out_withmul = irrep_out_withmul
        
        dtype = torch.get_default_dtype()
        
        
        self.irreps_mid = o3.Irreps()
        
        ##
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

                   
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
                w = initialize_weight_tensor_max_order(tensor_format="symmetric_tensor_ring_cp2",
                                             num_elements=num_elements,
                                             num_params=num_params,
                                             num_features=self.num_features,
                                             correlation=self.correlation,
                                             flexi_tucker_inner_rank=-1
                                            )
                # optimize contraction only implemented for cp
                
                parse_subscript_main = (
                        [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                        + [f"ik,ekc,bci,be -> bc"]
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
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                self.weights_max = w
            else:
                w = initialize_weight_tensor(tensor_format="symmetric_tensor_ring_cp2",
                                             num_elements=num_elements,
                                             num_params=num_params,
                                             num_features=self.num_features,
                                             i=i,
                                             flexi_tucker_inner_rank=-1,
                                            )
                
                # optimized contraction implemented for cp only 

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
                        torch.randn(
                        [BATCH_EXAMPLE, self.num_features, num_equivariance]
                        + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)

                
                # coefficients for product basis - this linear is used for all formats
                self.weights.append(w)
                
        self.shared_k2pipj_weights = torch.nn.Parameter(torch.randn((2, self.num_features, 2)) / 4.0)
        
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        irrep_out = self.irrep_out
        num_equivariance = 2 * irrep_out.lmax + 1

        out = self.graph_opt_main(
                self.U_tensors(self.correlation),
                self.weights_max,
                x,
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
            out = contract_features(c_tensor, x)

        extra_dim = "" if str(self.irrep_out) == "1x0e" else "i"
        out = torch.einsum(f'bc{extra_dim},acd->bacd{extra_dim}', out, self.shared_k2pipj_weights)
        return out

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]
