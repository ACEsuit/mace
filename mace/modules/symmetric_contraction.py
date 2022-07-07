from typing import Dict, Optional, Union

import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from opt_einsum import contract

from mace.tools.cg import U_matrix_real


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
        shared_weights: Optional[torch.Tensor] = None,
        element_dependent: Optional[bool] = None,
        num_elements: Optional[int] = None,
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

        if element_dependent is None:
            element_dependent = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleDict()
        for irrep_out in self.irreps_out:
            self.contractions[str(irrep_out)] = Contraction(
                irreps_in=self.irreps_in,
                irrep_out=o3.Irreps(str(irrep_out.ir)),
                correlation=correlation[irrep_out],
                internal_weights=self.internal_weights,
                element_dependent=element_dependent,
                num_elements=num_elements,
                weights=self.shared_weights,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = []
        for irrep in self.irreps_out:
            outs.append(self.contractions[str(irrep)](x, y))
        return torch.cat(outs, dim=-1)


class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        element_dependent: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.element_dependent = element_dependent
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
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        if element_dependent:
            # Tensor contraction equations
            self.equation_main = "...ik,ekc,bci,be -> bc..."
            self.equation_weighting = "...k,ekc,be->bc..."
            self.equation_contract = "bc...i,bci->bc..."
            if internal_weights:
                # Create weight for product basis
                self.weights = torch.nn.ParameterDict({})
                for i in range(1, correlation + 1):
                    num_params = self.U_tensors(i).size()[-1]
                    w = torch.nn.Parameter(
                        torch.randn(num_elements, num_params, self.num_features)
                        / num_params
                    )
                    self.weights[str(i)] = w
            else:
                self.register_buffer("weights", weights)

        else:
            # Tensor contraction equations
            self.equation_main = "...ik,kc,bci -> bc..."
            self.equation_weighting = "...k,kc->c..."
            self.equation_contract = "bc...i,bci->bc..."
            if internal_weights:
                # Create weight for product basis
                self.weights = torch.nn.ParameterDict({})
                for i in range(1, correlation + 1):
                    num_params = self.U_tensors(i).size()[-1]
                    w = torch.nn.Parameter(
                        torch.randn(num_params, self.num_features) / num_params
                    )
                    self.weights[str(i)] = w
            else:
                self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        if self.element_dependent:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.weights[str(self.correlation)],
                x,
                y,
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr),
                    self.weights[str(corr)],
                    y,
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)

        else:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.weights[str(self.correlation)],
                x,
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr),
                    self.weights[str(corr)],
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)
        resize_shape = torch.prod(torch.tensor(out.shape[1:]))
        return out.view(out.shape[0], resize_shape)

    def U_tensors(self, nu):
        return self._buffers[f"U_matrix_{nu}"]
