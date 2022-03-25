from ast import Tuple
from collections import OrderedDict
from typing import Optional
import torch
from e3nn.util.codegen import CodeGenMixin
from e3nn import o3
import torch.fx
from opt_einsum_fx import optimize_einsums_full

from torch import fx

from e3nn.util.jit import compile_mode


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: Optional[bool] = True,
        _specialized_code: Optional[bool] = True,
        _optimize_einsums: Optional[bool] = True,
        use_complex: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if use_complex:
            self.type = torch.complex128
        else:
            self.type = torch.float64

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        for corr in correlation:
            pass
        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        graphmod_left_right = codgen_symmetric_product(
            self.irreps_in,
            self.irreps_out,
            self.shared_weights,
            _specialized_code,
            _optimize_einsums,
            use_complex,
        )


def codgen_symmetric_product(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
    use_complex: bool = False,
) -> Tuple[fx.GraphModule, fx.GraphModule]:

    graph = fx.Graph()

    # = Function definitions =
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    x1s = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder("x2", torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder("w", torch.Tensor), tracer=tracer)

    empty = fx.Proxy(
        graph.call_function(torch.empty, ((),), dict(device="cpu")), tracer=tracer
    )
    if shared_weights:
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1])
        )[0].shape
    else:
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]),
            empty.expand(x2s.shape[:-1]),
            empty.expand(weights.shape[:-1]),
        )[0].shape
    del empty

    for corr in correlation:
        pass
