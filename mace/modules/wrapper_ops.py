"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""

import dataclasses
import itertools
from typing import Iterator, List, Optional

import numpy as np
import torch
from e3nn import o3

from mace.modules.symmetric_contraction import SymmetricContraction

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

if CUET_AVAILABLE:

    class O3_e3nn(cue.O3):
        def __mul__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> Iterator["O3_e3nn"]: # pylint: disable=no-self-argument
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

        def __lt__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> bool: # pylint: disable=no-self-argument
            rep2 = rep1._from(rep2)
            return (rep1.l, rep1.p) < (rep2.l, rep2.p)

        @classmethod
        def iterator(cls) -> Iterator["O3_e3nn"]:
            for l in itertools.count(0):
                yield O3_e3nn(l=l, p=1 * (-1) ** l)
                yield O3_e3nn(l=l, p=-1 * (-1) ** l)
else:
    print(
        "cuequivariance or cuequivariance_torch is not available. Cuequivariance acceleration will be disabled."
    )


@dataclasses.dataclass
class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    enabled: bool = False
    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    group: str = "O3"
    optimize_all: bool = False  # Set to True to enable all optimizations
    optimize_linear: bool = False
    optimize_channelwise: bool = False
    optimize_symmetric: bool = False
    optimize_fctp: bool = False

    def __post_init__(self):
        if self.enabled and CUET_AVAILABLE:
            self.layout_str = self.layout
            self.layout = getattr(cue, self.layout)
            self.group = O3_e3nn if self.group == "O3" else getattr(cue, self.group)


class Linear(torch.nn.Module):
    """Wrapper around o3.Linear that optionally uses cuet.Linear when enabled"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_linear)
        ):
            self.linear = cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                optimize_fallback=not cueq_config.optimize_linear, # pylint: disable=unexpected-keyword-arg
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
        else:
            self.linear = o3.Linear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )
            self.use_cuet = False

    def __getattr__(self, name):
        """Forward any unknown attribute access to the underlying linear object"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cuet and hasattr(self, "cueq_config"):
            return self.linear(x, use_fallback=not self.cueq_config.optimize_linear)
        return self.linear(x)


class TensorProduct(torch.nn.Module):
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct"""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)
        ):
            self.tp = cuet.ChannelwiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                optimize_fallback=not cueq_config.optimize_channelwise, # pylint: disable=unexpected-keyword-arg
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
        else:
            self.tp = o3.TensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions=instructions,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )
            self.use_cuet = False

    def __getattr__(self, name):
        """Forward any unknown attribute access to the underlying linear object"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tp, name)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_cuet and hasattr(self, "cueq_config"):
            return self.tp(
                x1, x2, weights, use_fallback=not self.cueq_config.optimize_channelwise
            )
        return self.tp(x1, x2, weights)


class FullyConnectedTensorProduct(torch.nn.Module):
    """Wrapper around o3.FullyConnectedTensorProduct/cuet.FullyConnectedTensorProduct"""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)
        ):
            self.tp = cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                optimize_fallback=not cueq_config.optimize_fctp, # pylint: disable=unexpected-keyword-arg
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
        else:
            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
            )
            self.use_cuet = False

    def __getattr__(self, name):
        """Forward any unknown attribute access to the underlying linear object"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tp, name)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.use_cuet and hasattr(self, "cueq_config"):
            return self.tp(x1, x2, use_fallback=not self.cueq_config.optimize_fctp)
        return self.tp(x1, x2)


class SymmetricContractionWrapper(torch.nn.Module):
    """Wrapper around SymmetricContraction/cuet.SymmetricContraction"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        num_elements: Optional[int] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        if (
            CUET_AVAILABLE
            and cueq_config is not None
            and cueq_config.enabled
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)
        ):
            self.sconctaction = cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                optimize_fallback=not cueq_config.optimize_symmetric, # pylint: disable=unexpected-keyword-arg
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
            self.layout = cueq_config.layout
        else:
            self.sconctaction = SymmetricContraction(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                num_elements=num_elements,
            )
            self.use_cuet = False

    def __getattr__(self, name):
        """Forward any unknown attribute access to the underlying linear object"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.sc, name)

    def forward(self, x: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        if self.use_cuet and hasattr(self, "cueq_config"):
            if self.layout == cue.mul_ir:
                x = torch.transpose(x, 1, 2)
            return self.sconctaction(
                x.flatten(1),
                attrs,
                use_fallback=not self.cueq_config.optimize_symmetric,
            )
        return self.sconctaction(x, attrs)
