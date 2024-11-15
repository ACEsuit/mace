"""
Wrapper class for o3.Linear that optionally uses cuet.Linear
"""
import dataclasses
import torch
from typing import List, Optional
import e3nn.o3 as o3

from mace.modules.symmetric_contraction import SymmetricContraction
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

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
            self.group = getattr(cue, self.group)

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
        if (CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled 
            and (cueq_config.optimize_all or cueq_config.optimize_linear)):
            self.linear = cuet.Linear(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                optimize_fallback=not cueq_config.optimize_linear
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
        else:
            self.linear = o3.Linear(
                irreps_in,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights
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
        if (CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled 
            and (cueq_config.optimize_all or cueq_config.optimize_channelwise)):
            self.tp = cuet.ChannelwiseTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2),
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                optimize_fallback=not cueq_config.optimize_channelwise
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
                internal_weights=internal_weights
            )
            self.use_cuet = False

    def __getattr__(self, name):
        """Forward any unknown attribute access to the underlying linear object"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.tp, name)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_cuet and hasattr(self, "cueq_config"):
            return self.tp(x1, x2, weights, use_fallback=not self.cueq_config.optimize_channelwise)
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
        if (CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled 
            and (cueq_config.optimize_all or cueq_config.optimize_fctp)):
            self.tp = cuet.FullyConnectedTensorProduct(
                cue.Irreps(cueq_config.group, irreps_in1),
                cue.Irreps(cueq_config.group, irreps_in2), 
                cue.Irreps(cueq_config.group, irreps_out),
                layout=cueq_config.layout,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                optimize_fallback=not cueq_config.optimize_fctp
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
        else:
            self.tp = o3.FullyConnectedTensorProduct(
                irreps_in1,
                irreps_in2,
                irreps_out,
                shared_weights=shared_weights,
                internal_weights=internal_weights
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
        if (CUET_AVAILABLE and cueq_config is not None and cueq_config.enabled 
            and (cueq_config.optimize_all or cueq_config.optimize_symmetric)):
            self.sconctaction = cuet.SymmetricContraction(
                cue.Irreps(cueq_config.group, irreps_in),
                cue.Irreps(cueq_config.group, irreps_out), 
                layout_in=cue.ir_mul,
                layout_out=cueq_config.layout,
                contraction_degree=correlation,
                num_elements=num_elements,
                optimize_fallback=not cueq_config.optimize_symmetric
            )
            self.use_cuet = True
            self.cueq_config = cueq_config
            self.layout = cueq_config.layout
        else:
            self.sconctaction = SymmetricContraction(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                num_elements=num_elements
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
            return self.sconctaction(x.flatten(1), attrs, use_fallback=not self.cueq_config.optimize_symmetric)
        return self.sconctaction(x, attrs)