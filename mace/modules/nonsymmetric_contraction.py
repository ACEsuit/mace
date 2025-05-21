from functools import cache
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
from typing import Optional
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.cg import O3_e3nn
from cuequivariance.group_theory.irreps_array.misc_ui import (
    assert_same_group,
    default_irreps,
)

class NonSymmetricContraction(torch.nn.Module):
    """
    Non-symmetric contraction
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        correlation: int,
        *,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        layout: Optional[cue.IrrepsLayout] = None,
        layout_in: Optional[cue.IrrepsLayout] = None,
        layout_out: Optional[cue.IrrepsLayout] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        math_dtype: Optional[torch.dtype] = None,
        use_fallback: Optional[bool] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(irreps_in, cue.Irreps):
            irreps_in = cue.Irreps(cueq_config.group, str(irreps_in))
        if not isinstance(irreps_out, cue.Irreps):
            irreps_out = cue.Irreps(cueq_config.group, str(irreps_out))
        if layout is None:
            layout = cueq_config.layout
        if layout_in is None:
            layout_in = cueq_config.layout
        # Normalize and validate irreps
        irreps_in, irreps_out = default_irreps(irreps_in, irreps_out)
        assert_same_group(irreps_in, irreps_out)
        if len(set(irreps_in.muls) | set(irreps_out.muls)) != 1:
            raise ValueError("Input/Output irreps must have the same multiplicity")

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.correlation = correlation

        # Build a single descriptor stacking degrees 1..correlation
        self.tp_module_list = []
        self.weights_param_list = torch.nn.ParameterList()
        for degree in range(1, correlation + 1):
            self.eq_poly = nonsymmetric_contraction(irreps_in, irreps_out, degree)
            self.tp_module = cuet.EquivariantTensorProduct(
                self.eq_poly,
                layout=layout,
                layout_in=layout_in,
                layout_out=layout_out,
                device=device,
                math_dtype=math_dtype or dtype,
                use_fallback=use_fallback,
            )
            self.tp_module_list.append(self.tp_module)
            weight_dim = self.eq_poly.inputs[0].dim
            self.weight = torch.nn.Parameter(
                torch.randn(
                    1,
                    weight_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            self.weights_param_list.append(self.weight)
        self.linears_list = torch.nn.ModuleList()
        for degree in range(correlation):
            linear = cuet.Linear(
                irreps_in,
                irreps_in,
                layout=layout_in,
                shared_weights=True,
                use_fallback=True,
            )
            self.linears_list.append(linear)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, "
            f"irreps_out={self.irreps_out}, "
            f"max_degree={self.correlation}, "
            f"device={self.weight.device}, "
            f"dtype={self.weight.dtype}"
        )

    def forward(
        self,
        x: torch.Tensor,
        node_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor, shape (batch, irreps_in.dim)
        Returns:
            torch.Tensor: output tensor, shape (batch, irreps_out.dim)
        """

        out = torch.empty(
            x.shape[0], self.irreps_out.dim, device=x.device, dtype=x.dtype
        )
        all_inputs = []
        for linear in self.linears_list:
            all_inputs.append(linear(x))
        for i, tp in enumerate(self.tp_module_list):
            tp_out = tp(
                self.weights_param_list[i],  # Select weights based on indices
                *all_inputs[:i + 1],  # Use inputs up to degree i
            )
            out.add_(tp_out)
        return out



def nonsymmetric_contraction(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    degree: int,
) -> cue.EquivariantPolynomial:
    """Construct the descriptor for a multi-input tensor product contraction.

    The multi-input contraction takes multiple inputs with the same irreps and 
    contracts them together: weight[path] × (input_1 ⊗ input_2 ⊗ ... ⊗ input_N) × basis_coefficients[path]

    Args:
        irreps_in (Irreps): The input irreps (same for all inputs), multiplicities treated in parallel.
        irreps_out (Irreps): The output irreps.
        degree (int): Number of different input tensors to contract.

    Returns:
        EquivariantPolynomial: The descriptor of the multi-input contraction.
            The operands are the weights, input_1, input_2, ..., input_N, and output.

    Example:
        >>> cue.descriptors.nonsymmetric(
        ...    16 * cue.Irreps("SO3", "0 + 1 + 2"),
        ...    16 * cue.Irreps("SO3", "0 + 1"),
        ...    3  # input_1 ⊗ input_2 ⊗ input_3
        ... )
    """
    return nonsymmetric_cached(irreps_in, irreps_out, degree)


@cache
def nonsymmetric_cached(
    irreps_in: cue.Irreps,
    irreps_out: cue.Irreps,
    degree: int,
) -> cue.EquivariantPolynomial:
    
    # Extract and validate multiplicities
    mul = irreps_in.muls[0]
    assert all(mul == m for m in irreps_in.muls), "All input irreps must have same multiplicity"
    assert all(mul == m for m in irreps_out.muls), "All output irreps must have same multiplicity"
    
    # Temporarily set multiplicities to 1 for processing
    irreps_in = irreps_in.set_mul(1)
    irreps_out = irreps_out.set_mul(1)

    # Define operand indices
    input_operands = list(range(1, degree + 1))  # [1, 2, ..., degree]
    output_operand = degree + 1

    # Create input operands (all have the same structure)
    input_operands_list = [
        cue.SegmentedOperand(ndim=1, segments=[(mul,)] * irreps_in.dim)
        for _ in range(degree)
    ]

    if degree == 0:
        # Special case: no inputs, just trainable parameters
        d = cue.SegmentedTensorProduct.from_subscripts("i_i")
        for _, ir in irreps_out:
            if not ir.is_scalar():
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                d.add_path(None, None, c=1, dims={"i": ir.dim})
        d = d.flatten_modes("i")
    
    elif degree == 1:
        # Special case: single input (linear layer)
        abc = "a"
        d = cue.SegmentedTensorProduct.from_subscripts(f"w_a_i+{abc}iw")
        
        # Add input segment
        d.add_segment(1, (irreps_in.dim,))
        
        # Use linear tensor product basis
        U = cue.reduced_tensor_product_basis(
            [irreps_in],
            keep_ir=irreps_out,
            layout=cue.ir_mul
        )
        
        for _, ir in irreps_out:
            u = U.filter(keep=ir)
            if len(u.segments) == 0:
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                [u] = u.segments
                d.add_path(None, 0, None, c=u)
        
        d = d.normalize_paths_for_operand(output_operand)
        d = d.flatten_coefficient_modes()
    
    else:
        # General case: multiple inputs
        abc = "abcdefgh"[:degree]
        d = cue.SegmentedTensorProduct.from_subscripts(
            f"w_{'_'.join(f'{a}' for a in abc)}_i+{abc}iw"
        )

        # Add input segments for each input operand
        for i in input_operands:
            d.add_segment(i, (irreps_in.dim,))

        # Compute general (non-symmetric) tensor product basis
        U = cue.reduced_tensor_product_basis(
            [irreps_in] * degree,  # List of input irreps for each position
            keep_ir=irreps_out,
            layout=cue.ir_mul
        )        
        # Add paths for each output irrep
        for _, ir in irreps_out:
            u = U.filter(keep=ir)
            if len(u.segments) == 0:
                d.add_segment(output_operand, {"i": ir.dim})
            else:
                [u] = u.segments  # Shape: (a, b, c, ..., i, w)
                d.add_path(None, *(0,) * degree, None, c=u)

        d = d.normalize_paths_for_operand(output_operand)
        d = d.flatten_coefficient_modes()

    # Restore multiplicities
    d = d.append_modes_to_all_operands("u", {"u": mul})
    
    # Verify input operands match expected structure
    for i in input_operands:
        assert d.operands[i] == input_operands_list[i-1]

    # Construct the final EquivariantPolynomial
    return cue.EquivariantPolynomial(
        # Inputs: weights + all different input operands
        [cue.IrrepsAndLayout(irreps_in.new_scalars(d.operands[0].size), cue.ir_mul)]
        + [cue.IrrepsAndLayout(mul * irreps_in, cue.ir_mul) for _ in range(degree)],
        
        # Outputs
        [cue.IrrepsAndLayout(mul * irreps_out, cue.ir_mul)],
        
        cue.SegmentedPolynomial(
            [d.operands[0]] + input_operands_list,
            [d.operands[-1]],
            [(cue.Operation([0] + list(range(1, degree + 1)) + [degree + 1]), d)],
        ),
    )