import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn._fc import _Layer as E3NNFCLayer
from torch import nn


def build_lora_irreps(
    irreps_in: o3.Irreps, irreps_out: o3.Irreps, rank: int
) -> o3.Irreps:
    """
    Choose an equivariant bottleneck irreps that preserves symmetry: for every irrep
    present in BOTH input and output, allocate `rank` copies.
    """
    in_set = {ir for _, ir in o3.Irreps(irreps_in)}
    out_set = {ir for _, ir in o3.Irreps(irreps_out)}
    shared = sorted(in_set & out_set, key=lambda ir: (ir.l, ir.p))
    if not shared:
        raise ValueError(
            f"No shared irreps between input ({irreps_in}) and output ({irreps_out}); cannot build equivariant LoRA."
        )
    parts = [f"{rank}x{ir}" for ir in shared]
    return o3.Irreps(" + ".join(parts))


class LoRAO3Linear(nn.Module):
    """LoRA for equivariant o3.Linear-like layers (preserves O(3) equivariance).

    Uses fused weight computation: W_merged = W_base + scaling * (W_A @ W_B)
    with automatic caching during inference (when grad is disabled).
    """

    def __init__(self, base_linear: o3.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.irreps_in = self.base.irreps_in
        self.irreps_out = self.base.irreps_out
        self.scaling = float(alpha) / float(rank)
        self.lora_irreps = build_lora_irreps(self.irreps_in, self.irreps_out, rank)

        # Use the same class as base to avoid layout mismatches if possible
        layer_type = type(self.base)
        self.lora_A = layer_type(
            self.irreps_in, self.lora_irreps, internal_weights=True, biases=False
        )
        self.lora_B = layer_type(
            self.lora_irreps, self.irreps_out, internal_weights=True, biases=False
        )

        # Match dtype/device to base
        base_param = next(self.base.parameters())
        self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
        self.lora_B.to(dtype=base_param.dtype, device=base_param.device)

        # Cache for merged weight (used during inference)
        self._cached_merged_weight: torch.Tensor | None = None

        # Build instruction mapping for weight composition
        self._build_instruction_mapping()

        with torch.no_grad():
            for p in self.lora_B.parameters():
                p.zero_()
            for p in self.lora_A.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=1e-3)

    def _build_instruction_mapping(self) -> None:
        """Build lookup tables for matching instructions between base, A, and B."""
        # lora_A: maps i_in -> (instruction_idx, i_out, path_weight)
        self._A_by_i_in = {}
        for idx, instr in enumerate(self.lora_A.instructions):
            self._A_by_i_in[instr.i_in] = (idx, instr.i_out, instr.path_weight)

        # lora_B: maps (i_in, i_out) -> (instruction_idx, path_weight)
        self._B_by_in_out = {}
        for idx, instr in enumerate(self.lora_B.instructions):
            self._B_by_in_out[(instr.i_in, instr.i_out)] = (idx, instr.path_weight)

    @staticmethod
    def _extract_weight_blocks(linear: o3.Linear) -> dict[int, torch.Tensor]:
        """Extract weight blocks indexed by instruction."""
        blocks = {}
        offset = 0
        for idx, instr in enumerate(linear.instructions):
            size = instr.path_shape[0] * instr.path_shape[1]
            block = linear.weight[offset : offset + size].reshape(instr.path_shape)
            blocks[idx] = block
            offset += size
        return blocks

    def compute_merged_weight(self) -> torch.Tensor:
        """Compute W_base + scaling * composed(W_A, W_B) in weight space."""
        base_blocks = self._extract_weight_blocks(self.base)
        A_blocks = self._extract_weight_blocks(self.lora_A)
        B_blocks = self._extract_weight_blocks(self.lora_B)

        merged_blocks = []
        for base_idx, base_instr in enumerate(self.base.instructions):
            i_in_base = base_instr.i_in
            i_out_base = base_instr.i_out
            pw_base = base_instr.path_weight

            # Find corresponding lora_A instruction
            if i_in_base not in self._A_by_i_in:
                merged_blocks.append(base_blocks[base_idx])
                continue

            A_idx, i_mid, pw_A = self._A_by_i_in[i_in_base]

            # Find corresponding lora_B instruction
            B_key = (i_mid, i_out_base)
            if B_key not in self._B_by_in_out:
                merged_blocks.append(base_blocks[base_idx])
                continue

            B_idx, pw_B = self._B_by_in_out[B_key]

            # Compose: W_delta = (pw_A * pw_B / pw_base) * (W_A @ W_B)
            ratio = (pw_A * pw_B) / pw_base
            delta = A_blocks[A_idx] @ B_blocks[B_idx]
            merged = base_blocks[base_idx] + self.scaling * ratio * delta
            merged_blocks.append(merged)

        return torch.cat([b.flatten() for b in merged_blocks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            # Training: use activation-space computation for correct gradient flow
            self._cached_merged_weight = None
            return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

        # Inference: use fused weight-space computation with caching
        if self._cached_merged_weight is None:
            self._cached_merged_weight = self.compute_merged_weight()

        original_weight = self.base.weight.data
        self.base.weight.data = self._cached_merged_weight
        try:
            return self.base(x)
        finally:
            self.base.weight.data = original_weight

    def merge_into_base(self) -> o3.Linear:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.copy_(self.compute_merged_weight())
        return self.base


class LoRADenseLinear(nn.Module):
    """LoRA for torch.nn.Linear.

    Uses fused weight computation: W_merged = W_base + scaling * (W_B @ W_A)
    with automatic caching during inference (when grad is disabled).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.scaling = float(alpha) / float(rank)

        # LoRA matrices: W_delta = W_B @ W_A
        # W_A: (rank, in_features), W_B: (out_features, rank)
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Match dtype/device to base
        base_param = next(self.base.parameters())
        self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
        self.lora_B.to(dtype=base_param.dtype, device=base_param.device)

        # Cache for weight delta (used during inference)
        self._cached_delta: torch.Tensor | None = None

        with torch.no_grad():
            nn.init.zeros_(self.lora_B.weight)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-3)

    def compute_delta(self) -> torch.Tensor:
        """Compute the LoRA weight delta: W_B @ W_A."""
        return self.lora_B.weight @ self.lora_A.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            # Training: compute fresh delta (gradients flow through B @ A)
            self._cached_delta = None
            delta = self.compute_delta()
        else:
            # Inference: use cached delta
            if self._cached_delta is None:
                self._cached_delta = self.compute_delta()
            delta = self._cached_delta

        merged_weight = self.base.weight + self.scaling * delta
        return F.linear(x, merged_weight, self.base.bias)

    def merge_into_base(self) -> nn.Linear:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.add_(self.scaling * self.compute_delta())
        return self.base


class LoRAFCLayer(nn.Module):
    """LoRA for e3nn.nn._fc._Layer used by FullyConnectedNet (scalar MLP).

    Uses fused weight computation: W_merged = W_base + scaling * (A @ B)
    with automatic caching during inference (when grad is disabled).

    Note: e3nn uses (in, out) weight layout, so delta = A @ B (not B @ A).
    """

    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("LoRAFCLayer requires a layer with a 'weight' parameter")
        self.base = base_layer

        w = self.base.weight  # type: ignore[attr-defined]
        in_f, out_f = int(w.shape[0]), int(w.shape[1])
        self.scaling = float(alpha) / float(rank)

        # LoRA matrices: delta = A @ B (e3nn layout: in_f x out_f)
        self.lora_A = nn.Parameter(torch.empty(in_f, rank, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.empty(rank, out_f, device=w.device, dtype=w.dtype))

        # Cache for weight delta (used during inference)
        self._cached_delta: torch.Tensor | None = None

        with torch.no_grad():
            nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
            nn.init.zeros_(self.lora_B)

    def compute_delta(self) -> torch.Tensor:
        """Compute the LoRA weight delta: A @ B."""
        return self.lora_A @ self.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            # Training: compute fresh delta (gradients flow through A @ B)
            self._cached_delta = None
            delta = self.compute_delta()
        else:
            # Inference: use cached delta
            if self._cached_delta is None:
                self._cached_delta = self.compute_delta()
            delta = self._cached_delta

        merged_weight = self.base.weight + self.scaling * delta

        # Temporarily patch weight for forward (dict manipulation preserves gradient flow)
        w_orig = self.base.weight
        del self.base._parameters["weight"]
        self.base.weight = merged_weight
        try:
            return self.base(x)
        finally:
            self.base.weight = w_orig
            self.base._parameters["weight"] = w_orig

    def merge_into_base(self) -> nn.Module:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.add_(self.scaling * self.compute_delta())
        return self.base


def inject_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    wrap_equivariant: bool = True,
    wrap_dense: bool = True,
    _is_root: bool = True,
) -> None:
    """Recursively replace eligible linears with LoRA-wrapped versions."""
    for child_name, child in list(module.named_children()):
        # Skip already wrapped
        if isinstance(child, (LoRAO3Linear, LoRADenseLinear, LoRAFCLayer)):
            continue
        # Equivariant o3.Linear
        if wrap_equivariant and isinstance(child, o3.Linear):
            try:
                wrapped = LoRAO3Linear(child, rank=rank, alpha=alpha)
            except ValueError:  # If no shared irreps, skip
                continue
            setattr(module, child_name, wrapped)
        # Dense nn.Linear
        if wrap_dense and isinstance(child, nn.Linear):
            wrapped = LoRADenseLinear(child, rank=rank, alpha=alpha)
            setattr(module, child_name, wrapped)
            continue
        # e3nn FullyConnectedNet internal layer
        if wrap_dense and isinstance(child, E3NNFCLayer):
            wrapped = LoRAFCLayer(child, rank=rank, alpha=alpha)
            setattr(module, child_name, wrapped)
            continue
        # Recurse
        inject_lora(child, rank, alpha, wrap_equivariant, wrap_dense, _is_root=False)

    if _is_root:
        for name, p in module.named_parameters():
            p.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    inject_lora(model, rank=rank, alpha=alpha, wrap_equivariant=True, wrap_dense=True)
    return model


def merge_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Merge LoRA weights into base weights and replace LoRA wrappers with merged base modules.

    This eliminates the inference overhead from LoRA by folding the low-rank
    adaptations directly into the original weight matrices. After merging:
    - LoRADenseLinear -> nn.Linear (with merged weights)
    - LoRAFCLayer -> e3nn _Layer (with merged weights)
    - LoRAO3Linear -> o3.Linear (with merged weights)

    Args:
        model: Model containing LoRA layers to merge.
        inplace: If True, modifies the model in place. If False, works on a deep copy.

    Returns:
        Model with LoRA weights merged into base layers. All parameters will have
        requires_grad=True after merging.
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    def merge_recursive(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRADenseLinear, LoRAFCLayer, LoRAO3Linear)):
                setattr(module, name, child.merge_into_base())
            else:
                merge_recursive(child)

    merge_recursive(model)

    # Re-enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True

    return model
