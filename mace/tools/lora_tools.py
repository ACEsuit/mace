import torch
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
    """LoRA for equivariant o3.Linear-like layers (preserves O(3) equivariance)."""

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

        with torch.no_grad():
            for p in self.lora_B.parameters():
                p.zero_()
            for p in self.lora_A.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = self.lora_B(self.lora_A(x))
        return base + self.scaling * delta


class LoRADenseLinear(nn.Module):
    """LoRA for torch.nn.Linear"""

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        in_f = self.base.in_features
        out_f = self.base.out_features
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)

        # match dtype/device to base
        base_param = next(self.base.parameters())
        self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
        self.lora_B.to(dtype=base_param.dtype, device=base_param.device)

        with torch.no_grad():
            nn.init.zeros_(self.lora_B.weight)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = self.lora_B(self.lora_A(x))
        return base + self.scaling * delta


class LoRAFCLayer(nn.Module):
    """LoRA for e3nn.nn._fc._Layer used by FullyConnectedNet (scalar MLP).
    Adds a low-rank delta on the weight matrix.
    """

    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("LoRAFCLayer requires a layer with a 'weight' parameter")
        self.base = base_layer

        w = self.base.weight  # type: ignore[attr-defined]
        in_f, out_f = int(w.shape[0]), int(w.shape[1])
        self.scaling = float(alpha) / float(rank)

        # Use explicit parameters to match e3nn layout [in, out]
        self.lora_A = nn.Parameter(
            torch.empty(in_f, rank, device=w.device, dtype=w.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(rank, out_f, device=w.device, dtype=w.dtype)
        )

        with torch.no_grad():
            torch.nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporarily patch the weight of the base layer to include LoRA delta
        # This avoids re-implementing the complex normalization/activation logic of e3nn _Layer
        w_orig = self.base.weight
        delta = self.lora_A @ self.lora_B
        weight_patched = w_orig + self.scaling * delta

        # Patch: self.base.weight is a Parameter. We replace it with a Tensor for the forward pass.
        # To do this safely in PyTorch, we must temporarily remove it from _parameters.
        del self.base._parameters["weight"]
        self.base.weight = weight_patched

        try:
            return self.base(x)
        finally:
            # Restore the original Parameter
            self.base.weight = w_orig
            self.base._parameters["weight"] = w_orig


def inject_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    wrap_equivariant: bool = True,
    wrap_dense: bool = True,
    _is_root: bool = True,
) -> None:
    """
    Recursively replace eligible linears with LoRA-wrapped versions.
    """

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
            if ("lora_A" in name) or ("lora_B" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False


def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    inject_lora(model, rank=rank, alpha=alpha, wrap_equivariant=True, wrap_dense=True)
    return model


def _merge_dense_lora(lora_module: LoRADenseLinear) -> nn.Linear:
    """
    Merge LoRA weights for nn.Linear.

    For nn.Linear, the forward is: base(x) + scaling * lora_B(lora_A(x))
    Which equals: x @ W_base.T + b + scaling * x @ W_A.T @ W_B.T
    So: W_merged = W_base + scaling * (W_B @ W_A)
    """
    with torch.no_grad():
        # lora_A.weight: (rank, in_features)
        # lora_B.weight: (out_features, rank)
        # delta: (out_features, in_features)
        delta = lora_module.lora_B.weight @ lora_module.lora_A.weight
        lora_module.base.weight.add_(lora_module.scaling * delta)
    return lora_module.base


def _merge_fc_lora(lora_module: LoRAFCLayer) -> nn.Module:
    """
    Merge LoRA weights for e3nn _Layer.

    The forward computes: base(x) with weight = w_orig + scaling * (lora_A @ lora_B)
    So we simply add the delta to the base weight permanently.
    """
    with torch.no_grad():
        # lora_A: (in_f, rank)
        # lora_B: (rank, out_f)
        # delta: (in_f, out_f) - matches e3nn weight layout
        delta = lora_module.lora_A @ lora_module.lora_B
        lora_module.base.weight.add_(lora_module.scaling * delta)
    return lora_module.base


def _merge_o3_lora(lora_module: LoRAO3Linear) -> o3.Linear:
    """
    Merge LoRA weights for o3.Linear by direct weight composition.

    For o3.Linear, each instruction connects an input irrep index to an output
    irrep index. The LoRA composition B(A(x)) goes through an intermediate
    bottleneck representation. We match instructions by their (i_in, i_out)
    indices and compose the weight blocks.

    Formula: W_merged = W_base + scaling * (pw_A * pw_B / pw_base) * (W_A @ W_B)
    """
    base = lora_module.base
    lora_A = lora_module.lora_A
    lora_B = lora_module.lora_B
    scaling = lora_module.scaling

    with torch.no_grad():
        # Extract weight blocks indexed by instruction
        def extract_weight_blocks(linear):
            blocks = {}
            offset = 0
            for idx, instr in enumerate(linear.instructions):
                size = instr.path_shape[0] * instr.path_shape[1]
                block = linear.weight[offset : offset + size].reshape(instr.path_shape)
                blocks[idx] = block
                offset += size
            return blocks

        base_blocks = extract_weight_blocks(base)
        A_blocks = extract_weight_blocks(lora_A)
        B_blocks = extract_weight_blocks(lora_B)

        # Build lookup tables for lora_A and lora_B instructions
        # lora_A: maps i_in -> (instruction_idx, i_out)
        A_by_i_in = {}
        for idx, instr in enumerate(lora_A.instructions):
            A_by_i_in[instr.i_in] = (idx, instr.i_out)

        # lora_B: maps (i_in, i_out) -> instruction_idx
        B_by_in_out = {}
        for idx, instr in enumerate(lora_B.instructions):
            B_by_in_out[(instr.i_in, instr.i_out)] = idx

        # Compute merged weight blocks
        merged_blocks = []
        for base_idx, base_instr in enumerate(base.instructions):
            i_in_base = base_instr.i_in
            i_out_base = base_instr.i_out
            pw_base = base_instr.path_weight

            # Find corresponding lora_A instruction (input -> bottleneck)
            if i_in_base not in A_by_i_in:
                # No LoRA for this path, keep base unchanged
                merged_blocks.append(base_blocks[base_idx])
                continue

            A_idx, i_mid = A_by_i_in[i_in_base]
            pw_A = lora_A.instructions[A_idx].path_weight

            # Find corresponding lora_B instruction (bottleneck -> output)
            B_key = (i_mid, i_out_base)
            if B_key not in B_by_in_out:
                # No LoRA for this path, keep base unchanged
                merged_blocks.append(base_blocks[base_idx])
                continue

            B_idx = B_by_in_out[B_key]
            pw_B = lora_B.instructions[B_idx].path_weight

            # Compose: W_delta = (pw_A * pw_B / pw_base) * (W_A @ W_B)
            ratio = (pw_A * pw_B) / pw_base
            delta = A_blocks[A_idx] @ B_blocks[B_idx]
            merged = base_blocks[base_idx] + scaling * ratio * delta
            merged_blocks.append(merged)

        # Flatten merged blocks back into weight tensor
        merged_weight = torch.cat([b.flatten() for b in merged_blocks])
        base.weight.copy_(merged_weight)

    return base


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

    Example:
        >>> model = load_model(...)
        >>> inject_lora(model, rank=4)
        >>> train(model)  # Train with LoRA
        >>> merge_lora_weights(model)  # Merge for fast inference
        >>> save_model(model)
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    _merge_lora_recursive(model)

    # Re-enable gradients for all parameters (they were frozen during LoRA training)
    for param in model.parameters():
        param.requires_grad = True

    return model


def _merge_lora_recursive(module: nn.Module) -> None:
    """Recursively merge LoRA layers in a module."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRADenseLinear):
            merged = _merge_dense_lora(child)
            setattr(module, name, merged)
        elif isinstance(child, LoRAFCLayer):
            merged = _merge_fc_lora(child)
            setattr(module, name, merged)
        elif isinstance(child, LoRAO3Linear):
            merged = _merge_o3_lora(child)
            setattr(module, name, merged)
        else:
            _merge_lora_recursive(child)
