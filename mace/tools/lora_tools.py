from e3nn import o3
try:
    from e3nn.nn._fc import _Layer as E3NNFCLayer  # type: ignore
except Exception:  # pragma: no cover - e3nn not available in some envs
    E3NNFCLayer = None  # type: ignore
import torch
import torch.nn as nn
from typing import Tuple


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters(recurse=True):
        p.requires_grad = False


def build_lora_irreps(irreps_in: o3.Irreps, irreps_out: o3.Irreps, rank: int) -> o3.Irreps:
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
        _freeze_module(self.base)
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
        try:
            base_param = next(self.base.parameters())
            self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
            self.lora_B.to(dtype=base_param.dtype, device=base_param.device)
        except StopIteration:
            pass
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
    """LoRA for torch.nn.Linear (scalar MLPs; does not affect equivariance)."""

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if base_linear.bias is not None:
            # Keep bias in the base path only
            bias = base_linear.bias
        self.base = base_linear
        _freeze_module(self.base)
        in_f = self.base.in_features
        out_f = self.base.out_features
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        # Match dtype/device to base
        try:
            base_param = next(self.base.parameters())
            self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
            self.lora_B.to(dtype=base_param.dtype, device=base_param.device)
        except StopIteration:
            pass
        with torch.no_grad():
            nn.init.zeros_(self.lora_B.weight)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = self.lora_B(self.lora_A(x))
        return base + self.scaling * delta


class LoRAFCLayer(nn.Module):
    """LoRA for e3nn.nn._fc._Layer used by FullyConnectedNet (scalar MLP).
    Adds a low-rank delta on the weight matrix; preserves scalar nature.
    """

    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("LoRAFCLayer requires a layer with a 'weight' parameter")
        self.base = base_layer
        _freeze_module(self.base)
        w = self.base.weight  # type: ignore[attr-defined]
        in_f, out_f = int(w.shape[0]), int(w.shape[1])
        self.scaling = float(alpha) / float(rank)
        # Use explicit parameters to match e3nn layout [in, out]
        self.lora_A = nn.Parameter(torch.empty(in_f, rank, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.empty(rank, out_f, device=w.device, dtype=w.dtype))
        with torch.no_grad():
            torch.nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate e3nn _Layer normalization exactly
        W = self.base.weight  # type: ignore[attr-defined]
        h_in = getattr(self.base, "h_in")
        var_in = getattr(self.base, "var_in")
        var_out = getattr(self.base, "var_out")
        act = getattr(self.base, "act", None)

        delta = self.lora_A @ self.lora_B
        W_sum = W + self.scaling * delta

        if act is not None:
            denom = (h_in * var_in) ** 0.5
            w = W_sum / denom
            x = x @ w
            x = act(x)
            x = x * (var_out ** 0.5)
        else:
            denom = (h_in * var_in / var_out) ** 0.5
            w = W_sum / denom
            x = x @ w
        return x


def _replace_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    # Works for standard submodules and for ModuleList/ModuleDict immediate children
    parent._modules[name] = new_child


def inject_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    wrap_equivariant: bool = True,
    wrap_dense: bool = True,
    verbose: bool = False,
    freeze_non_lora: bool = True,
) -> Tuple[int, int]:
    """
    Recursively replace eligible linears with LoRA-wrapped versions.
    Returns (num_equivariant_wrapped, num_dense_wrapped).
    """
    num_eq = 0
    num_dense = 0
    for child_name, child in list(module.named_children()):
        # Skip already wrapped
        if isinstance(child, (LoRAO3Linear, LoRADenseLinear)):
            continue
        # Equivariant o3.Linear
        if wrap_equivariant and isinstance(child, o3.Linear):
            try:
                wrapped = LoRAO3Linear(child, rank=rank, alpha=alpha)
            except Exception as exc:  # If no shared irreps, skip
                if verbose:
                    print(f"[LoRA] Skip {child_name}: {exc}")
            else:
                _replace_child(module, child_name, wrapped)
                num_eq += 1
                if verbose:
                    print(f"[LoRA] Wrapped equivariant {child_name}: {child.irreps_in} -> {child.irreps_out}")
                # Do not recurse into the wrapper internals (base/lora_A/lora_B)
                continue
        # Dense nn.Linear
        if wrap_dense and isinstance(child, nn.Linear):
            wrapped = LoRADenseLinear(child, rank=rank, alpha=alpha)
            _replace_child(module, child_name, wrapped)
            num_dense += 1
            if verbose:
                print(f"[LoRA] Wrapped dense {child_name}: {child.in_features} -> {child.out_features}")
            # Do not recurse into the wrapper internals
            continue
        # e3nn FullyConnectedNet internal layer
        if wrap_dense and (E3NNFCLayer is not None) and isinstance(child, E3NNFCLayer):  # type: ignore[arg-type]
            # Wrap all FC layers; LoRA forward replicates normalization exactly
            try:
                wrapped = LoRAFCLayer(child, rank=rank, alpha=alpha)
            except Exception as exc:
                if verbose:
                    print(f"[LoRA] Skip {child_name} (fc layer): {exc}")
            else:
                _replace_child(module, child_name, wrapped)
                num_dense += 1
                if verbose:
                    w = child.weight
                    print(f"[LoRA] Wrapped e3nn FC layer {child_name}: weight {tuple(w.shape)}")
                continue
        # Recurse
        sub_eq, sub_dense = inject_lora(child, rank, alpha, wrap_equivariant, wrap_dense, verbose)
        num_eq += sub_eq
        num_dense += sub_dense
    # Optionally freeze everything except LoRA A/B
    if freeze_non_lora:
        for name, p in module.named_parameters():
            if ("lora_A" in name) or ("lora_B" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False
    return num_eq, num_dense


# Backward-compatible aliases
def inject_LinLoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    inject_lora(model, rank=rank, alpha=alpha, wrap_equivariant=True, wrap_dense=False, verbose=True)
    return model


def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    inject_lora(model, rank=rank, alpha=alpha, wrap_equivariant=True, wrap_dense=True, verbose=True)
    return model
    