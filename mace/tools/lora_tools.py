from e3nn import o3
import torch
import torch.nn as nn

def inject_LinLoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    """
    Inject Low-Rank Adaptation (LoRA) layers into the given model.
    
    Args:
        model (nn.Module): The neural network model to modify.
        rank (int): The rank for the LoRA layers.
        alpha (int): Scaling factor for the LoRA layers.
    """

    for child_name, child in list(model.named_children()): 
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, o3.Linear):
            print(f"Input features: {child.irreps_in}")  
            print(f"Output features: {child.irreps_out}") 
            print(f"Injecting LoRA into module: {child_name}")
            setattr(model, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_LoRAs(child, rank=rank, alpha=alpha)
    return model

def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    """
    Inject Low-Rank Adaptation (LoRA) layers into the given model.
    
    Args:
        model (nn.Module): The neural network model to modify.
        rank (int): The rank for the LoRA layers.
        alpha (int): Scaling factor for the LoRA layers.
    """

    for child_name, child in list(model.named_children()): 
        if isinstance(child, LoRAEquivariantLayer):
            continue
        if hasattr(child, "irreps_in") and hasattr(child, "irreps_out") and hasattr(child, "weight"):    #what are the layers that dont trigger?        
            print(f"Input features: {child.irreps_in}")  
            print(f"Output features: {child.irreps_out}") 
            print(f"Injecting LoRA into module: {child_name}")
            setattr(model, child_name, LoRAEquivariantLayer(child, rank=rank, alpha=alpha))
        else:
            inject_LoRAs(child, rank=rank, alpha=alpha)
    return model

class LoRAEquivariantLayer(nn.Module):
    original_layer: o3.Irreps
    lora_A: o3.Irreps
    lora_B: o3.Irreps
    irreps_in: o3.Irreps
    irreps_out: o3.Irreps
    lora_irreps: o3.Irreps
    scaling: float
    rank: int
    alpha: float

    def __init__(self, original_layer:o3.Irreps, rank=4, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank


        self.irreps_in = original_layer.irreps_in
        self.irreps_out = original_layer.irreps_out
        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)
        self.lora_irreps = build_lora_irreps(original_layer.irreps_in, original_layer.irreps_out, rank)
        layer_type = type(original_layer)
        # LoRA layers
        self.lora_A = layer_type(self.irreps_in, self.lora_irreps, internal_weights=True)
        self.lora_B = layer_type(self.lora_irreps, self.irreps_out, internal_weights=True)
        
        # Initialize LoRA parameters to ensure ΔW starts at 0
        with torch.no_grad():
            for p in self.lora_B.parameters():
                p.zero_()
            for p in self.lora_A.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=1e-3)  # Small std normal initialization

    def forward(self, x):
        delta = self.lora_B(self.lora_A(x))
        return self.scaling * delta

def build_lora_irreps(irreps_in: o3.Irreps,
                      irreps_out: o3.Irreps,
                      rank: int) -> o3.Irreps:
    """
    Construct internal bottleneck irreps: for every irrep type present in BOTH
    input and output allocate `rank` copies. This keeps the LoRA update
    strictly equivariant and avoids useless blocks.
    """
    in_set = {ir for _, ir in irreps_in}
    out_set = {ir for _, ir in irreps_out}
    shared = sorted(in_set & out_set, key=lambda ir: (ir.l, ir.p))
    if not shared:
        raise ValueError(
            f"No shared irreps between input ({irreps_in}) and output ({irreps_out}); "
            "cannot build an equivariant linear LoRA bottleneck."
        )
    parts = [f"{rank}x{ir}" for ir in shared]
    lora_irreps = o3.Irreps(" + ".join(parts))
    return lora_irreps

class LoRALinear(nn.Module):
    linear_layer: o3.Linear
    lora_A: o3.Linear
    lora_B: o3.Linear
    irreps_in: o3.Irreps
    irreps_out: o3.Irreps
    lora_irreps: o3.Irreps
    scaling: float
    rank: int
    alpha: float

    def __init__(self, linear_layer: nn.Module, rank=4, alpha=1):
        super().__init__()
        self.linear_layer = linear_layer
        self.linear_layer.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_irreps = o3.Irreps(f"{rank}x0e")
        self.irreps_in = linear_layer.irreps_in
        self.irreps_out = linear_layer.irreps_out
        # Create the trainable LoRA matrices
        self.lora_A = o3.Linear(linear_layer.irreps_in, self.lora_irreps, internal_weights=True)
        self.lora_B = o3.Linear(self.lora_irreps, linear_layer.irreps_out, internal_weights=True)
        
        # Initialize LoRA parameters, so ΔW starts at 0
        with torch.no_grad():
            for p in self.lora_B.parameters():
                p.zero_()

        with torch.no_grad():
            for p in self.lora_A.parameters():
                if p.dim() >= 2:
                    # small std normal (keeps update tiny)
                    p.normal_(mean=0.0, std=1e-3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear_layer(x)          
        delta = self.lora_B(self.lora_A(x))  
        return base + self.scaling * delta