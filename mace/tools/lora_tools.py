from e3nn import o3
import torch
import torch.nn as nn

def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
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



class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Module, rank=4, alpha=1):
        super().__init__()
        self.linear_layer = linear_layer
        self.linear_layer.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_irreps = o3.Irreps(f"{rank}x0e")

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