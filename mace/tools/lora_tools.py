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
    for name, module in model.named_modules():
        print(f"Module {name}: {module.__class__.__name__}")
        if isinstance(module, o3.Linear):
            print(f"Input features: {module.in_features}")  
            print(f"Output features: {module.out_features}") 
            print(f"Injecting LoRA into module: {name}")
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            lora_layer=lora_layer.to(dtype=module.weight.dtype, device=module.weight.device)
            setattr(module, name, lora_layer)
    return model



class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Module, rank=4, alpha=1):
        super().__init__()
        self.linear_layer = linear_layer
        self.liner_layer.requires_grad_(False)
        self.scaling = alpha / rank
        
        self.lora_A = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, linear_layer.out_features, bias=False)
        
        # Initialize LoRA parameters, so ΔW starts at 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)  
        nn.init.zeros_(self.lora_B.weight) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer + self.scaling * self.lora_B(self.lora_A(x))