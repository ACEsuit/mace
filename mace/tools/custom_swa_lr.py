# Patched SWALR scheduler for Stage Two, if soft-freezing (learning rate rescaling) requested
# Keeps the defaults of SWALR, just introduces non-uniform lr
import math

from torch.optim.swa_utils import SWALR


class CustomSWALR(SWALR):
    def __init__(self, optimizer, swa_lr, **kwargs):
        # Extract anneal settings early
        self.anneal_epochs = kwargs.get("anneal_epochs", 1)
        self.anneal_strategy = kwargs.get("anneal_strategy", "linear")
        self.swa_lr = swa_lr

        # Compute lr scaling ratios
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        self.lr_ratios = [group["lr"] / max_lr for group in optimizer.param_groups]
        self.base_lrs = [swa_lr * r for r in self.lr_ratios]

        # Call parent constructor (sets up internal state)
        super().__init__(optimizer, swa_lr=swa_lr, **kwargs)

    def get_lr(self):
        anneal_step = getattr(self, "_anneal_step", 0)
        if self.anneal_strategy == "linear":
            alpha = 1.0 - anneal_step / self.anneal_epochs
        elif self.anneal_strategy == "cos":
            alpha = 0.5 * (1 + math.cos(math.pi * anneal_step / self.anneal_epochs))
        else:
            raise ValueError(f"Invalid annealing strategy: {self.anneal_strategy}")

        return [
            self.swa_lr + (base_lr - self.swa_lr) * alpha for base_lr in self.base_lrs
        ]
