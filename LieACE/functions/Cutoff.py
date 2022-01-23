import torch

class PolynomialCutoff(torch.nn.Module):
    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer('p', torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer('r_max', torch.tensor(r_max, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max).type(torch.get_default_dtype())