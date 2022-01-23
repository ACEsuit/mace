import torch

class BesselBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_basis=8):
        super().__init__()

        bessel_weights = np.pi * torch.linspace(
            start=1.0, end=num_basis, steps=num_basis, dtype=torch.get_default_dtype())
        r_max_tensor = torch.tensor(r_max, dtype=torch.get_default_dtype())

        self.register_buffer('bessel_weights', bessel_weights)
        self.register_buffer('r_max', r_max_tensor)
        self.register_buffer('pre_factor', 2.0 / r_max_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)  # [..., num_basis]
        return self.pre_factor * (numerator / x.unsqueeze(-1))