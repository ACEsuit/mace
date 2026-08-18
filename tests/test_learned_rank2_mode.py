import torch

from mace.modules.models import (
    _learned_rank2_features,
    _symmetric_tensor_from_six,
)


def test_symmetric_tensor_from_six():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    tensor = _symmetric_tensor_from_six(values)

    expected = torch.tensor(
        [[[1.0, 4.0, 5.0],
          [4.0, 2.0, 6.0],
          [5.0, 6.0, 3.0]]]
    )

    assert torch.allclose(tensor, expected)


def test_learned_rank2_features_have_gradients():
    params = torch.nn.Parameter(
        torch.tensor([[1.0, 0.8, 0.6, 0.05, -0.03, 0.02]], dtype=torch.float64)
    )
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
        ],
        dtype=torch.float64,
    )

    tensor, irreps = _learned_rank2_features(quaternions, params)
    loss = tensor.square().sum() + irreps.square().sum()
    loss.backward()

    assert tensor.shape == (2, 3, 3)
    assert irreps.shape == (2, 6)
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
    assert params.grad.abs().sum() > 0.0
