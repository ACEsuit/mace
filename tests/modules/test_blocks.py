import numpy as np
import torch

from mace.modules.blocks import AtomicEnergiesBlock, ScaleShiftBlock


def test_atomic_energies_block_to():
    # Create test data
    energies = np.array([[1.0, 3.0, 4.0]])
    block = AtomicEnergiesBlock(atomic_energies=energies).to(torch.float32)

    # Test dtype conversion
    block.to(dtype=torch.float64)
    assert block.atomic_energies.dtype == torch.float64
    assert torch.allclose(block.atomic_energies, torch.tensor(energies, dtype=torch.float64))

    # Test device conversion
    if torch.cuda.is_available():
        block.to(device='cuda')
        assert block.atomic_energies.device.type == 'cuda'

    # Test both dtype and device
    block.to(dtype=torch.float32, device='cpu')
    assert block.atomic_energies.dtype == torch.float32
    assert block.atomic_energies.device.type == 'cpu'

    # Test forward pass still works
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    output = block(x)
    expected = torch.tensor([[1.0], [3.0], [4.0]], dtype=torch.float32)
    assert torch.allclose(output, expected)

def test_scale_shift_block_to():
    # Create test data
    scale = 2.0
    shift = 1.0
    block = ScaleShiftBlock(scale=scale, shift=shift).to(torch.float32)

    # Test dtype conversion
    block.to(dtype=torch.float64)
    assert block.scale.dtype == torch.float64
    assert block.shift.dtype == torch.float64
    assert torch.allclose(block.scale, torch.tensor(scale, dtype=torch.float64))
    assert torch.allclose(block.shift, torch.tensor(shift, dtype=torch.float64))

    # Test device conversion
    if torch.cuda.is_available():
        block.to(device='cuda')
        assert block.scale.device.type == 'cuda'
        assert block.shift.device.type == 'cuda'

    # Test both dtype and device
    block.to(dtype=torch.float32, device='cpu')
    assert block.scale.dtype == torch.float32
    assert block.shift.dtype == torch.float32
    assert block.scale.device.type == 'cpu'
    assert block.shift.device.type == 'cpu'

    # Test forward pass still works
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    head = torch.tensor([0, 0], dtype=torch.long)
    output = block(x, head)
    expected = torch.tensor([3.0, 5.0], dtype=torch.float32)  # 2.0 * x + 1.0
    assert torch.allclose(output, expected)

    # Test with multiple scales/shifts
    multi_block = ScaleShiftBlock(scale=[2.0, 3.0], shift=[1.0, 2.0]).to(torch.float32)
    multi_block.to(dtype=torch.float64)
    assert multi_block.scale.dtype == torch.float64
    assert multi_block.shift.dtype == torch.float64
    assert torch.allclose(multi_block.scale, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(multi_block.shift, torch.tensor([1.0, 2.0], dtype=torch.float64))
