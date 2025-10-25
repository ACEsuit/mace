
import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class Random3DRotation(BaseTransform):
    """
    Apply a random SO(3) rotation to all magnetic moments in a configuration.
    A single rotation is applied per structure, preserving relative orientation.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        if hasattr(data, 'magmom') and data.magmom is not None:
            device = data.magmom.device
            dtype = data.magmom.dtype

            # === Step 1: Sample a random unit quaternion (uniform over SO(3))
            u1, u2, u3 = torch.rand(3, device=device, dtype=dtype)
            q1 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
            q2 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
            q3 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
            q4 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

            # === Step 2: Convert quaternion to rotation matrix
            R = torch.tensor([
                [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
                [2*(q2*q3 + q1*q4),     1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2)],
                [2*(q2*q4 - q1*q3),         2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)]
            ], device=device, dtype=dtype)

            # === Step 3: Apply to magmom (shape [N, 3])
            data.magmom = torch.matmul(data.magmom, R.T)

        return data

def create_random_rotation_loader(original_loader):
    """
    Create a new DataLoader with hemisphere rotation augmentation.
    
    Args:
        original_loader: Original PyTorch Geometric DataLoader
    
    Returns:
        New DataLoader with hemisphere rotation transform
    """
    transform = Random3DRotation()
    
    # Apply transform to dataset
    dataset = original_loader.dataset
    
    # Create new dataset with transform
    class TransformedDataset:
        def __init__(self, original_dataset, transform):
            self.dataset = original_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data = self.dataset[idx]
            return self.transform(data)
    
    transformed_dataset = TransformedDataset(dataset, transform)
    
    is_shuffle = False if original_loader.sampler.__class__ == torch.utils.data.sampler.SequentialSampler else True

    # Create new DataLoader with same parameters
    new_loader = torch_geometric.dataloader.DataLoader(
        transformed_dataset,
        batch_size=original_loader.batch_size,
        shuffle=is_shuffle,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last
    )
    
    return new_loader