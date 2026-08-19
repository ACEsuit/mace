import importlib

import numpy as np
import torch

from mace.tools.torch_geometric.dataloader import DataLoader as _TgDataLoader

# --- Optional torch_geometric support ---
if importlib.util.find_spec("torch_geometric") is not None:
    from torch_geometric.transforms import BaseTransform  # pylint: disable=import-error

    has_tg = True
else:
    has_tg = False

    # --- Minimal stub classes so code still runs ---
    class BaseTransform:
        """Fallback stub if torch_geometric is unavailable."""

        def forward(self, data):
            return data


class Random3DRotation(BaseTransform):
    """
    Augment the magnetic moments with the symmetries the target model actually has.

    Two modes, because spin rotation and time reversal are not equally valid:

    ``mode="non-soc"`` (default) draws from the full O(3)_spin: a random proper
    rotation, times a random sign. A non-SOC energy is invariant under rotating the
    moments independently of the lattice, so both halves are genuine symmetries and
    both are augmented.

    ``mode="soc"`` applies ONLY the sign, m -> -m. With spin-orbit coupling the energy
    depends on how the moments are oriented relative to the lattice, so a free spin
    rotation is *not* a symmetry -- augmenting with it would teach the model an
    invariance it must not have, washing out the coupling it is meant to learn. Time
    reversal remains a symmetry at zero field, so the sign flip is kept.

    A single transform is applied per structure, preserving relative orientation.

    Assigning to `data` below is deliberate and safe. Callers reach this through
    BaseTransform.__call__, which is `self.forward(copy.copy(data))`, so the
    object here is already a shallow copy and the dataset's own sample is never
    touched. Do not add another copy: AtomicData cannot be cloned (Data.clone
    rebuilds via cls(), and AtomicData.__init__ takes 27 required arguments),
    and copying again would only cost a dict per sample per epoch.
    """

    def __init__(self, mode: str = "non-soc"):
        super().__init__()
        if mode not in ("soc", "non-soc"):
            raise ValueError(f"mode must be 'soc' or 'non-soc', got {mode!r}")
        self.mode = mode

    def forward(self, data):
        if hasattr(data, "magmom") and data.magmom is not None:
            device = data.magmom.device
            dtype = data.magmom.dtype
            sample_dtype = (
                torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
            )

            u1, u2, u3 = torch.rand(3, device=device, dtype=sample_dtype)
            q1 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
            q2 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
            q3 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
            q4 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

            R = torch.tensor(
                [
                    [
                        1 - 2 * (q3**2 + q4**2),
                        2 * (q2 * q3 - q1 * q4),
                        2 * (q2 * q4 + q1 * q3),
                    ],
                    [
                        2 * (q2 * q3 + q1 * q4),
                        1 - 2 * (q2**2 + q4**2),
                        2 * (q3 * q4 - q1 * q2),
                    ],
                    [
                        2 * (q2 * q4 - q1 * q3),
                        2 * (q3 * q4 + q1 * q2),
                        1 - 2 * (q2**2 + q3**2),
                    ],
                ],
                device=device,
                dtype=sample_dtype,
            ).to(dtype)

            # SOC: the proper rotation is NOT a symmetry, so drop it and keep only the
            # sign. non-SOC: keep both, i.e. draw from the full O(3)_spin.
            if self.mode == "soc":
                R = torch.eye(3, device=device, dtype=dtype)
            if torch.rand((), device=device) < 0.5:
                R = -R

            # === Step 3: Apply to magmom (shape [N, 3])
            data.magmom = torch.matmul(data.magmom, R.T)
            if hasattr(data, "magforces") and data.magforces is not None:
                data.magforces = torch.matmul(data.magforces, R.T)

        return data


def create_random_rotation_loader(original_loader, mode: str = "non-soc"):
    if not has_tg:
        raise ImportError(
            "torch_geometric is required for DataLoader functionality.\n"
            "Install it via: pip install torch-geometric"
        )

    transform = Random3DRotation(mode=mode)

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

    # Under distributed training the original loader has a DistributedSampler
    # already sharding the dataset per rank; replacing it with shuffle=True
    # would make every rank iterate the whole dataset, duplicating samples and
    # breaking the effective epoch size. Pass the sampler through in that case
    # (and don't also set shuffle=, which DataLoader forbids alongside a
    # sampler). Otherwise keep the previous shuffle-based behavior.
    sampler = getattr(original_loader, "sampler", None)
    is_distributed_sampler = isinstance(
        sampler, torch.utils.data.distributed.DistributedSampler
    )
    loader_kwargs = dict(
        batch_size=original_loader.batch_size,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory,
        drop_last=original_loader.drop_last,
    )
    if is_distributed_sampler:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = not isinstance(
            sampler, torch.utils.data.SequentialSampler
        )

    new_loader = _TgDataLoader(transformed_dataset, **loader_kwargs)

    return new_loader
