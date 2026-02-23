from collections.abc import Mapping, Sequence
from typing import Optional

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from .batch import Batch
from .data import Data
from .dataset import Dataset


class Collater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        if isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        if isinstance(elem, int):
            return torch.tensor(batch)
        if isinstance(elem, str):
            return batch

        # For these types, we can use a more generic approach or keep them as is if needed
        # but to reduce return statements, we can group them.
        if isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        if isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        if isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")

    def collate(self, batch):  # Deprecated...
        return self(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[list[str]] = None,
        exclude_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        if follow_batch is None:
            follow_batch = [None]
        if exclude_keys is None:
            exclude_keys = [None]
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )
