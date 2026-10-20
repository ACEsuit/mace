"""Kernel backends for torch. The reference one is mandatory and always here."""

from mace_torch.backends.reference import ReferenceBackend

__all__ = ["ReferenceBackend"]
