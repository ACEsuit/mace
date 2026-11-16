"""Kernel backends for torch. The reference one is mandatory and always here."""

from mace_core.kernels import KernelSpec, UnsupportedKernelSpecError

from mace_torch.backends.reference import ReferenceBackend

__all__ = ["ReferenceBackend", "require_built"]


def require_built(backend, kernel_spec: KernelSpec):
    """The op ``backend`` builds for ``kernel_spec``, refusing a declined one.

    The protocol lets a backend decline some ops and leave them to the
    reference. The model does not fall back case by case, so a declined op is
    refused here, at build, rather than failing on the first call.
    """
    op = getattr(backend, f"make_{kernel_spec.op}")(kernel_spec)
    if op is None:
        raise UnsupportedKernelSpecError(
            f"backend {backend.name!r} declined {kernel_spec.op!r} for "
            f"{kernel_spec!r}, and a model runs every op on the backend it is "
            f"handed. Build the model on a backend that builds this op, such "
            f"as the reference."
        )
    return op
