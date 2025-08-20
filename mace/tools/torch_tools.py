###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from contextlib import contextmanager
from typing import Dict, Union

import numpy as np
import torch
from e3nn.io import CartesianTensor

TensorDict = Dict[str, torch.Tensor]


def to_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generates one-hot encoding from indices.

    Args:
        indices: A tensor of shape (N x 1) containing class indices.
        num_classes: An integer specifying the total number of classes.
        dtype: The desired data type of the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (N x num_classes) containing the one-hot encodings.
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def tensor_dict_to_device(td: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) if v is not None else None for k, v in td.items()}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def init_device(device_str: str) -> torch.device:
    if "cuda" in device_str:
        assert torch.cuda.is_available(), "No CUDA device available!"
        if ":" in device_str:
            # Check if the desired device is available
            assert int(device_str.split(":")[-1]) < torch.cuda.device_count()
        logging.info(
            f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}"
        )
        torch.cuda.init()
        return torch.device(device_str)
    if device_str == "mps":
        assert torch.backends.mps.is_available(), "No MPS backend is available!"
        logging.info("Using MPS GPU acceleration")
        return torch.device("mps")
    if device_str == "xpu":
        torch.xpu.is_available()
        devices = torch.xpu.device_count()
        is_available = devices > 0
        assert is_available, logging.info("No XPU backend is available")
        torch.xpu.memory_stats()
        logging.info("Using XPU GPU acceleration")
        return torch.device("xpu")

    logging.info("Using CPU")
    return torch.device("cpu")


dtype_dict = {
    "float32": torch.float32,
    "float64": torch.float64,
    "": torch.get_default_dtype(),
    None: torch.get_default_dtype(),
}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])


def get_change_of_basis() -> torch.Tensor:
    return CartesianTensor("ij=ji").reduced_tensor_products().change_of_basis


def spherical_to_cartesian(t: torch.Tensor, change_of_basis: torch.Tensor):
    # Optionally handle device mismatch
    if change_of_basis.device != t.device:
        change_of_basis = change_of_basis.to(t.device)
    return torch.einsum("ijk,...i->...jk", change_of_basis, t)


def cartesian_to_spherical(t: torch.Tensor):
    """
    Convert cartesian notation to spherical notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def voigt_to_matrix(t: torch.Tensor):
    """Converts a tensor from Voigt notation to matrix notation.

    Args:
        t: Input tensor in one of the following formats:
            - (6,) tensor in Voigt notation
            - (3, 3) tensor in matrix notation
            - (9,) tensor that can be reshaped to (3, 3)

    Returns:
        torch.Tensor: A (3, 3) tensor in matrix notation.

    Raises:
        ValueError: If the input tensor shape is not (6,), (3, 3), or (9,).
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def init_wandb(project: str, entity: str, name: str, config: dict, directory: str):
    import wandb

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        dir=directory,
        resume="allow",
    )


@contextmanager
def default_dtype(dtype: Union[torch.dtype, str]):
    """Context manager for configuring the default_dtype used by torch

    Args:
        dtype (torch.dtype|str): the default dtype to use within this context manager
    """
    init = torch.get_default_dtype()
    if isinstance(dtype, str):
        set_default_dtype(dtype)
    else:
        torch.set_default_dtype(dtype)

    yield

    torch.set_default_dtype(init)


def check_module_dtypes(module: torch.nn.Module) -> Dict[str, torch.dtype]:
    """Check the dtype of all parameters and buffers in a torch.nn.Module.

    Args:
        module: The torch.nn.Module to check

    Returns:
        Dict mapping tensor names to their dtypes
    """
    dtypes = {}
    # Check parameters
    dtypes.update({name: param.dtype for name, param in module.named_parameters()})
    # Check buffers
    dtypes.update({name: buffer.dtype for name, buffer in module.named_buffers()})
    return dtypes
