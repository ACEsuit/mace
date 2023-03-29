###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict

import numpy as np
import torch
from e3nn.io import CartesianTensor

TensorDict = Dict[str, torch.Tensor]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

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
    if device_str == "cuda":
        assert torch.cuda.is_available(), "No CUDA device available!"
        logging.info(
            f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}"
        )
        torch.cuda.init()
        return torch.device("cuda")
    if device_str == "mps":
        assert torch.backends.mps.is_available(), "No MPS backend is available!"
        logging.info("Using MPS GPU acceleration")
        return torch.device("mps")

    logging.info("Using CPU")
    return torch.device("cpu")


dtype_dict = {"float32": torch.float32, "float64": torch.float64}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])


def get_complex_default_dtype():
    default_dtype = torch.get_default_dtype()
    if default_dtype == torch.float64:
        return torch.complex128

    if default_dtype == torch.float32:
        return torch.complex64

    raise NotImplementedError


def spherical_to_cartesian(t: torch.Tensor):
    """
    Convert spherical notation to cartesian notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def cartesian_to_spherical(t: torch.Tensor):
    """
    Convert cartesian notation to spherical notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def voigt_to_matrix(t: torch.Tensor):
    """
    Convert voigt notation to matrix notation
    :param t: (6,) tensor or (3, 3) tensor
    :return: (3, 3) tensor
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

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), but has shape {t.shape}"
    )


def init_wandb(project: str, entity: str, name: str, config: dict):
    import wandb

    wandb.init(project=project, entity=entity, name=name, config=config)


class DataParallelModel(torch.nn.Module):
    def __init__(self, model):
        super(DataParallelModel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()

    def forward(self, batch, training, compute_force, compute_virials, compute_stress):
        return self.model(
            batch,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)
