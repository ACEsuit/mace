import logging

import torch


def freeze_layers(model: torch.nn.Module, n: int) -> None:
    """
    Freezes the first `n` layers of a model. If `n` is negative, freezes the last `|n|` layers.

    Args:
        model (torch.nn.Module): The model.
        n (int): The number of layers to freeze.
    """
    layers = list(model.children())
    num_layers = len(layers)

    logging.info(f"Total layers in model: {num_layers}")

    if abs(n) > num_layers:
        logging.warning(
            f"Requested {n} layers, but model only has {num_layers}. Adjusting `n` to fit the model."
        )
        n = num_layers if n > 0 else -num_layers

    frozen_layers = layers[:n] if n > 0 else layers[n:]

    logging.info(f"Freezing {len(frozen_layers)} layers.")

    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False


def freeze_param(model: torch.nn.Module, n: int) -> None:
    """
    Freezes the first `n` named parameters of a model. If `n` is negative, freezes the last `|n|` parameters.

    Args:
        model (torch.nn.Module): The model.
        n (int): The number of parameters to freeze.
    """
    param_list = list(model.named_parameters())
    num_params = len(param_list)

    logging.info(f"Total named parameters: {num_params}")

    if abs(n) > num_params:
        logging.warning(
            f"Requested {n} parameter groups, but model only has {num_params}. Adjusting `n` to fit the model."
        )
        n = num_params if n > 0 else -num_params

    freeze_until = n if n > 0 else num_params + n

    for idx, param in enumerate(param_list):
        param.requires_grad = (
            idx >= freeze_until
        )  # Freeze first `n` or last `|n|` parameters

    logging.info(
        f"Froze {freeze_until if n > 0 else num_params - freeze_until} parameter groups."
    )
