import logging

import torch


def _clamp_index(n: int, total: int) -> int:
    """Clamp n so that |n| never exceeds total."""
    if abs(n) > total:
        logging.warning(f"Requested {n}, but only {total} available. Adjusting.")
        return total if n > 0 else -total
    return n


def freeze_layers(model: torch.nn.Module, n: int) -> None:
    """
    Freezes the first `n` layers of a model. If `n` is negative, freezes the last `|n|` layers.
    Args:
        model (torch.nn.Module): The model.
        n (int): The number of layers to freeze (0/None = no freeze).
    """
    if not n:
        logging.info("No layers frozen.")
        return

    layers = list(model.children())
    num_layers = len(layers)
    n = _clamp_index(n, num_layers)

    frozen_layers = layers[:n] if n > 0 else layers[n:]
    frozen_count = len(frozen_layers)

    logging.info(f"Total layers in model: {num_layers}")
    logging.info(f"Freezing {frozen_count} layers.")

    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False


def freeze_param(model: torch.nn.Module, n: int) -> None:
    """
    Freezes the first `n` named parameters of a model. If `n` is negative, freezes the last `|n|` parameters.
    Args:
        model (torch.nn.Module): The model.
        n (int): The number of parameters to freeze (0/None = no freeze).
    """
    if not n:
        logging.info("No parameter groups frozen (n=0 or None).")
        return

    param_list = list(model.named_parameters())
    num_params = len(param_list)
    n = _clamp_index(n, num_params)

    freeze_until = n if n > 0 else num_params + n
    frozen_count = freeze_until if n > 0 else num_params - freeze_until

    logging.info(f"Total named parameters: {num_params}")
    logging.info(f"Freezing {frozen_count} parameter groups.")

    for idx, (_name, param) in enumerate(param_list):
        if idx < freeze_until:
            param.requires_grad = False
