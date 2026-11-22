"""Named-module freezing and preservation of parameter gradient flags."""

import argparse
import logging
from contextlib import contextmanager


def parse_freeze(value):
    """Keep legacy integer levels intact; additionally accept module names."""
    try:
        return int(value)
    except ValueError:
        names = [name.strip() for name in value.split(",")]
        if any(not name for name in names):
            raise argparse.ArgumentTypeError(
                "--freeze requires an integer level or comma-separated module names"
            ) from None
        return ",".join(names)


def freeze_named_modules(model, selection):
    """Freeze named modules without re-enabling any existing frozen weights."""
    selectors = [name.strip() for name in selection.split(",")]
    if any(not name for name in selectors):
        raise ValueError("--freeze requires non-empty module names")
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters())
    frozen_ids = set()
    for selector in selectors:
        if selector not in modules:
            raise ValueError(f"--freeze module {selector!r} does not exist")
        parameter_ids = {id(param) for param in modules[selector].parameters()}
        if not parameter_ids:
            raise ValueError(f"--freeze module {selector!r} has no parameters")
        frozen_ids.update(parameter_ids)
    if not any(
        param.requires_grad and id(param) not in frozen_ids
        for param in parameters.values()
    ):
        raise ValueError("--freeze leaves no trainable parameters")
    for param in parameters.values():
        if id(param) in frozen_ids:
            param.requires_grad_(False)
    logging.info("Frozen modules: %s", ", ".join(selectors))
    logging.info(
        "Trainable tensors: %s",
        [name for name, param in parameters.items() if param.requires_grad],
    )
    logging.info(
        "Trainable parameters: %s",
        sum(param.numel() for param in parameters.values() if param.requires_grad),
    )


@contextmanager
def preserve_grad_state(model):
    """Disable parameter gradients for evaluation, restoring them even on failure."""
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        for param in model.parameters():
            param.requires_grad_(False)
        yield
    finally:
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad_(requires_grad)
