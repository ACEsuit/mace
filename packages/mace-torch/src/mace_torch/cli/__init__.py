"""Console entry points, and nothing else.

Each module here parses, builds the configuration and calls the stages. No
logic lives in this package: what a run does belongs to
:mod:`mace_torch.train`, and a command line that carried behaviour of its own
would be a second place for it that only the command line can reach.
"""

from __future__ import annotations

import sys

from mace_core.cli import main


def _command(*path: str) -> int:
    return main([*path, *sys.argv[1:]])


def run_train_main() -> int:
    return _command("train")


def eval_configs_main() -> int:
    return _command("eval")


def prepare_data_main() -> int:
    return _command("data", "prepare")


def finetuning_select_main() -> int:
    return _command("data", "select")


def plot_train_main() -> int:
    return _command("plot", "training")


def convert_device_main() -> int:
    return _command("model", "convert-device")


def select_head_main() -> int:
    return _command("model", "select-head")


def polar_density_cube_main() -> int:
    return _command("eval", "density-cube")
