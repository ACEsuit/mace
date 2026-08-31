"""Console entry points for MACE, dispatching to the legacy or the v1 engine.

Every ``mace_*`` script installed on the system comes from here. The engine is
chosen by ``--engine {legacy,v1}`` or the ``MACE_ENGINE`` environment variable,
and defaults to ``legacy``, so an installed MACE behaves exactly as it did
before the launcher existed.

The engine flag is removed from ``argv`` before the target's parser ever sees
it, and nothing else is touched: no reordering, no re-expansion, no second
parsing pass. That matters most for ``--config``, which both legacy training
parsers register with configargparse as a YAML config file. An extra pass over
argv there would silently change which value wins.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Callable, List, Tuple

from mace_launcher import audit

__all__ = [
    "active_learning_md",
    "convert_device",
    "create_lammps_model",
    "cueq_to_e3nn",
    "e3nn_cueq",
    "eval_configs",
    "finetuning_select",
    "plot_train",
    "polar_density_cube",
    "prepare_data",
    "run_train",
    "select_head",
]

ENGINES = ("legacy", "v1")
DEFAULT_ENGINE = "legacy"
ENGINE_ENV_VAR = "MACE_ENGINE"

#: script name -> module basename, shared by both engines. Four of the twelve
#: scripts are not named after their module, so this cannot be derived.
TARGETS = {
    "mace_active_learning_md": "active_learning_md",
    "mace_create_lammps_model": "create_lammps_model",
    "mace_eval_configs": "eval_configs",
    "mace_plot_train": "plot_train",
    "mace_run_train": "run_train",
    "mace_prepare_data": "preprocess_data",
    "mace_finetuning_select": "fine_tuning_select",
    "mace_convert_device": "convert_device",
    "mace_select_head": "select_head",
    "mace_e3nn_cueq": "convert_e3nn_cueq",
    "mace_cueq_to_e3nn": "convert_cueq_e3nn",
    "mace_polar_density_cube": "polar_density_cube",
}


def _take_engine(argv: List[str]) -> Tuple[str, List[str]]:
    """Pull ``--engine`` out of argv, returning the engine and what remains.

    Accepts ``--engine v1`` and ``--engine=v1``. Everything else is passed
    through untouched, including a bare ``--`` and anything after it, so an
    argument that merely looks like the flag reaches the target intact.
    """
    engine = os.environ.get(ENGINE_ENV_VAR, DEFAULT_ENGINE)
    remaining: List[str] = []
    index = 0
    saw_separator = False
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            saw_separator = True
        if not saw_separator and argument == "--engine":
            if index + 1 >= len(argv):
                raise SystemExit("--engine needs a value: one of legacy, v1")
            engine = argv[index + 1]
            index += 2
            continue
        if not saw_separator and argument.startswith("--engine="):
            engine = argument.split("=", 1)[1]
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if engine not in ENGINES:
        raise SystemExit(
            f"unknown engine {engine!r}: expected one of {', '.join(ENGINES)}"
        )
    return engine, remaining


def _load_main(script: str, engine: str) -> Callable[[], object]:
    """Import the target module and return its ``main``.

    A v1 capability that does not exist yet is a routine state during the
    migration, so it gets a sentence naming the script rather than an
    ImportError naming a module the user never typed.
    """
    module_basename = TARGETS[script]
    if engine == "v1":
        audit.install()
        try:
            module = importlib.import_module(f"mace_torch.cli.{module_basename}")
        except ImportError as error:
            raise SystemExit(
                f"capability {script} not yet available on v1 engine "
                f"(run it with --engine legacy, or MACE_ENGINE=legacy). "
                f"Underlying import error: {error}"
            ) from error
    else:
        module = importlib.import_module(f"mace.cli.{module_basename}")
    return module.main


def _dispatch(script: str) -> None:
    """Run one console script on the selected engine."""
    engine, argv = _take_engine(sys.argv[1:])
    main = _load_main(script, engine)
    # The target reads sys.argv itself rather than taking it as a parameter,
    # which is why this is assigned instead of passed.
    sys.argv = [sys.argv[0], *argv]
    main()


# One function per console script. They are spelled out rather than generated
# so that an entry point is greppable from its name, and so that a typo in
# pyproject.toml fails at install time instead of at first run.
def active_learning_md() -> None:
    _dispatch("mace_active_learning_md")


def create_lammps_model() -> None:
    _dispatch("mace_create_lammps_model")


def eval_configs() -> None:
    _dispatch("mace_eval_configs")


def plot_train() -> None:
    _dispatch("mace_plot_train")


def run_train() -> None:
    _dispatch("mace_run_train")


def prepare_data() -> None:
    _dispatch("mace_prepare_data")


def finetuning_select() -> None:
    _dispatch("mace_finetuning_select")


def convert_device() -> None:
    _dispatch("mace_convert_device")


def select_head() -> None:
    _dispatch("mace_select_head")


def e3nn_cueq() -> None:
    _dispatch("mace_e3nn_cueq")


def cueq_to_e3nn() -> None:
    _dispatch("mace_cueq_to_e3nn")


def polar_density_cube() -> None:
    _dispatch("mace_polar_density_cube")
