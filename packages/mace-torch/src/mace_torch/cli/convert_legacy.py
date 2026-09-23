"""Converting a legacy checkpoint into a v1 one, from a command line.

A v0.3 checkpoint is not loadable by v1; it is convertible, once, with this.
The extraction runs under an interpreter that has the legacy package: the one
named with ``--python``, or an environment built from a pinned legacy release
with ``--pinned-environment``. Without either it runs under the current
interpreter, which works only where the legacy package is installed beside v1.

Given a reference and its structures, the converted model is checked against
them before anything is written, and a model that does not reproduce them is
not written at all.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mace_core.observables import load_default_catalogue

from mace_torch.deploy.legacy import (
    PINNED_LEGACY,
    ExtractionFailed,
    convert_legacy,
    pinned_environment,
)
from mace_torch.deploy.neutral_io import NeutralImportError
from mace_torch.deploy.reference import ReferenceError, VerificationError

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace_convert_legacy",
        description="Convert a pickled v0.3 checkpoint into a v1 checkpoint.",
    )
    parser.add_argument("source", type=Path, help="the pickled legacy checkpoint")
    parser.add_argument(
        "output",
        type=Path,
        help="the v1 checkpoint's name, without a suffix",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--python",
        type=Path,
        help="an interpreter that has the legacy package installed",
    )
    where.add_argument(
        "--pinned-environment",
        type=Path,
        metavar="DIRECTORY",
        help=(
            f"build, or reuse, an environment here with {PINNED_LEGACY} and run "
            f"the extraction in it"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="a golden-harness JSON written from the source model",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        help="the directory of structures the reference names",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="which head the reference was written for, by position",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (arguments.reference is None) != (arguments.fixtures is None):
        print(
            "--reference and --fixtures go together: one names the values, the "
            "other the structures they were computed on.",
            file=sys.stderr,
        )
        return 2
    try:
        python = (
            pinned_environment(arguments.pinned_environment)
            if arguments.pinned_environment is not None
            else arguments.python
        )
        conversion = convert_legacy(
            arguments.source,
            arguments.output,
            load_default_catalogue(),
            python=python,
            reference=arguments.reference,
            fixtures=arguments.fixtures,
            head=arguments.head,
        )
    except (ExtractionFailed, NeutralImportError, ReferenceError) as error:
        print(f"conversion refused: {error}", file=sys.stderr)
        return 2
    except VerificationError as error:
        print(str(error), file=sys.stderr)
        return 1
    if conversion.report is not None:
        print(conversion.report.describe(), file=sys.stderr)
    print(conversion.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
