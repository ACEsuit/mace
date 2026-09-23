"""Converting a legacy checkpoint: extract where legacy lives, import here.

A legacy checkpoint is a pickle, so reading it executes the legacy classes it
was written from. That never happens in this process. The extraction script
ships inside this package as a file, not a module, and runs as a subprocess of
an interpreter that has the legacy package: the one given, or an environment
this module provisions with a pinned legacy release. What comes back is the
neutral artifact, two files with no pickle in either, and the import side reads
only those.

Two modes, one script:

* **development**: an interpreter that already has the in-tree legacy package,
  which is how the published models are converted and how the tests run.
* **packaged**: an isolated environment built from a pinned legacy release,
  for a user who has only v1 installed and a checkpoint of their own.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from mace_core.observables import ObservableCatalogue

from mace_torch.deploy.neutral_io import ImportedModel, import_neutral
from mace_torch.deploy.reference import (
    VerificationError,
    VerificationReport,
    verify_against_reference,
)

__all__ = [
    "PINNED_LEGACY",
    "Conversion",
    "ExtractionFailed",
    "convert_legacy",
    "extract",
    "extractor_path",
    "pinned_environment",
]

#: The legacy release a packaged extraction runs against. The oldest one every
#: published foundation model unpickles under.
PINNED_LEGACY = "mace-torch==0.3.17"

#: What the pinned environment needs besides the legacy release: the writer of
#: the tensor file.
PINNED_EXTRAS = ("safetensors>=0.4",)


class ExtractionFailed(RuntimeError):
    """The extraction did not produce an artifact."""


@dataclass(frozen=True)
class Conversion:
    """What a conversion produced.

    Attributes:
        neutral: The neutral artifact's sidecar.
        checkpoint: The v1 checkpoint's sidecar.
        model: The converted model.
        report: The comparison against a reference, when one was given.
    """

    neutral: Path
    checkpoint: Path
    model: ImportedModel
    report: VerificationReport | None


def extractor_path() -> Path:
    """The extraction script, as installed with this package."""
    path = Path(str(files("mace_torch") / "legacy-extractor" / "extract_legacy.py"))
    if not path.exists():
        raise ExtractionFailed(
            f"the extraction script is not at {path}. It ships with this "
            f"package, so the installation is incomplete."
        )
    return path


def extract(
    source: str | Path,
    output: str | Path,
    *,
    python: str | Path | None = None,
    timeout: float | None = None,
) -> Path:
    """Run the extraction script under ``python`` and return the sidecar.

    Args:
        source: The pickled legacy checkpoint.
        output: Where the artifact goes, without a suffix.
        python: An interpreter that has the legacy package. The current one by
            default, which is right only when it has it.
        timeout: Seconds before the extraction is abandoned.

    Raises:
        ExtractionFailed: With the script's own message when it refuses, and
            its whole error output when it fails any other way.
    """
    interpreter = str(python or sys.executable)
    completed = subprocess.run(
        [interpreter, str(extractor_path()), str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        refused = [
            line.removeprefix("extraction refused: ")
            for line in completed.stderr.splitlines()
            if line.startswith("extraction refused: ")
        ]
        if refused:
            raise ExtractionFailed(refused[-1])
        raise ExtractionFailed(
            f"the extraction under {interpreter} failed with exit status "
            f"{completed.returncode}:\n{completed.stderr.strip()}"
        )
    sidecar = Path(completed.stdout.strip().splitlines()[-1])
    if not sidecar.exists():
        raise ExtractionFailed(
            f"the extraction reported {sidecar} and wrote nothing there."
        )
    return sidecar


def pinned_environment(
    directory: str | Path,
    *,
    pin: str = PINNED_LEGACY,
    timeout: float | None = None,
) -> Path:
    """An environment with the pinned legacy release, built once and reused.

    Args:
        directory: Where it lives. Reused when it already has the release.
        pin: The requirement to install.
        timeout: Seconds before the installation is abandoned.

    Returns:
        The environment's interpreter.

    Raises:
        ExtractionFailed: If the environment cannot be created or the release
            cannot be installed, with the installer's output, since a missing
            wheel for this platform and a missing network look alike from here.
    """
    root = Path(directory)
    interpreter = root / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    marker = root / "mace-legacy-pin.txt"
    if interpreter.exists() and marker.exists() and marker.read_text() == pin:
        return interpreter
    # Linked, not copied, as `python -m venv` does outside Windows. A copied
    # interpreter can be one that runs only from where it was installed:
    # measured with a uv-managed CPython on macOS, the copy aborts on its
    # first run, inside the pip bootstrap.
    builder = venv.EnvBuilder(
        with_pip=True, clear=True, symlinks=sys.platform != "win32"
    )
    try:
        builder.create(root)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExtractionFailed(
            f"could not create an environment at {root} with {sys.executable}: {error}"
        ) from error
    completed = subprocess.run(
        [str(interpreter), "-m", "pip", "install", "--quiet", pin, *PINNED_EXTRAS],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise ExtractionFailed(
            f"could not install {pin} into {root}. The extraction needs that "
            f"release to unpickle the checkpoint, and nothing else can stand in "
            f"for it. The installer said:\n{completed.stderr.strip()}"
        )
    marker.write_text(pin)
    return interpreter


def convert_legacy(
    source: str | Path,
    output: str | Path,
    catalogue: ObservableCatalogue,
    *,
    python: str | Path | None = None,
    reference: str | Path | None = None,
    fixtures: str | Path | None = None,
    head: int = 0,
) -> Conversion:
    """Extract, import, verify when asked, and write a v1 checkpoint.

    Args:
        source: The pickled legacy checkpoint.
        output: The v1 checkpoint's name, without a suffix. The neutral
            artifact is written beside it, its name ending in ``-neutral``.
        catalogue: The observable declarations.
        python: The interpreter the extraction runs under. See
            :func:`pinned_environment` for one built from a pinned release.
        reference: A golden-harness JSON written from the source model.
        fixtures: The directory of structures the reference names. Required
            with ``reference``.
        head: Which head the reference was written for.

    Raises:
        ExtractionFailed: If the extraction refuses or fails.
        NeutralImportError: If the artifact cannot be built faithfully.
        VerificationError: If the converted model does not reproduce the
            reference. Nothing is written in that case.
    """
    from mace_torch.train.checkpoint import write_model

    target = Path(output)
    neutral = extract(source, target.with_name(target.name + "-neutral"), python=python)
    imported = import_neutral(neutral, catalogue)
    report = None
    if reference is not None:
        if fixtures is None:
            raise ValueError(
                "a reference was given without the directory of structures it "
                "names, so there is nothing to evaluate the model on."
            )
        report = verify_against_reference(
            imported.engine,
            reference,
            fixtures,
            z_table=imported.z_table,
            cutoff=imported.config.model.r_max,
            head=head,
        )
        if not report.passed:
            raise VerificationError(report)
    checkpoint = write_model(target, imported.engine, imported.metadata)
    return Conversion(
        neutral=neutral, checkpoint=checkpoint, model=imported, report=report
    )
