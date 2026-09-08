#!/usr/bin/env python3
"""Generate the CI configuration that `capabilities.toml` owns.

Three regions of two workflow files are derived from the manifest rather than
maintained beside it:

  * the `outputs:` map and the `filters:` block of the `changes` job in
    `ci-extensions.yaml`, which decide which extension jobs a pull request
    starts;
  * the per-file coverage floors enforced in `nightly.yaml`'s
    `coverage-report`, and the informative listing of the same files in
    `ci-core.yaml`'s `coverage` job.

WHY GENERATED. Both are lists that only ever go stale in the direction nothing
checks. A capability that ports to `packages/` keeps an extension job watching
paths that no longer hold its code, and keeps a coverage floor on a legacy
file that `coverage report --include` then matches nothing against -- which
exits 1 with "No data to report", a message that reads as a broken coverage
run rather than as a floor whose subject moved. Driving both from one manifest
makes a RET-* pull request touch the manifest and the deletion, and a default
flip touch the manifest and the flag default.

WHY BOTH COVERAGE SHAPES. The nightly gate and the per-pull-request job
measure different denominators, and the same percentage means a different
thing in each: `mace/modules/utils.py` reads 87% over the nightly's full scope
and 76% over the pull request's `tests/unit + tests/workflows` slice, against
a floor of 85. So the nightly listing is a **gate** (`--fail-under`) and the
pull request's is **informative** (the percentage and the floor, side by side,
no threshold). Emitting only the nightly's would leave the per-pull-request
job with no rendering of the floors at all; emitting a threshold into it would
fail honest pull requests on a number measured somewhere else.

THE VALUES ARE NOT DECIDED HERE. The floors were measured under the selection
`nightly.yaml` documents; this file moves them, it does not choose them.

    python3 tests/architecture/generate_ci.py            # same as --check
    python3 tests/architecture/generate_ci.py --check    # exit 1 on drift
    python3 tests/architecture/generate_ci.py --write    # rewrite in place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Importable both as a script and as `tests.architecture.generate_ci`, so the
# meta-test and the command line share one module rather than two copies of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.architecture import capabilities  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

CI_EXTENSIONS = WORKFLOWS / "ci-extensions.yaml"
CI_CORE = WORKFLOWS / "ci-core.yaml"
NIGHTLY = WORKFLOWS / "nightly.yaml"

#: Region name -> (file, the indentation the block sits at). A region is
#: delimited in the workflow by
#:     # >>> generated from capabilities.toml: <name>
#:     # <<< generated from capabilities.toml: <name>
#: and everything between the markers is replaced. The markers are comments in
#: every position they are used, including inside the `filters:` block scalar,
#: which dorny/paths-filter parses as YAML.
BEGIN = "# >>> generated from capabilities.toml: {name}"
END = "# <<< generated from capabilities.toml: {name}"

GENERATOR = "tests/architecture/generate_ci.py"


def _filter_outputs() -> List[str]:
    """The `changes` job's outputs: one per capability that owns a filter.

    The `|| 'true'` fallback is what makes the job correct outside a pull
    request: the filter step is skipped on push, dispatch and schedule, every
    output is then empty, and the fallback runs everything.
    """
    lines = [
        "# Outside a pull request the filter step below is skipped and every",
        "# output is empty, so the `|| 'true'` fallback runs everything.",
    ]
    for entry in capabilities.ci_filters():
        lines.append(
            f"{entry.name}: ${{{{ steps.filter.outputs.{entry.name} || 'true' }}}}"
        )
    return lines


def _wrap(text: str, width: int, prefix: str) -> List[str]:
    """Wrap a note into comment lines, so a long rationale stays readable."""
    words = text.split()
    lines: List[str] = []
    current = prefix
    for word in words:
        candidate = f"{current} {word}" if current != prefix else f"{prefix}{word}"
        if len(candidate) > width and current != prefix:
            lines.append(current)
            current = f"{prefix}{word}"
        else:
            current = candidate
    if current != prefix:
        lines.append(current)
    return lines


def _filter_body() -> List[str]:
    """The paths-filter `filters:` document.

    The shared set is emitted as a YAML anchor rather than repeated, exactly as
    the hand-written version did: a change to the model, the tools, the data
    layer or the capability contract can break any extension, so every filter
    starts from it.
    """
    lines = ["shared: &shared"]
    lines += [f"  - '{path}'" for path in capabilities.shared_ci_paths()]
    for entry in capabilities.ci_filters():
        lines.append(f"{entry.name}:")
        lines.append("  - *shared")
        for path in entry.paths:
            note = entry.path_notes.get(path)
            if note:
                lines += _wrap(note, 78, "  # ")
            lines.append(f"  - '{path}'")
    return lines


def _floor_rows() -> List[Tuple[str, int, str]]:
    return [(f.path, f.floor, f.owner) for f in capabilities.coverage_floors()]


def _floor_lines() -> List[str]:
    """The floor rows themselves, grouped under the capability that owns them.

    Two whitespace-separated fields per data line and nothing else. Both
    consumers read them with `read -r file floor`, and
    `tests/unit/test_ci_gates.py` parses the same heredoc with a two-way
    `split()`, so the owner goes in a group comment rather than a trailing one.
    """
    lines: List[str] = []
    owner = None
    for path, floor, floor_owner in _floor_rows():
        if floor_owner != owner:
            owner = floor_owner
            lines.append(f"# {owner}")
        lines.append(f"{path} {floor}")
    return lines


def _nightly_floors() -> List[str]:
    """The enforced table: the data lines of the FLOORS heredoc.

    Only the rows are generated. The reasoning above the heredoc -- which files
    and why, which selection the percentages are measured under, what to do
    when a module moves -- is hand-written and stays there, because a list of
    percentages that a reader has to open another file to understand cannot be
    raised or migrated by anyone but its author. `tests/unit/test_ci_gates.py`
    holds that prose in place.
    """
    lines = [
        f"# Generated by {GENERATOR} from the coverage_floors entries of",
        "# tests/architecture/capabilities.toml. The comment above each group",
        "# is the capability that owns those floors: when it ports to",
        "# packages/, its floors move with it, in the same commit, by editing",
        "# the manifest and re-running the generator.",
    ]
    return lines + _floor_lines()


def _core_floors() -> List[str]:
    """The informative rendering: every floor's file, its percentage, no gate.

    No `--fail-under` anywhere in here, and that is deliberate rather than an
    omission: this job measures `tests/unit + tests/workflows` only, so the
    same floor would fail honest pull requests on a number the gate was never
    calibrated against. `tests/unit/test_ci_gates.py` asserts the absence.
    """
    lines = [
        f"# Generated by {GENERATOR} from tests/architecture/capabilities.toml.",
        "# Informative only: this job measures tests/unit + tests/workflows, a",
        "# narrower denominator than the nightly gate the floors are calibrated",
        "# against, so it reports the two numbers and gates on neither.",
        'echo "## Per-file coverage floors (informative: narrower selection than the nightly gate)" >> "$GITHUB_STEP_SUMMARY"',
        "echo '```' >> \"$GITHUB_STEP_SUMMARY\"",
        'while read -r file floor; do',
        "  case \"$file\" in ''|'#'*) continue ;; esac",
        '  measured=$(coverage report --include="$file" 2>/dev/null | tail -1 | awk \'{print $NF}\')',
        '  printf \'%-34s floor %3s%%   measured %s\\n\' "$file" "$floor" "${measured:-n/a}" \\',
        '    >> "$GITHUB_STEP_SUMMARY"',
        "done <<'FLOORS'",
    ]
    lines += _floor_lines()
    lines.append("FLOORS")
    lines.append("echo '```' >> \"$GITHUB_STEP_SUMMARY\"")
    return lines


#: Region name -> (path, the lines to put between the markers).
REGIONS: Dict[str, Tuple[Path, List[str]]] = {}


def _regions() -> Dict[str, Tuple[Path, List[str]]]:
    return {
        "filter-outputs": (CI_EXTENSIONS, _filter_outputs()),
        "filter-paths": (CI_EXTENSIONS, _filter_body()),
        "coverage-floors": (NIGHTLY, _nightly_floors()),
        "coverage-floors-informative": (CI_CORE, _core_floors()),
    }


def _replace(text: str, name: str, body: List[str]) -> str:
    """Swap the body of one generated region, preserving its indentation."""
    begin = BEGIN.format(name=name)
    end = END.format(name=name)
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == begin]
    ends = [i for i, line in enumerate(lines) if line.strip() == end]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise SystemExit(
            f"region {name!r}: expected exactly one begin and one end marker, "
            f"found {len(starts)} and {len(ends)}. The markers are\n"
            f"  {begin}\n  {end}"
        )
    indent = lines[starts[0]][: len(lines[starts[0]]) - len(lines[starts[0]].lstrip())]
    rendered = [f"{indent}{line}" if line else "" for line in body]
    return "\n".join(lines[: starts[0] + 1] + rendered + lines[ends[0] :]) + "\n"


def render(path: Path) -> str:
    """The full text one workflow file should have, given the manifest."""
    text = path.read_text(encoding="utf-8")
    for name, (owner, body) in _regions().items():
        if owner == path:
            text = _replace(text, name, body)
    return text


def files() -> List[Path]:
    seen: List[Path] = []
    for owner, _ in _regions().values():
        if owner not in seen:
            seen.append(owner)
    return seen


def drift() -> List[Path]:
    """The workflow files whose generated regions no longer match the manifest."""
    return [
        path for path in files() if path.read_text(encoding="utf-8") != render(path)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the files")
    parser.add_argument("--check", action="store_true", help="exit 1 on drift")
    args = parser.parse_args()

    problems = capabilities.problems()
    if problems:
        print("capabilities.toml is not valid, so nothing was generated:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if args.write:
        for path in files():
            rendered = render(path)
            if path.read_text(encoding="utf-8") == rendered:
                print(f"unchanged  {path.relative_to(REPO_ROOT)}")
                continue
            path.write_text(rendered, encoding="utf-8")
            print(f"written    {path.relative_to(REPO_ROOT)}")
        return 0

    stale = drift()
    for path in stale:
        print(f"STALE      {path.relative_to(REPO_ROOT)}")
    if stale:
        print(
            "\nThe generated regions disagree with tests/architecture/"
            "capabilities.toml. Edit the manifest, never the region, then run:\n"
            f"    python3 {GENERATOR} --write"
        )
        return 1
    for path in files():
        print(f"in sync    {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
