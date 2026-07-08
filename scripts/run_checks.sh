#!/usr/bin/env bash
# Local convenience wrapper: format, then run the same core checks CI runs.
# CI test jobs go through the .github/actions/run-tests composite action;
# the commands below mirror its defaults for the PR-gating suites.
set -euo pipefail
cd "$(dirname "$0")/.."

# Format in place (CI checks formatting via pre-commit but never rewrites).
# tests/ is excluded from formatting/lint by repo convention.
python -m black mace scripts
python -m isort mace scripts

pre-commit run --all-files --show-diff-on-failure
python -m pytest tests/unit -m "not slow" -n auto --timeout=600
python -m pytest tests/workflows -n 2 --timeout=600
