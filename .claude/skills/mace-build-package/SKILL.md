---
name: mace-build-package
description: Build or test an independent distribution in the native MACE workspace.
---

Read the package's CLAUDE.md. Install mace-core before framework packages;
the root is a development workspace, not an installable distribution.
Build with `python -m build packages/<distribution>`. Test the wheel in a fresh
environment so an editable checkout cannot hide missing files. Check CLI
entry-point ownership and absence of model artifacts in wheels. Use ruff and
ty at the package path. A local build does not authorize tags or publishing.
