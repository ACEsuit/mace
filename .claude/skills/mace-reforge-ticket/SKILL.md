---
name: mace-reforge-ticket
description: Take a MACE Reforge board ticket from start to a reviewable PR - session ritual, branch naming, scope discipline, the pre-PR verification commands for both toolchains, the feature-inventory gate, deletion-only RET tickets, and the review gate. Use when starting, implementing, or wrapping up work on a reforge ticket.
---

# mace-reforge-ticket

Procedure for one reforge ticket. The rules it depends on are in
`mace-reforge`; read that first.

## Session ritual

1. Read the ticket on the board (<https://github.com/orgs/ACEsuit/projects/2>).
2. Read the files the ticket names. The ticket ID resolves to a GitHub issue in
   `ACEsuit/mace` - work from the issue, since that is where discussion lands.
3. **Restate the acceptance criteria in your own words before writing any code.**
   If restating them exposes an ambiguity, resolve it on the ticket, not in the
   implementation.
4. Check that the ticket's dependencies are merged. If they are not, say so and
   stop - a ticket built on an unmerged dependency produces a PR nobody can review.

Where a new file goes is answered by `docs/reforge/target_layout.md`, and which
gate the ticket sits behind by `docs/reforge/plan.md` §4 - consult them rather
than inferring a location from the surrounding code.

Never skip step 3. It is the step that makes two people implementing the same
ticket produce comparable work.

## One ticket, one PR

```
branch: reforge/<TICKET-ID>-<slug>      e.g. reforge/inf-3-import-guard
base:   mace-reforge                   (the integration branch, cut from develop)
```

Ticket PRs target **`mace-reforge` on `ACEsuit/mace`**, not `develop` - the
rewrite integrates on that branch and reaches `develop` as its own reviewed step.
Push the work branch wherever you normally push (your fork or upstream); the PR
base is what matters.

Before branching, make sure your `mace-reforge` is current: `git pull upstream
mace-reforge`. Branching from a local copy that is weeks behind produces a PR
whose diff contains other people's already-merged work.

- **The out-of-scope section is binding.** If the ticket turns out bigger than
  its `Size` label, propose a split on the ticket. Do not silently expand scope,
  and do not fold in an unrelated fix you noticed along the way.
- **A tolerance change is never part of a feature PR.** Same for regenerating a
  golden reference, and for reformatting anything under `mace/`.
- **`RET-*` tickets are deletion-only.** Zero new v1 logic: the PR removes a
  piece of `mace/` and degrades its parity test to a frozen golden, once the v1
  counterpart has full parity. Nothing else changes in the same PR.
- **If the ticket implements v0.3 functionality, flip its rows in the feature
  inventory in the same PR** - `tests/golden/feature_inventory.md`, gated by
  `tests/golden/check_inventory.py`. Phase gates audit that file, so a PR that
  implements a capability and leaves its row `todo` fails the gate later, far
  from the cause.

## Before opening the PR

Both toolchains, then the tests that can catch a behaviour change:

```bash
pre-commit run --all-files                                  # legacy toolchain, mace/**
prek run --all-files                                        # new toolchain, packages/**
python -m pytest packages/*/tests tests/architecture         # fast
python -m pytest tests/golden -m "not slow"                  # the committed numbers
python -m pytest tests/parity -m "not slow"                  # the live oracle
python -m pytest tests/unit -m "not slow" -n auto            # legacy suite still green
```

The new toolchain expands to `ruff check packages/ && ruff format --check
packages/ && ty check packages/` if you would rather run it directly. Skip the
suites whose directories do not exist yet on your branch, and say which ones you
skipped rather than reporting a clean run.

Two traps in the legacy toolchain, both of which read as a false pass locally:

- pylint's cyclic-import check (`R0401`) is hidden by pre-commit's concurrency.
  Reproduce what CI sees with `PRE_COMMIT_NO_CONCURRENCY=1 pre-commit run --all-files`.
- `tests/` is excluded from formatting and lint by both pre-commit and CI, so
  test files are not auto-formatted and a lint pass says nothing about them.

**A GPU-marked ticket needs a GPU.** Either run the `-m gpu` selection on a GPU
host, or state in the PR description that you are relying on the GPU CI
workflow. Do not report a GPU ticket as verified from a CPU-only run.

## Reproducing a CI job locally

CI has no shell scripts: every test job runs through the
`.github/actions/run-tests` composite action, and its inputs (`tests`,
`markers`, `require-caps`, `allow-network`, `splits`, `coverage`) map 1:1 to
pytest flags. A job's `with:` block **is** its local reproduction recipe - read
it instead of guessing the command.

## Review

- Every PR is reviewed before merge. Numerics and physics changes need a
  reviewer with physics expertise; infra, CI and packaging changes need one
  comfortable with the tooling.
- When the ticket carries a **Review focus**, that focus is the merge gate.
  Address it explicitly in the PR description.
- Write the PR description so it can be reviewed without the ticket open:
  what behaviour is pinned, what was verified, what was deliberately left out.
