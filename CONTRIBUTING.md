# Contributing to MACE during the v1 rewrite

MACE is being rewritten toward v1.0. The rewrite happens in the open on the
`mace-reforge` branch, while v0.3.x stays the line users run in production.

This document covers that overlap: what we can accept into v0.3 while the
rewrite is under way, and where each kind of change belongs. It is deliberately
narrow and temporary. It will be folded into the project's permanent
contribution guide once the rewrite lands.

**The rule behind every rule here:** what we can accept into v0.3 is decided by
whether we can carry it into v1. Anything we cannot carry over becomes work that
gets deleted, so the answer has to be no at the start rather than at the end.

## Branches

Three, and no others:

- **`main`** carries releases. A release is cut by merging `develop` into `main`
  and releasing from there.
- **`develop`** is where v0.3 is developed, and where v1 lands when it is ready.
- **`mace-reforge`** is the rewrite.

No separate maintenance branch is needed: `main` already holds the released
state, and stays on v0.3 until v1 is released. So v0.3 fixes go to `develop`,
and reach users at the next cut.

When the rewrite is finished, whatever is new on `develop` is migrated into v1,
then `mace-reforge` merges back into `develop`, and `develop` into `main` for the
v1 release.

## Where does my change go?

| You want to | Send it to | Notes |
|---|---|---|
| Fix a bug in v0.3 | `develop` | Always accepted |
| Fix a bug in v1 | `mace-reforge` | Always accepted |
| Add a new feature | **v1, if that is possible** | Otherwise `develop`, under two conditions |
| Change core behaviour, or touch shared files | **v1 only** | Not accepted against v0.3 |
| Work a rewrite ticket | `mace-reforge` | Branch `reforge/<TICKET-ID>-<slug>` |

### Bug fixes are always allowed

Both lines take fixes, and a fix may touch a shared file. That is the
difference between fixing behaviour and adding it. Fix the bug where it lives.

### New features: v1 first

**If you can build the feature in v1, build it there.** A feature added to v0.3
has to be ported: someone reimplements it against the new architecture and proves
it still computes the same numbers. Building it once, in the architecture it will
live in, skips that entirely.

That is not always available, and the honest reason is timing: the v1 stack is
still being scaffolded, so the surface your feature needs may not exist yet. You
may also need the feature in a released version before v1 ships. When v1 genuinely
is not an option, v0.3 accepts the feature if it meets both conditions below.
Say in the PR description why v1 was not possible, so we know what to carry over
and why it exists in two places.

**1. It is self-contained in its own directory.** Its code lives under its own
module directory, its tests under their own test directory, and an optional
dependency (if it needs one) behind its own extra. The test is blast radius, not
line count: adding a new thing beside the existing ones is fine, changing what
the existing ones do is not. Additive wiring is normally fine: one entry in a
registry, one extra in `setup.cfg`. If you are unsure whether your change crosses
the line, say so in the PR description rather than guessing, and we will decide it
there.

**2. It is tested on numbers, not on absence of errors.** This is the condition
that actually matters, and it is worth being blunt about why.

The rewrite reproduces *behaviour*, and it proves it by comparing against
committed reference numbers. So a test that asserts your feature runs without
raising tells the rewrite nothing. There is nothing to reproduce, and nothing
that would notice if the port changed an answer. A test that pins **what your
feature computes**, on a fixed input, becomes the contract we carry it over
against.

Concretely, that means tests which:

- assert values, meaning energies, forces, whatever your feature produces, not
  just that a call returned;
- run on a committed, fixed input rather than random data, with seeds fixed
  where randomness is unavoidable;
- are deterministic: no wall-clock, no network, no dependence on host thread
  count for a tight tolerance;
- carry a capability marker if they need an optional dependency or a GPU, so
  they skip cleanly where it is absent;
- live in their own directory under `tests/`.

**A feature whose numbers are pinned is a feature we can guarantee we carry
over. A feature without them is not, and we would rather say so now.**

### Core and shared changes go straight to v1

Changes to core behaviour, or to files that everything else depends on, are not
accepted against v0.3. Implement them in v1 instead. In the current tree the
shared surface is, in practice: the model and block definitions
(`mace/modules/models.py`, `blocks.py`, `symmetric_contraction.py`), the
differentiable glue (`mace/modules/utils.py`), the backend dispatch
(`mace/modules/wrapper_ops.py`), the data layer (`mace/data/atomic_data.py`),
the argument parser (`mace/tools/arg_parser.py`), the training loop
(`mace/tools/train.py`), and the ASE calculator (`mace/calculators/mace.py`).

The reason is not that those files are precious. It is that a change there
cannot be carried across as a unit: it has to be re-decided against the new
architecture, so doing it twice is the only outcome.

## What the rewrite changes for you

Four things worth knowing before you plan work:

- **`mace/` is frozen on `mace-reforge`, not on `develop`.** On the rewrite
  branch the existing package is not edited at all: it is the reference the new
  implementation is measured against, the source of truth for what the numbers
  must be. On `develop` it keeps taking ordinary changes, which is what makes the
  routing above possible.
- **Your v0.3 work is not stranded.** When the rewrite finishes, whatever is new
  on `develop` is migrated into v1 before the branches merge. That migration is
  the reason for the two conditions above: a self-contained feature with pinned
  numbers can be carried over and verified, and one without them cannot.
- **A fix reaches users at the next release cut**, when `develop` is merged into
  `main`. If you need a v0.3 fix sooner than that, or after `develop` has moved
  on to v1, say so on the issue rather than assuming either way.
- **v1 will not preserve code compatibility, and checkpoints migrate through an
  explicit converter** rather than by loading directly. Functionality is
  preserved; imports and file layout are not.

## Toolchains

Two toolchains, split by path, and they never mix on one file:

- `mace/**` uses `black`, `isort`, `pylint` and `mypy`, via `pre-commit`. This
  tree is not reformatted, even where a diff would look cleaner.
- `packages/**`, the v1 stack, uses `ruff` for lint and format plus `ty`, via
  `prek`.

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```

Be aware of what is actually enforced today: the `lint` job in
`.github/workflows/ci-core.yaml` is a single `pre-commit run --all-files`, and
the hooks exclude `tests/`, `.github/`, `docs/`, `README.md` and `LICENSE.md`.
`mypy` is configured in `.mypy.ini` and `ruff.toml` selects one rule, but
neither is invoked by any hook or job yet.

## Tests

`tests/` is organised by execution requirement, and capability markers are
enforced. A test whose optional dependency is missing skips locally, and fails
in the CI job that guarantees that dependency.

```bash
python -m pytest tests/unit -m "not slow" -n auto    # the fast loop
python -m pytest tests/workflows -n 2                 # end-to-end CLI trainings
```

CI has no shell scripts: every test job runs through the
`.github/actions/run-tests` composite action, whose inputs map 1:1 to pytest
flags. A job's `with:` block is its local reproduction recipe.

Committed reference numbers live in `tests/golden/` and are edit-locked. If your
change makes one fail, the first assumption is that the change is wrong.
Regenerating a reference, or widening a tolerance, is a separate PR with a
physics justification. It is never part of a feature.

## Where the rewrite's decisions live

- **[The ticket board](https://github.com/orgs/ACEsuit/projects/2)** is the
  backlog, one issue per ticket. Ticket bodies are self-contained.
- **[The execution plan](https://github.com/ACEsuit/mace/blob/mace-reforge/docs/reforge/plan.md)**
  covers the transition strategy, the phases and their exit gates, and a ticket
  index.
- **[The target layout](https://github.com/ACEsuit/mace/blob/mace-reforge/docs/reforge/target_layout.md)**
  says where each piece ends up in the new tree.

Both live under `docs/reforge/` on the `mace-reforge` branch, and reach this
branch when the rewrite merges.

If a ticket does not determine what to build, raise it on the ticket rather than
improvising a design.
