# One page per family of goldens

A family that adds goldens adds a page here, and does not edit
`tests/golden/README.md`. That parent file describes what is true of every
family — the harness rule, the no-dropped-keys rule, the three surfaces, the
regeneration lock. This directory holds what is true of one.

Name the page after the target module it belongs to, so `targets/magnetic.py`
is documented by `docs/magnetic.md`, and answer four questions in it:

* **what the goldens pin** — which model class, which checkpoint, which
  channels, and which fixtures;
* **why those and not others** — a fixture set chosen by chemistry, a size
  chosen because the larger ones share the architecture, an anchor chosen
  because the CLI cannot emit it;
* **what regenerating requires** — a download, an optional package at a
  pinned commit, a GPU. If the target is not part of `--target all`, the
  reason belongs here;
* **how to run its tests** — the selection line, including any capability
  marker, and why that marker is unavoidable.

Keeping this out of the parent file is what lets two families land
independently. Both would otherwise append to the same section, and every
pair of them would conflict.

If one family is split across two targets because they differ in what running
them requires — a tier that downloads and a tier that does not — give both
modules the same `DOC` and write one page. Two pages would split one story.
