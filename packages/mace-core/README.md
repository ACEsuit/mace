# mace-core

Framework-agnostic contract and pure math for MACE v1: types, config,
observables, the kernel-backend protocol, Clebsch-Gordan, neighbours and the
data spec. It imports no torch, no jax and no e3nn, so both implementation
packages can depend on it without depending on each other.

Distribution `mace-core`, import name `mace_core`.

What is here so far:

- `mace_core.config` — `ReforgeBaseConfig`, the pydantic base every v1
  config schema derives from: `load()` reads one TOML/YAML/JSON file and
  validates it once (`from_dict()` for a caller that edits the parsed dict
  first); unknown keys are pydantic errors at every level;
  `to_resolved_dict()` is the fully defaulted, round-trippable export and
  `to_user_dict()` holds only what was set. The base has no command-line
  knowledge.
- `mace_core.cli`: the `mace` program. Its commands come from the packages
  that implement them, through the `mace.commands` entry point group, so
  torch and jax plug in and this package declares none. A command with a
  schema reads one file with `--config`, writes its few explicit flags into
  it, and validates once; there is no general override syntax.
- `mace_core.metadata` — `ModelMetadata`, the versioned record every trained
  model carries (config as written and as resolved, provenance, a summary per
  data source, per head its E0s and the sources it consumed, DOI, citations,
  notes, and the records of the models it was built from), with a JSON round
  trip,
  `ConfigRecord.from_config()` to embed a config in its fixed-point form, and
  `format_citations()`.
