# mace-core

Framework-agnostic contract and pure math for MACE v1: types, config,
observables, the kernel-backend protocol, Clebsch-Gordan, neighbours and the
data spec. It imports no torch, no jax and no e3nn, so both implementation
packages can depend on it without depending on each other.

Distribution `mace-core`, import name `mace_core`.

What is here so far:

- `mace_core.config` — `ReforgeBaseConfig`, the pydantic base every v1
  config schema derives from: one TOML/YAML/JSON file plus dotted CLI
  overrides (`--model.num_interactions 3`),
  precedence defaults < file < CLI,
  unknown keys rejected with the nearest valid neighbour named, and
  `to_resolved_dict()` for the fully defaulted, round-trippable export.
- `mace_core.metadata` — `ModelMetadata`, the versioned record every trained
  model carries (config as written and as resolved, provenance, a summary per
  data source, E0 details per head, DOI, citations, notes, and the records of
  the models it was fine-tuned or distilled from), with a JSON round trip,
  `ConfigRecord.from_config()` to embed a config in its fixed-point form, and
  `format_citations()`.
