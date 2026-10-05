# mace-core

Framework-agnostic contract and pure math for MACE v1: types, config,
observables, the kernel-backend protocol, Clebsch-Gordan, neighbours and the
data spec. It imports no torch, no jax and no e3nn, so both implementation
packages can depend on it without depending on each other.

Distribution `mace-core`, import name `mace_core`.

What is here so far:

- `mace_core.config` — `BaseConfig`, the pydantic base every v1 config schema
  derives from: one TOML/YAML/JSON file, validated once; unknown keys are
  errors, a validated config is frozen, and its resolved export loads back to
  the same config. `config.cli` holds the `--a.b value` override grammar for a
  command line to compose with it.
- `mace_core.metadata` — `ModelMetadata`, the versioned record every trained
  model carries (config as written and as resolved, provenance, a summary per
  data source, per head the sources it consumed, DOI, citations and notes),
  with a JSON round trip, `ConfigRecord.from_config()` to embed a config in
  its fixed-point form, and `format_citations()`.
