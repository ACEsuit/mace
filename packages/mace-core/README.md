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
  data source, per head the sources it consumed, DOI, citations and notes,
  and, for a trained model, how the training run ended), with a JSON round
  trip, `ConfigRecord.from_config()` to embed a config in its fixed-point
  form, and `format_citations()`. The schema is versioned `major.minor`;
  which records a release reads is stated at `SCHEMA_VERSION`.
- `mace_core.neighbors` — `get_neighborhood`, the pure-function neighbour list
  every graph builder calls, over swappable backends registered in the
  `mace.neighbor_backends` entry point group (`matscipy` by default, and
  `brute_force`), selected by the `NeighborListConfig` section. The search box
  and the effective cell the graph carries are decided once, by
  `search_geometry`, whichever backend runs.
- `mace_core.graph` — the flat graph dict a model is handed: `GRAPH_SCHEMA`
  as data, `validate_graph` for the boundaries, the numpy reference `collate`,
  `graph_sample` from a `Configuration`, and the padding policies and budgets.
