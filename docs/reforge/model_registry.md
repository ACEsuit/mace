# Foundation registry and native loaders

The registry in `mace_core.model_registry` describes 24 retained foundation
models. It is shared by PyTorch and JAX and imports neither framework. Each
record contains the canonical name, old aliases, family default, precision,
license, chemistry and theory scope, citations, model hyperparameters, committed
golden references and badge filename. Metadata uses `ModelMetadata` schema 1.2.
Registry summaries omit E0 values; the existing checkpoint schema is unchanged.

```python
from mace_core.model_registry import get_model, list_models, export_registry

entry = get_model("mace-mh-1")
model_config = entry.hyperparameters()
print(entry.metadata.citations, entry.license, entry.recommendation)
export_registry("registry.json")  # same records and labels used at runtime
```

Publication is still pending. Records have no invented artifact URL or model
DOI; paper DOIs remain citations. A canonical loader call therefore reports the
missing publication and accepts a converted local checkpoint as the alternative.
A release must fill in the native URL after publication. V1 bundles no model;
the first published load downloads a JSON/safetensors pair into the shared XDG
cache. See [artifact resolution](artifacts.md) for offline use and cache reuse.

## Names and compatibility

Every retained legacy alias resolves within its loader family and warns with
its canonical replacement. Calling a loader without a model logs its default
canonical ID. Polar still requires an explicit model. The user approved keeping
`mace-mh-0`, `mace-mh-1` and `mace-mdp` as published, without assigning a size
category; these are explicit exceptions to FM-3's size-suffix rule. MatPES uses
`medium` and OMOL `extra-large`, matching their published descriptions.

All retained entries remain recommended within their recorded scope. This label
makes no validation claim. No additional model is deprecated; the old aliases
are. ANI-CC has a tombstone pointing to OFF23 medium and has no V1 loader.
Its source checkpoint and literal CUDA default are retired with it. The three
historical ASL notices are replaced by the registry's single license notice;
MIT entries print none. Publication owners must verify the license of the
actual release artifact before publishing.

| Canonical ID | Family | Old aliases | Default | License |
| --- | --- | --- | --- | --- |
| `mace-mp-0-small` | `mace_mp` | `small` | — | MIT |
| `mace-mp-0-medium` | `mace_mp` | `medium` | — | MIT |
| `mace-mp-0-large` | `mace_mp` | `large` | — | MIT |
| `mace-mp-0b-small` | `mace_mp` | `small-0b` | — | MIT |
| `mace-mp-0b-medium` | `mace_mp` | `medium-0b` | — | MIT |
| `mace-mp-0b2-small` | `mace_mp` | `small-0b2` | — | MIT |
| `mace-mp-0b2-medium` | `mace_mp` | `medium-0b2` | — | MIT |
| `mace-mp-0b2-large` | `mace_mp` | `large-0b2` | — | MIT |
| `mace-mp-0b3-medium` | `mace_mp` | `medium-0b3` | — | MIT |
| `mace-mpa-0-medium` | `mace_mp` | `medium-mpa-0` | yes | MIT |
| `mace-omat-0-small` | `mace_mp` | `small-omat-0` | — | ASL |
| `mace-omat-0-medium` | `mace_mp` | `medium-omat-0` | — | ASL |
| `mace-matpes-pbe-0-medium` | `mace_mp` | `mace-matpes-pbe-0` | — | ASL |
| `mace-matpes-r2scan-0-medium` | `mace_mp` | `mace-matpes-r2scan-0` | — | ASL |
| `mace-mh-0` | `mace_mp` | `mh-0` | — | ASL |
| `mace-mh-1` | `mace_mp` | `mh-1` | — | ASL |
| `mace-off23-small` | `mace_off` | `small` | — | ASL |
| `mace-off23-medium` | `mace_off` | `medium` | yes | ASL |
| `mace-off23-large` | `mace_off` | `large` | — | ASL |
| `mace-omol-0-extra-large` | `mace_omol` | `extra_large` | yes | ASL |
| `mace-polar-1-small` | `mace_polar` | `polar-1-s` | — | ASL |
| `mace-polar-1-medium` | `mace_polar` | `polar-1-m` | — | ASL |
| `mace-polar-1-large` | `mace_polar` | `polar-1-l` | — | ASL |
| `mace-mdp` | `mace_mdp` | `default` | yes | ASL |

This table is generated from `list_models()`; the JSON export also includes
recommendation scope, citations and hyperparameters.

## PyTorch loaders

```python
from mace_torch.calculators.foundations import mace_mp, load_foundation

calc = mace_mp("/models/converted.json", device="cpu")
loaded = load_foundation("/models/converted.json", device="cpu")
raw = mace_mp("/models/converted.json", return_raw_model=True)
```

`mace_mp`, `mace_off`, `mace_omol`, `mace_mdp` and `mace_polar` return the native
ASE calculator. `return_raw_model=True` returns the native PyTorch module;
`load_foundation` provides the structured model, configuration and metadata for
extension. MP and Polar retain `float32` defaults; OFF, OMOL and MDP retain
`float64`. `device=""` chooses CUDA when available, otherwise CPU.

Pass `precision=PrecisionConfig(...)` to select an operator policy. Both the
calculator and the raw module read checkpoint tensors directly in the selected
precisions. `load_foundation` also records that policy in the resolved metadata.

OMOL requires its `omol` head. Multi-head MH retains all heads and requires the
caller to choose one explicitly. MDP warns that it predicts dipoles and
polarizability without energy or forces and rejects a different `model_type`.
These compatibility tokens are checked at the loader boundary; the checkpoint
configuration determines the architecture. Legacy pickle paths are rejected
with instructions to run `mace model convert`.

MP retains `dispersion`, `dispersion_damping`, `dispersion_xc`,
`dispersion_cutoff` and `dispersion_coord_cutoff`, including the released
`damping` alias, which takes precedence. D3 defaults remain `bj`, `pbe` and
40 Bohr for both cutoffs; explicit cutoff arguments are in Angstrom.

Readers require the current metadata schema. Earlier V1 schemas are not migrated.
A version mismatch reports both the stored and required versions. Cached JSON
and tensor files may have different URL-derived stems; both readers receive
the resolved pair explicitly.

## Extending a loaded model

```python
from mace_core.observables import ObservableSpec
from mace_torch.finetune.extend import NewSpeciesInit, extend_elements
from mace_torch.finetune.readouts import extend_readouts

# Supply each head's isolated-atom energy for each added element, in eV.
extend_elements(loaded, "with-silicon", {head: {14: -100.0}
                for head in loaded.heads}, NewSpeciesInit(seed=8))
extend_readouts(loaded, [ObservableSpec(name="new_scalar", irreps="0e",
                per_atom=False, units="eV")], "with-readout", seed=21)
```

Adding species uses FT-3's existing fresh/donor initialization and raw-weight
normalization. Existing predictions are preserved to arithmetic roundoff.
Adding readouts uses the ordinary model initializer, keeps every trained
canonical tensor, and rejects declarations that change an existing operator's
layout or shape. New readouts need training. Both paths preserve the parent
record and checkpoint precision. Descendants have no registry badge.

## Support matrix

The existing nightly `foundations` job emits one JSON file per registry entry,
with independent framework rows. PyTorch covers all 24 models; JAX covers
MP-0 small, MPA-0 medium and OFF23 medium. Each row records code/version,
hardware, reference hashes and expected values, selected precision/tolerance,
measured deviations and failure details. The job exports registry labels from
the same runtime records and uploads the matrix with them. Its cache key
includes the registry file. The schedule remains dormant on V1 until the
workflow is integrated into the default branch or explicitly dispatched.

A published artifact obtains support only after architecture construction,
committed-golden comparison and its matrix row succeed. A missing publication
is `pending_publication`; a local candidate can only be `validated_local`.
A numerical mismatch is `failed` and fails the validation command. Polar's
agreed long-range stress correction remains explicitly excluded from direct
legacy stress comparison. The matrix tests the family default precision as
well as fp64; a failure at either precision prevents a supported row.
The PyTorch default check calls the public loader with its dtype argument
omitted and reads the resulting engine's interface dtype. A failed check
records the precision it expected, the precision it observed when available,
and the hardware used. Earlier successful checks stay in the same row.

To reproduce local candidate validation without publishing:

```bash
python -m tests.foundations.registry_badges --framework torch \
  --artifacts candidate-map.json --output badges
python -m tests.foundations.registry_badges --framework jax \
  --artifacts candidate-map.json --output badges
```

The candidate map assigns canonical IDs to local native checkpoint paths. For
published validation, omit it and set `MACE_CI_ALLOW_NETWORK=1`. The
[FM-3 evidence](gate3/2026-11-02/fm3.md) records current results and remaining
publication and numerical restrictions.
