# Published foundation migration scope

The roster contains 25 entries: 24 retained models and ANI-CC retired. Names
below identify conversion candidates, not published native downloads. Every
retained row still needs its native artifact URL and artifact DOI before release.
The registry refuses unpublished downloads; use a validated local native pair.
Old aliases remain accepted with a canonical-name warning. No additional
model is declared obsolete. Preserve each model's published domain and license.

| Loader | Legacy name | V1 name | Default dtype | Status |
| --- | --- | --- | --- | --- |
| `mace_mp` | `small` | `mace-mp-0-small` | float32 | Native publication pending. |
| `mace_mp` | `medium` | `mace-mp-0-medium` | float32 | Native publication pending. |
| `mace_mp` | `large` | `mace-mp-0-large` | float32 | Native publication pending. |
| `mace_mp` | `small-0b` | `mace-mp-0b-small` | float32 | Native publication pending. |
| `mace_mp` | `medium-0b` | `mace-mp-0b-medium` | float32 | Native publication pending. |
| `mace_mp` | `small-0b2` | `mace-mp-0b2-small` | float32 | Native publication pending. |
| `mace_mp` | `medium-0b2` | `mace-mp-0b2-medium` | float32 | Native publication pending. |
| `mace_mp` | `large-0b2` | `mace-mp-0b2-large` | float32 | Native publication pending. |
| `mace_mp` | `medium-0b3` | `mace-mp-0b3-medium` | float32 | Native publication pending. |
| `mace_mp` | `medium-mpa-0` | `mace-mpa-0-medium` | float32 | Native publication pending. |
| `mace_mp` | `small-omat-0` | `mace-omat-0-small` | float32 | Native publication pending. |
| `mace_mp` | `medium-omat-0` | `mace-omat-0-medium` | float32 | Native publication pending. |
| `mace_mp` | `mace-matpes-pbe-0` | `mace-matpes-pbe-0-medium` | float32 | Native publication pending. |
| `mace_mp` | `mace-matpes-r2scan-0` | `mace-matpes-r2scan-0-medium` | float32 | Native publication pending. |
| `mace_mp` | `mh-0` | `mace-mh-0` | float32 | Native publication pending. |
| `mace_mp` | `mh-1` | `mace-mh-1` | float32 | Native publication pending. |
| `mace_off` | `small` | `mace-off23-small` | float64 | Native publication pending. |
| `mace_off` | `medium` | `mace-off23-medium` | float64 | Native publication pending. |
| `mace_off` | `large` | `mace-off23-large` | float64 | Native publication pending. |
| `mace_omol` | `extra_large` | `mace-omol-0-extra-large` | float64 | Native publication pending. |
| `mace_polar` | `polar-1-s` | `mace-polar-1-small` | float32 | Native publication pending. |
| `mace_polar` | `polar-1-m` | `mace-polar-1-medium` | float32 | Native publication pending. Float32 parity open; float64 evidence does not close it. |
| `mace_polar` | `polar-1-l` | `mace-polar-1-large` | float32 | Native publication pending. Float32 parity open; float64 evidence does not close it. |
| `mace_mdp` | `default` | `mace-mdp` | float64 | Native publication pending. |
| `mace_anicc` | `default` | `MACE-OFF23` | — | Retired: ANI-CC is superseded by OFF23; no ANI conversion. |

MH and MDP retain their published names without an invented size suffix.
MatPES uses `medium`; OMOL uses `extra-large`. Short names such as `medium`
are family-scoped. See the registry for supported chemistry, references and
licenses; a familiar alias does not establish suitability for a new system.

The default precision is unchanged. In particular Polar medium/large remain
float32 by default with unresolved float32 parity; migration must report this
rather than silently switch to float64. Float64 checks are separate evidence.

Conversion checks compare functions and gradients in the canonical reduced
basis. Save/reload of the V1 checkpoint preserves its weights exactly. Reducing
a redundant source basis is not invertible; retain the neutral original.

Arbitrary user-trained v0.3 checkpoints have no supported automated conversion.
Keep the pinned v0.3 environment or retrain in V1. The generic importer can read
additional architectures for parity research, but that is not a user migration
support promise. `MagneticMACE` is a non-executable legacy base; executable
magnetic variants have separate FM-5 evidence.

V1 ships no model artifact. Legacy bundled pickles were available in source or
editable installations, not in the legacy wheel. Downloads use the XDG cache
(`XDG_CACHE_HOME`, then the platform default) independently of `work_dir`.
