# Artifact downloads and the V1 cache

`mace_core.artifacts.resolve_artifact` accepts an existing local path, an HTTPS
URL or an `ArtifactRef`. It returns a local file path and imports no tensor
framework. Replay data and JAX checkpoints use this resolver; FM-3's foundation
loaders must use it too.

Downloads live under `$XDG_CACHE_HOME/mace`, or `~/.cache/mace` when that variable
is unset or empty. Training results belong to `--work_dir`. V1 rejects explicit
`--downloads_dir` values and names these two replacements in its error message.

Each URL has a cache name consisting of its normalized URL digest and filename.
GitHub `blob` and `raw` links share the raw-content URL's cache entry; release
asset links pass through. Before downloading, the resolver also checks both
historical names: the basename stripped to alphanumeric characters and
underscores, and the basename with its query removed. An adopted file stays
where it is so legacy can still find it. A cache hit performs no network request.

Each transfer writes to its own temporary `.part` sibling and replaces the
destination only after the complete response arrives. HTML responses and a
body whose size disagrees with `Content-Length` are rejected. Read errors,
timeouts and Python interruptions remove that transfer's temporary file;
an existing destination remains intact. Concurrent cold requests may transfer
the same bytes twice, but cannot modify one another's temporary file or expose
a partially written destination. Subsequent requests reuse the completed file.
The socket timeout remains 120 seconds per read.

The resolver handles bytes, without loading or converting models. A cached
legacy pickle can be reused as the input to `mace model convert`; it cannot
serve as a native checkpoint. A converted artifact at a new URL requires its
own download. The model readers validate the safetensors/JSON format and
metadata. Caching does not replace those checks or authenticate an artifact.

## Bundled files removed from V1

V1's runtime packages and wheels contain no foundation checkpoints or standalone
foundation E0 tables. The tiny committed checkpoints under JAX's test fixtures
are test inputs and are excluded from the wheels.

| Legacy file or behavior | V1 replacement |
| --- | --- |
| `mace_mp()` uses bundled `mace-mpa-0-medium.model` in a source checkout | Resolve the native artifact through the shared cache. The first load on a fresh checkout downloads it once. |
| `2023-12-03-mace-mp.model`, with no loader consumer | Removed from V1 runtime packaging. |
| `mp_vasp_e0.json`, with no consumer | Checkpoint E0 tensors provide model-specific values. |
| `ani500k_large_CC.model` and `mace_anicc` | Retired; use MACE-OFF23. |

These files were absent from the legacy wheel, so the bundled-model shortcut
only existed in source/editable installs. The frozen legacy tree retains its
files until its retirement tickets execute. For offline V1 use, populate the
cache first or pass local converted checkpoints. This page supplies the FM-4
migration notes for REL-1 and the future FM-3 registry; publication of the
converted foundation roster remains separate work.
