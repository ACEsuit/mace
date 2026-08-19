# The published foundation models

Targets: `foundations` (tracked artifact, part of `all`) and
`foundations-network` (downloaded artifacts, not part of `all`).

The anchors pin the *architecture*; these pin the *artifacts users actually
load*. Three published checkpoints, in two tiers, described once in
`foundation_artifacts.py` — the loader call, the fixture selection and the
sha256 — and read from there by both `regenerate.py` and
`test_foundation_goldens.py`, so a reference cannot have been generated with
different arguments than the test replays.

| reference | artifact | loader | fixtures | tier |
|---|---|---|---|---|
| `mpa0_medium_e3nn_cpu_fp64.json` | `mace-mpa-0-medium.model`, tracked in this repo | `mace_mp(default_dtype="float64", device="cpu")` | all six | every PR, no marker |
| `mp_small_e3nn_cpu_fp64.json` | `2023-12-10-mace-128-L0_energy_epoch-249.model` | `mace_mp(model="small", …)` | all six | nightly, `network` |
| `off_small_e3nn_cpu_fp64.json` | `MACE-OFF23_small.model` | `mace_off(model="small", …)` | molecular only | nightly, `network` |

MPA-0 medium is the one published artifact that can be pinned per pull
request: `download_mace_mp_checkpoint` short-circuits to the tracked file for
`model=None`, so an unqualified `mace_mp()` is both the highest-traffic model
in the project and download-free. MACE-OFF is organic chemistry, so it is
evaluated on the `molecular` tag rather than on slabs.

**Identity is measured, not assumed.** An alias is a name and names get
re-pointed upstream, so each reference records the sha256 of the checkpoint it
was generated from, and the test digests *the file the loader actually
opened* — observed by watching `torch.load` during the loader call, rather
than by rebuilding the cache path here, which would be a second copy of
`mace_off`'s naming rule. A re-uploaded artifact then fails with "this is not
the artifact this golden pins" instead of looking like a physics regression.
For the network tier a further test asserts the registry's URL is still the
one in `foundations_models.py`. The tracked MPA-0 file is byte-identical to
the release artifact at `mace_mp_urls["medium-mpa-0"]`, so the digest
identifies the model by either route.

**Two traps, both silent, both now pinned by a test.**

* `mace_mp` defaults to **float32** and to CUDA-if-present, so every loader
  call in the registry states `default_dtype="float64"` and `device="cpu"`,
  and a test asserts both the registry entry and the loaded weights.
* the tracked checkpoints are **in git but not in the wheel**: `setup.cfg`
  declares no `package_data` and `MANIFEST.in` carries only `py.typed`, so the
  published wheel is ~300 KB and holds `foundations_models.py` without the
  directory of the same name. CI installs the wheel, so in the very job the
  per-PR golden runs in, `local_model_path` points at nothing and
  `mace_mp()` would *download*. `tracked_checkpoint_in_place()` points the
  package at the copy in the checkout (same digest), and the tracked tier
  additionally runs inside `no_network()`, where any download attempt raises —
  so "needs no network" is a gate rather than a claim about a warm cache.

`mace_anicc` looks like a fourth no-download golden and is not: its tracked
`ani500k_large_CC.model` was serialised with CUDA TorchScript archives, and
e3nn's `CodeGenMixin.__setstate__` calls `torch.jit.load(buffer)` with no
`map_location`, so it cannot be loaded on a host without CUDA at all — not
even with `map_location="cpu"`. That refusal is pinned as a measurement in
`test_foundation_goldens.py`; if it ever starts failing, the artifact was
re-exported and the golden becomes possible.

## Regenerating

`foundations` is part of `--target all`. `foundations-network` is not, because
running it downloads the published releases; name it explicitly when you mean
to refresh a downloaded artifact's reference.

## Running

The tracked tier carries no marker and runs in the ci-core `unit` job with
everything else. The `network` pair runs in the nightly `foundations` job,
which passes `allow-network: "true"` and `require-caps: network,polar`: there
a skip is a failure, so a dead download host or a replaced upstream artifact
turns the job red instead of quietly shrinking the golden set.

```bash
MACE_CI_ALLOW_NETWORK=1 python -m pytest tests/golden -v   # includes the downloads
```
