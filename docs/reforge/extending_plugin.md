# Extending MACE from outside — a third-party plugin (entry points)

This is the companion to [Extending MACE](extending_mace.md). That doc built the magnetic feature as
an **in-tree extra** — code under `mace_torch/extras/magnetic/`, shipped with MACE. Here we ship the
*same* feature as a **standalone, `pip`-installable package** (`mace-magnetic`) that plugs into MACE
**without touching the MACE repo at all**. The only mechanism is **entry points**.

Same registries, same `config.toml`, same trained-weight format. The only differences are where the
code lives and that the plugin releases on its own cadence. When to choose a plugin over an in-tree
extra or built-in is the last section.

---

## 1. The entry-point groups MACE exposes

MACE scans three entry-point groups at startup. A feature plugin like magnetic uses the first; the
other two are for swapping *compute kernels*, not adding features.

| Group | An entry is… | Contract | Selected |
|---|---|---|---|
| `mace.plugins` | a **module** imported at startup | its `@register_*` decorators populate the in-process registries (models, transforms, embeddings, losses, observable rows) | active once installed; referenced from `config.toml` by the names it registers |
| `mace.kernel_backends.torch` / `.jax` | a **`KernelBackend`** implementation | Protocol: the equivariant tensor-product / symmetric-contraction ops + the canonical weight layout (RFC-01 / BKD-1) | by name in config/build |
| `mace.electrostatics_backends` | a **long-range / electrostatics solver** | Protocol: k-space op + optional SCF span (RFC-09); a separate registry from the kernel backends | by name; bit-parity swaps are free |

The distinction: `mace.plugins` entries are **imported for their side effects** (their decorators run
and register things, and they stay active). The two backend groups are **named implementations you
pick one of** — they don't turn on just by being installed.

---

## 2. The package skeleton

A plugin is an ordinary Python distribution that depends on MACE and declares one entry point:

```text
mace-magnetic/                       # a separate repo / PyPI package — not in the MACE tree
├── pyproject.toml
├── src/mace_magnetic/
│   ├── __init__.py                  # the mace.plugins target: runs the @register_* decorators
│   ├── embedding.py                 # MagmomEmbedding                              (extending_mace.md §1)
│   ├── observables.yaml             # the magforces row                            (§2)
│   ├── transforms.py                # RotateMagmom                                 (§3)
│   └── model.py                     # MagneticScaleShiftMACE / MagneticSCFMACE     (§5)
└── tests/
```

```toml
# mace-magnetic/pyproject.toml
[project]
name = "mace-magnetic"
dependencies = ["mace-torch>=1.0", "sphericart-torch==1.0.9"]   # depends on MACE, never the reverse

[project.entry-points."mace.plugins"]           # the whole hook: one line
magnetic = "mace_magnetic"                       # MACE imports this module on discovery
```

That entry point is the *entire* integration surface. There is no MACE-side registration list, no
edit to any MACE file.

---

## 3. Registration — identical to the in-tree extra

The module code is what [Extending MACE](extending_mace.md) §1, §3 and §5 already showed; **only the
import root changes** (`mace_magnetic` instead of `mace_torch.extras.magnetic`). The `__init__.py` runs the
decorators and loads the observable rows:

```python
# src/mace_magnetic/__init__.py
from mace_torch.models import register_model
from mace_torch.data import register_transform
from mace_torch.nn import register_input_embedding
from mace_torch.observables import register_observables_yaml

from .embedding import MagmomEmbedding
from .transforms import RotateMagmom
from .model import MagneticScaleShiftMACE, MagneticSCFMACE

register_input_embedding("magmom")(MagmomEmbedding)
register_transform("rotate_magmom")(RotateMagmom)
register_model("MagneticScaleShiftMACE")(MagneticScaleShiftMACE)
register_model("MagneticSCFMACE")(MagneticSCFMACE)
register_observables_yaml(__file__, "observables.yaml")   # magforces
```

`config.toml` is byte-for-byte the one from the in-tree example — it refers to the feature only by
name (`magmom`, `magforces`, `MagneticScaleShiftMACE`), and does not know or care that it now comes
from an external package.

---

## 4. Versioning and compatibility

- The plugin **pins a MACE range** (`mace-torch>=1.0`) and builds only against the **public registry
  API** — `register_*`, `ObservableSpec`, `BaseMACE`, the model-transform hook. That surface is the
  contract; internal modules are not.
- MACE exposes its version; the loader can **warn and skip** a plugin whose declared target is
  incompatible, rather than importing it into a mismatched API.
- Because config and weights are backend-neutral, a model trained with the plugin installed produces a
  normal checkpoint — anyone who later installs `mace-magnetic` at a compatible version can load it.

---

## 5. Discovery, load order, fault tolerance

- At startup MACE enumerates the `mace.plugins` entry points (`importlib.metadata`) and imports each
  **once**; the decorators register on import.
- **Order-independent** — registration is by name. A name collision (two sources registering
  `MagneticSCFMACE`) is a hard error that names both, never a silent last-wins.
- A plugin that **raises on import is logged and skipped** — it never crashes MACE. The names it would
  have registered are simply absent, so `config.toml` referencing them fails fast with a clear
  "unknown model/observable" (and a hint that a plugin may be missing or broken).
- A missing optional dep is the same story as the extra: the module guards the import and registers
  nothing, so the capability is just off.

---

## 6. Testing and distribution

- **Tests live in the plugin repo.** Install `mace-torch` + the plugin and run the same three checks
  as [Extending MACE](extending_mace.md) §6.3 — the transform value test, the tiny-model golden
  (energy + `magforces`), and rotation equivariance. A CI matrix pins the MACE versions you support.
- **Distribute on PyPI.** Users `pip install mace-magnetic`; nothing needs to change in MACE or
  coordinate with a MACE release.

---

## 7. Plugin vs in-tree extra vs built-in

All three use the **same registries** and the **same `config.toml`**. The only axes are *where the
code lives* and *who releases it*.

| | Lives in | Released by | Reach for it when |
|---|---|---|---|
| **External plugin** (this doc) | its own repo / PyPI package | you, independently | you don't own MACE, want your own cadence, or the feature is niche / experimental / proprietary |
| **In-tree extra** ([Extending MACE](extending_mace.md)) | the MACE monorepo, `extras/<name>/` | the MACE team, with MACE | it belongs with MACE but carries a heavy optional dep and should be gated by a capability marker |
| **Built-in** | the main `mace-torch` tree (`nn/`, `data/`, `models/`) | the MACE team, with MACE | it is broadly useful, has no heavy dep, and is part of the default model |

Moving between them is mechanical: the module code and the `@register_*` calls are identical; you only
change where the files sit and how they're discovered (an entry point in a separate package → an entry
point in MACE's own `pyproject` → imported directly in the built-in tree). The user's `config.toml`
never changes.
