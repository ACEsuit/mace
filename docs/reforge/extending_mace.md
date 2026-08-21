# Extending MACE — a complete worked example

The [target layout §3](target_layout.md) lists the extension points and the "spectrum" (config knob →
observable row → registered plugin → new model). This doc is the other half: **one end-to-end
example of a genuinely complex feature**, walked change-by-change, so you can see how the pieces
compose and exactly which of them are config vs code.

**Three ways to ship a feature.** All three use the *same* registries and the *same* `config.toml`;
they differ only in where the code lives and who releases it:

- **Built-in** — in the main `mace-torch` tree, part of the default install and the default model.
- **In-tree extra** — in the MACE monorepo under `extras/<name>/`, optional and installed on demand
  (`mace[<name>]`), gated by a capability marker. Ships and is maintained with MACE.
- **External plugin** — a separate `pip`-installable package that hooks in through entry points, with
  no changes to the MACE repo; released on its own cadence. Its own doc:
  [Extending via a plugin](extending_plugin.md).

This example is built as an **extra** (which is exactly how `#1244` ships it today); the final section
shows the **built-in** variant, and the plugin doc shows it out-of-tree.

The running example is the real `MagneticMACE` family (`#1244`): the model takes per-atom magnetic
moments as a **vector input**, outputs their conjugate **magnetic forces** `−dE/dmagmom`, augments
training by randomly rotating the magmom field, and ships an **SCF inference wrapper** that relaxes the
moments to equilibrium. It deliberately touches every layer:

| Change | Layer | Config or code? |
|---|---|---|
| 1. New **input** (per-atom `magmom`, a 3-vector) | graph input | registered embedding (spherical harmonics + radial) |
| 2. New **derivative observable** (magnetic forces `−dE/dmagmom`, a `1o` vector) | observable table | **config only** |
| 3. **Data augmentation** (SO(3) rotation of `magmom`) | transform registry | small registered module |
| 4. **Loss** for `magforces` | loss config | **config only** |
| 5. **Models** (`MagneticScaleShiftMACE` + the SCF inference wrapper) | model registry | registered models |
| 6. **Calculator** exposure + the `[magnetic]` extra | deployment | small |

As an extra (`mace[magnetic]`), all of its code lives together under one feature directory
(`mace_torch/extras/magnetic/`) and **nothing here touches a shared MACE file** — each change is a
config field, a registry decorator, or an entry point. Everything below is illustrative — the
`packages/` tree is future work — but every mechanism, and the feature it describes, is the real one.

---

## 1. A new input feature — `magmom` per atom

Per-atom and per-graph inputs are declared, not hardcoded. `magmom` is a per-atom **vector** `[n, 3]`.
It does not enter as a raw number: a registered embedding expands it into **spherical-harmonic node
attributes** (its direction) plus a **radial embedding of its magnitude** (an invariant), which the
interaction blocks then tensor-product into the node features.

```python
# mace_torch/extras/magnetic/embedding.py  — lives in the extra, registered by decorator
import sphericart.torch                                  # the [magnetic] extra's dependency
from mace_torch.nn import register_input_embedding

@register_input_embedding("magmom")
class MagmomEmbedding(torch.nn.Module):
    """Per-atom magmom vector [n, 3] -> (angular node attrs, invariant node feats)."""
    def __init__(self, max_ell: int, num_radial: int):
        super().__init__()
        self.sph = sphericart.torch.SolidHarmonics(max_ell)   # direction -> Y_l  (l=1 is 1o)
        self.radial = ChebyshevBasis(num_radial)              # |magmom| -> {num_radial}x0e

    def forward(self, magmom):                     # magmom: [n_nodes, 3]
        node_attrs = self.sph(magmom)              # spherical-harmonics irreps (1o + 2e + …)
        node_feats = self.radial(magmom.norm(dim=-1, keepdim=True))
        return node_attrs, node_feats              # wired into the interaction tensor product
```

```toml
# config.toml (the training run config) — declare the input; the data-key convention names its source
[data.inputs]
magmom = { key = "REF_magmom", per_atom = true, irreps = "1o" }
```

No shared-file changes: the model reads `config.data.inputs`, finds `magmom`, and wires
`MagmomEmbedding`. The magmom vector is expanded through spherical harmonics, so its `l=1` part is
**`1o`** (odd parity) — the model treats it as a polar vector, which is why energy is invariant only
under a *joint* rotation/inversion of positions **and** moments, not moments alone.

A per-system input (total charge, spin, electronic temperature) uses the same recipe — only the
`per_atom` flag changes: `true` folds into node features, `false` into the graph-level features.

## 2. The derived output — magnetic forces `−dE/dmagmom` (config only)

The magnetic model does **not** read out a magnetic moment: its only readout is **energy** (invariant
scalars). The magnetic output is the moment's **conjugate force**, `magforces = −dE/dmagmom`, obtained
by autograd exactly like `forces = −dE/dpositions`. It is a per-atom `1o` vector, declared as a
derivative observable:

```yaml
# mace_torch/extras/magnetic/observables.yaml  — the extra ships its own observable rows
magforces:
  derivation: "autograd(energy, wrt=magmom)"   # -dE/dmagmom
  per_atom: true
  irreps: "1o"                                  # a 3-vector, same convention as magmom
  default_loss_weight: 1.0
```

```toml
# config.toml
[model]
observables = ["energy", "forces", "stress", "magforces"]
```

The derivative engine already differentiates energy w.r.t. positions and strain; adding `magmom` as a
target is the same machinery (legacy `--compute_magforces`). **Zero code.** Declaring an observable
with no matching data key is a hard error.

### Illustrative: an equivariant moment readout (not in `#1244`)

`#1244` predicts **no** magnetic moment — its magnetic output is `magforces` above. But an equivariant
readout is a one-row change in the observable table, so here is what a *predicted* moment would look
like. **This block is illustrative** — it is not in the current implementation; it is shown only to
demonstrate the readout mechanism.

```yaml
# mace_torch/extras/magnetic/observables.yaml  — illustrative, NOT in #1244
magnetic_moment:
  derivation: readout        # a learned equivariant readout over node features
  per_atom: true
  irreps: "1o"               # same convention as the magmom input
  normalization: "component" # scale-only; a 1o vector can be scaled but not shifted
  default_loss_weight: 1.0
```

```toml
# config.toml  — illustrative
[model]
observables = ["energy", "forces", "stress", "magforces", "magnetic_moment"]
```

`MACEOutputs` would build the equivariant `1o` readout head automatically; the result appears as
`output.extras["magnetic_moment"]`. **Zero code** — the head, its typed output, its `normalization`
and its loss term are all derived from this one row. (`normalization` is a user knob: a non-scalar like
`1o` can be scaled but not shifted; only scalars such as energy take the classic **scale-shift**.) That
is the payoff of the declarative table: a genuinely new *predicted* property is a row, not a model
change.

## 3. Data augmentation — random SO(3) rotation of `magmom` (a registered transform)

Magnetic training augments by randomly rotating the **magmom field alone** — positions are held. It
samples magmom orientations; it is a magmom-space augmentation, not a whole-system rotation. This is a
**data transform**, not model code:

```python
# mace_torch/extras/magnetic/transforms.py
from mace_torch.data import register_transform

@register_transform("rotate_magmom")
class RotateMagmom:
    """One random SO(3) rotation of the magmom field (and its magforces); positions unchanged."""
    def __call__(self, config, rng):
        R = random_rotation(rng)                       # a proper rotation
        config.magmom = config.magmom @ R.T
        if config.magforces is not None:
            config.magforces = config.magforces @ R.T  # the label rotates with its input
        return config
```

```toml
# config.toml
[data]
transforms = ["rotate_magmom"]     # legacy --data_aug_magmom
```

## 4. Loss — a term for `magforces` (config only)

Because `magforces` is a declared observable, its loss term is **generated automatically** with its
`default_loss_weight`; you only override the weights in config. Tuning the loss is never new code:

```toml
# config.toml
[loss]
weights = { energy = 1.0, forces = 10.0, magforces = 1.0 }
```

**Loss hyperparameters are config too**, not a new loss. A common case is *not* writing a new reduction
but tuning one — e.g. switching to a Huber loss and setting its `huber_delta` (already supported today
via `--huber_delta`). In the reforge that is a per-loss `params` block:

```toml
# config.toml
[loss]
weights = { energy = 1.0, forces = 10.0, magforces = 1.0 }
type    = "huber"
params  = { huber_delta = 0.02 }      # a loss hyperparameter — config, not code
```

Only a genuinely new *reduction* (say a custom magnetic-anisotropy penalty) is code — one
`@register_loss` module selected by `LossConfig(name=...)`. Re-weighting and re-tuning existing losses,
as above, need none.

## 5. The models — the magmom channel is real code

Everything so far is config plus two small registered modules. The models themselves are the code:

```python
# mace_torch/extras/magnetic/model.py
from mace_torch.models import BaseMACE, register_model

@register_model("MagneticScaleShiftMACE")
class MagneticScaleShiftMACE(BaseMACE):
    """Energy model with the magmom input channel; energy readout + magforces = -dE/dmagmom.
    The magmom tensor-product channel is model code, not a readout row — the honest boundary."""
    def forward(self, graph):
        ...

@register_model("MagneticSCFMACE")
class MagneticSCFMACE(torch.nn.Module):
    """Inference wrapper: relax magmom with LBFGS until magforce ~ 0, then report the energy and the
    equilibrated moments. It deliberately does NOT differentiate through the loop — it is an eval-time
    equilibration convenience, not a training-time implicit-diff hook."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, graph):
        magmom = graph["magmom"].requires_grad_(True)
        # optimizer = torch.optim.LBFGS([magmom]); step(closure) with magmom.grad = -magforces
        ...
```

```toml
# config.toml
[model]
model = "MagneticScaleShiftMACE"       # the trainable model
# MagneticSCFMACE wraps it for inference-time equilibration; it is not a training transform
```

`MagneticScaleShiftMACE` is a real `BaseMACE` subclass — the magmom channel lives in the forward, so it
is genuine model code, unlike the config rows above. `MagneticSCFMACE` is a thin **inference wrapper**;
because it is not differentiated through, it needs no implicit-diff contract. Both are **registered,
not patched** — no shared-file changes. (If a variant needed an accelerated long-range solver, that
would be a swappable **electrostatics backend** with its own entry-point registry — the model never
selects it.)

## 6. Running it: calculator, packaging, tests

The feature is trainable at this point. Three last pieces make it usable and safe, each following an
existing pattern.

### 6.1 Using it in a calculator (no code)

The ASE calculator reads the observable table for what each output is (per-atom vs per-graph), so it
returns the magnetic outputs by name with nothing to wire:

```python
# a user script (ASE) — nothing in the package changes
atoms.calc = MACECalculator("my_magnetic_model.model")
atoms.get_potential_energy()                 # energy
atoms.arrays["MACE_magmoms"]                 # equilibrated moments (from the SCF wrapper)
atoms.arrays["magforces"]                    # -dE/dmagmom
```

### 6.2 Packaging the extra

Packaging the magnetic extra is three coordinated places. Magnetic needs a fast spherical-harmonics
library (**`sphericart-torch`**, used by the embedding in §1), so that dependency goes in the optional
group; a feature with no such dependency just leaves the group's requirement list empty.

**(a) Declare the optional dependency group** in the package's `pyproject.toml`. The extra is named
after the **whole capability**, not the library — `[magnetic]` gates the entire magnetic feature:

```toml
# packages/mace-torch/pyproject.toml
[project.optional-dependencies]
magnetic = ["sphericart-torch==1.0.9", "torch-geometric"]   # NOT in [project.dependencies]
```

```bash
pip install mace-torch              # base: no sphericart → the magnetic feature is not available
pip install mace-torch[magnetic]    # opt in: deps present → the whole magnetic capability turns on
```

**(b) The guard lives in the feature's own module, not in a shared file.** This block sits in the
magnetic feature's **own** `__init__.py`, and MACE reaches it through the `mace.plugins` entry point,
never by naming `magnetic`. So registration and the dependency check are both local to the feature:

```python
# mace_torch/extras/magnetic/__init__.py   (or a separate mace-magnetic package)
try:
    import sphericart.torch
except ImportError:
    HAVE_MAGNETIC = False        # dep missing → this module registers nothing; MACE is untouched
else:
    HAVE_MAGNETIC = True
    register_model("MagneticScaleShiftMACE")(MagneticScaleShiftMACE)
    register_model("MagneticSCFMACE")(MagneticSCFMACE)
    register_transform("rotate_magmom")(RotateMagmom)
    # ... and the magforces observable row
```

```toml
# this feature's OWN pyproject.toml — an entry point, NOT an edit to a shared file
[project.entry-points."mace.plugins"]
magnetic = "mace_torch.extras.magnetic"     # MACE imports this on discovery → its decorators run
```

MACE enumerates `mace.plugins` and imports whatever is installed; the loader tolerates a module that
registers nothing. There is **no `if magnetic` branch anywhere in MACE**: install the extra → deps
present → the module registers on discovery; skip it → nothing registers and
`model = "MagneticScaleShiftMACE"` fails fast with "unknown model".

**(c) Gate the tests on a capability marker** so they *skip* where the extra is absent and are
*required* where it is (the skip-o-fail contract), so a broken-but-installed extra fails CI instead of
silently skipping:

```python
# tests/extensions/magnetic/test_magnetic.py
@pytest.mark.magnetic     # capability marker (requires sphericart-torch), auto-added from the dir
def test_magforces_golden(): ...
```

**The whole extra lives in one directory.** Every file the magnetic feature owns — the embedding (§1),
its observable row (§2), the transform (§3) and the models (§5) — sits under a single feature
directory, wired by the `__init__.py` above:

```text
mace_torch/extras/magnetic/          # everything the feature owns lives here
├── __init__.py                      # the registration entry point (the mace.plugins target)
├── embedding.py                     # MagmomEmbedding             (§1)
├── observables.yaml                 # the magforces row           (§2)
├── transforms.py                    # RotateMagmom                (§3)
└── model.py                         # MagneticScaleShiftMACE / MagneticSCFMACE   (§5)
```

**Nothing of the extra touches a shared file**, and `config.toml` refers to it only by name (`magmom`,
`magforces`, `MagneticScaleShiftMACE`), never by path.

### 6.3 Tests (three kinds, all small)

1. **Value test** for the transform — a fixed input through `rotate_magmom` gives the expected rotated
   `magmom` (and `magforces`).
2. **Golden** for a tiny trained model — energy + `magforces` on committed fixtures, so the numbers can
   never silently drift.
3. **Equivariance** — under a proper rotation the `magmom` input and the `magforces` output rotate with
   the frame while the energy stays invariant. (Since `magmom` is `1o`, inversion is a *joint*
   operation on positions and moments — there is no moments-only symmetry to test.)

### 6.4 The command line — no new flags

The feature adds **no new CLI flags**; the command is unchanged (`mace train --config config.toml`).
The launcher is config-file-first with **dotted CLI overrides**, so every field the feature declares is
reachable on the command line for free (unknown keys are a hard error):

```bash
mace train --config config.toml \
    data.inputs.magmom.key=REF_magmom \
    model.model=MagneticScaleShiftMACE \
    loss.weights.magforces=2.0
```

---

## What was config vs code

- **Config only:** the input declaration, the `magforces` observable, the loss weights, the transform
  chaining, the model selection.
- **Small registered modules:** the input embedding, the augmentation transform, and the two models.
  All live inside the extra directory.
- Because we shipped it as an **extra**, **zero** shared files were touched: everything entered through
  a config field, a registry decorator, or an entry point, and all of the code sits under
  `mace_torch/extras/magnetic/`.

### If it shipped built-in instead

If the maintainers decided magnetic belongs **built-in** rather than as an extra, the *mechanisms* are
unchanged — only where the files sit and a couple of packaging details drop away:

- The same modules move from `mace_torch/extras/magnetic/` into the main `mace-torch` tree
  (`mace_torch/nn/`, `mace_torch/data/`, `mace_torch/models/`), still registered by the same decorators.
- The observable row goes in the shared `defaults/observables.yaml` instead of a feature-local file.
- No `mace.plugins` entry point, no `[magnetic]` optional-dependency group, no capability marker — its
  dependencies (here `sphericart-torch`) would be base dependencies and its tests run unconditionally.

And one rule relaxes. An extra must **not** touch a shared file — that is precisely what keeps it
self-contained — but built-in work **may**. Adding `MagmomEmbedding` straight into the existing
`mace_torch/nn/embedding.py` is perfectly fine for built-in functionality; it is not off-limits. Keeping
each feature in its own file (`mace_torch/nn/magmom_embedding.py`) is **preferred for readability, not
required**. So "no edits to existing shared files" is a property of the **extra path**, not a blanket
prohibition.

### Reusing it under JAX (inference)

Everything above is `mace-torch`. Two things are shared across backends for free: the **`config.toml`**
and the **trained weights** — the canonical layout is backend-neutral, so a torch checkpoint loads into
the JAX model. Everything else in this extra is torch: `MagmomEmbedding` (which uses `sphericart-torch`),
the two magnetic models, and the observable row the extra ships — they sit in the torch extra, not in
`mace-core`. To run under JAX you re-provide those in `mace-jax` (same `extras/magnetic/` layout, JAX
registries, `sphericart`'s JAX bindings). A **standard** model needs no port (its modules and observables
already ship in `mace-jax` / `mace-core`); a **custom** extra does. Only the framework-agnostic contract
a backend builds against — the `ObservableSpec` schema, the config schema, the canonical layout — lives
in `mace-core`. If you want the declarative rows shared instead of duplicated, lift them from the extra
into `mace-core`; then only the module code stays per-backend.

The takeaway for methodological work: whether it ships as an extra or built-in, even a feature that adds
a vector input, its conjugate force, augmentation, a loss term, and an SCF inference wrapper is a
bounded set of **declarations plus a couple of registered modules** — the "hard" part is only the
genuinely new physics in the forward, and even that plugs in through the registry rather than rewriting
the framework.
