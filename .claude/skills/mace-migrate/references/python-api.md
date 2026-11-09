# Python and data migration

V1 exposes `mace_core`, `mace_torch` and `mace_jax`. It does not alias the old
`mace.*` imports. This is a caller migration, not a search-and-replace of package
names: the model factory takes resolved configuration and physical inputs, and
the graph builder returns tensor mappings rather than `AtomicData` objects.

Contents: [imports](#import-and-call-map) · [label keys](#label-keys) ·
[retained runtime controls](#retained-runtime-controls).

## Import and call map

| Legacy surface | V1 callable | Caller change |
| --- | --- | --- |
| `mace.calculators.MACECalculator` / `MagneticMACECalculator` | `mace_torch.calculators.MACECalculator` | Native checkpoints through `model_paths`; singular `model_path` fails. |
| `mace.calculators.mace_mp` | `mace_torch.calculators.mace_mp` | Canonical registry names; legacy family aliases warn. Publication is pending. |
| `mace.calculators.mace_off` | `mace_torch.calculators.mace_off` | Use OFF23 names and the declared precision. |
| `mace.calculators.mace_polar` | `mace_torch.calculators.mace_polar` | Preserve Polar precision; medium/large float32 parity is open. |
| `mace.calculators.mace_omol` | `mace_torch.calculators.mace_omol` | OMOL extra-large registry entry. |
| `mace.calculators.mace_mdp` | `mace_torch.calculators.mace_mdp` | MDP dielectric registry entry. |
| `mace.modules.MACE` / `ScaleShiftMACE` and response classes | `mace_torch.models.build.build_model` | Pass `ResolvedConfig`, observable catalogue, elements, heads, E0s and statistics. `plain` and `scale_shift` are distinct model kinds. Energy plus dipole uses `scale_shift` with both observables. |
| `mace.modules.loss` classes | `mace_torch.train.loss.build_loss` | Pass the structured loss configuration and declared outputs; no class-name registry. |
| `mace.data.utils.Configuration` / `mace.data.Configuration` | `mace_core.data.Configuration` | Explicit properties, weights and physical cell. |
| `mace.data.AtomicData.from_config` | `mace_torch.graph.collate.build_batch` | Build tensor mappings from configurations, elements and cutoff; no torch_geometric object. |
| `mace.data.utils.KeySpecification` | `mace_core.data.KeySpecification` | Explicit graph/atom key mappings at the read boundary. |
| `mace.data.utils.load_from_xyz` | `mace_core.data.read_configurations` | Pass a `KeySpecification`; receive configurations and isolated-atom energies. |
| `mace.tools.AtomicNumberTable` | `mace_core.elements.AtomicNumberTable` | Explicit element table shared by graph and model. |
| Direct `torch.load` of a legacy model | `mace_torch.deploy.loader.load_deployed` | Load the native JSON/safetensors pair; choose device, precision and backend at load time. |
| `mace.cli.run_train.run(args)` | `mace_torch.cli.run_train.run` | Takes `ResolvedConfig`, returns `TrainedModel`; parsing is separate. |
| Legacy CLI argument parser | `mace_torch.cli.run_train.parse` | Explicit flags or structured file; omitted fine-tuning fields remain omitted. |
| `ChebychevBasis` / `ChebyshevBasisGeneral` | `mace_torch.backends.radial.ChebyshevBasis` | Use `num_basis` and `include_constant`; no `r_max` parameter. Preserve whether the constant mode was included. |
| `mace.modules.radial.ZBLBasis` | `mace_torch.nn.radial.ZBLBasis` | No `r_max` constructor argument; the physical screening/cutoff convention is unchanged. |
| `mace.tools.cg` helpers | `mace_core.clebsch_gordan.full_symmetric_tensor_product_basis` | The new `mace_core.clebsch_gordan` module exposes canonical basis construction and conversion; adapt to its explicit irrep/basis contract. |
| `e3nn.o3.Irreps` in MACE callers | `mace_core.clebsch_gordan.irreps.Irreps` | Use `Irreps.parse`; this is the V1 descriptor, not an e3nn-compatible replacement. |
| `e3nn.o3.spherical_harmonics` in MACE callers | `mace_torch.backends.harmonics.spherical_harmonics` | Explicit V1 harmonic convention; do not forward arbitrary e3nn keyword arguments. |

Radial physical blocks receive `node_atomic_numbers` and `edge_index` rather
than a one-hot array and species table. The node embedding still takes one-hot
species. These are signature changes; the old call arguments cannot be forwarded.

For a converted or trained local native checkpoint:

```python
from mace_core.config.precision import PrecisionConfig
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.loader import load_deployed

model = load_deployed("model.json", device="cpu", precision=PrecisionConfig(model="float64"))
calculator = MACECalculator(models=model, device="cpu")
```

To use a different kernel backend, supply `backend="cueq"` to `load_deployed`
on a compatible CUDA installation and use the same device in the calculator.
Canonical weights require no e3nn/cuEq/OEq checkpoint rewrite. Electrostatics
solver replacement is more restrictive: a solver that changes numerics is model
state, not a transparent runtime choice.

`MACEOutput.total_energy` replaces the model dictionary's `energy`;
`node_energies` replaces `node_energy`. Additional outputs live in `extras`.
ASE still exposes its usual `energy`, `forces` and Voigt `stress` names; its
compatibility mapping also keeps `LES_alphas`, `LES_kappas` and `MACE_magmoms`.
The last maps to the fixed-point result `converged_magmom`. Do not rename ASE
keys merely because the core tensor name differs.

## Label keys

Shapes below describe one structure or one atom's value, respectively. Per-atom
arrays prepend the atom count. The twelve conventional file keys are unchanged.

| Convention | File key | Storage | Value shape |
| --- | --- | --- | --- |
| energy | REF_energy | graph | () |
| forces | REF_forces | atom | (3,) |
| stress | REF_stress | graph | (3, 3) |
| virials | REF_virials | graph | (3, 3) |
| dipole | dipole | graph | (3,) |
| polarizability | polarizability | graph | (3, 3) |
| charges | REF_charges | atom | () |
| total_charge | total_charge | graph | () |
| total_spin | total_spin | graph | () |
| elec_temp | elec_temp | graph | () |
| magmom | REF_magmom | atom | (3,) |
| magforces | REF_magforces | atom | (3,) |

`head` is no longer a default file key. Choose a head per source in
`data.heads.<name>` or pass `head_name` to the reader; a property in the XYZ does
not select the readout. Stress and virials accept 3×3 matrices or six-component
Voigt values (`xx, yy, zz, yz, xz, xy`). Flat nine-component arrays are rejected:
reshape them explicitly to `(3, 3)` before writing. Duplicate isolated-atom
energies for the same element/head must agree; conflicting references fail.

Training rejects nondefault property-key flags. The data API and `data prepare`
can translate custom keys into the common convention. This example reads an
existing file whose energy label is named `DFT_energy`:

```python
from mace_core.data import KeySpecification, read_configurations

keys = KeySpecification.from_defaults().apply_overrides({"energy_key": "DFT_energy"})
parsed = read_configurations("custom.xyz", keys, head_name="pbe")
configurations = parsed.configurations
```

Embedding inputs are declared separately with their graph/atom scope. Unknown
properties are carried without guessing shapes or converting every value into
a tensor. For large datasets prepare V1 HDF5 shards and their manifest, then
reference the prepared source per head; do not treat a legacy shard as a native
manifest by changing its filename.

Forces remain `-dE/dpositions`; exported edge gradients are `+dE/dvectors`.
Virials are minus the strain derivative and stress is its positive derivative
divided by volume. Aperiodic stress is zero; the physical cell is not replaced
by a padding box. The calculator scales dipoles with `length_units_to_A`, an
intentional correction to legacy. Direct dipole parity uses unit length factor.

Legacy `charges_history` contains one detached final snapshot. The equivalent
view is `output.extras["spin_charge_density"].detach().unsqueeze(-1)`, not a
trace of iterations. Magnetic SCF diagnostics are `fixed_point_history`, `fixed_point_steps` and
`fixed_point_converged` in the `FixedPointDriver` output extras. Neither convention requires an extra model output.

## Retained runtime controls

These controls are KEEP. ASE padding also has explicit `pad_num_atoms`,
`pad_num_edges` and padding-policy inputs. The six ML-IAP controls below apply
to the Python compatibility callback; the native runtime uses its own declared
capacities and execution contract, not a Python profiler.

| Runtime control | Status | Use |
| --- | --- | --- |
| MACE_ASE_PAD_NUM_ATOMS | KEEP | ASE atom padding when the explicit value is zero. |
| MACE_ASE_PAD_NUM_EDGES | KEEP | ASE edge padding when the explicit value is zero. |
| MACE_TIME | KEEP | ML-IAP per-step timing. |
| MACE_PROFILE | KEEP | ML-IAP profiling. |
| MACE_PROFILE_START | KEEP | First profiled ML-IAP step. |
| MACE_PROFILE_END | KEEP | Last profiled ML-IAP step. |
| MACE_ALLOW_CPU | KEEP | Allow CPU tensors in the ML-IAP coupling. |
| MACE_FORCE_CPU | KEEP | Force CPU evaluation in that coupling. |
