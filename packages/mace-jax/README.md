# mace-jax

Native JAX inference for reduced-basis, single-head MACE energy checkpoints.
Requires Python 3.12+, JAX 0.11+ and Flax NNX 0.12.8+. Install alongside the local core:

```bash
python -m pip install -e packages/mace-core -e packages/mace-jax
```

Load a native V1 safetensors/JSON pair and evaluate an ASE structure:

```python
from ase.build import molecule
from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms

apply, params = load_model("model.json", dtype="float64")
graph = graph_from_atoms(
    molecule("H2O"),
    elements=apply.elements,
    cutoff=apply.config.model.r_max,
    dtype=apply.dtype,
)
outputs = apply(params, graph)
print(outputs.total_energy, outputs.forces)
```

`params` is the NNX state, including model buffers; `apply` merges it into the
static graph definition without changing it. The output is the shared
`MACEOutput`, with total energy `[G]`, node energies `[N]`, forces `[N, 3]`,
stress `[G, 3, 3]` and virials `[G, 3, 3]`. Units are eV and Å. Forces are
`-dE/dR`; virials are `-dE/dstrain`; stress is `+dE/dstrain / volume` and is zero
for fully nonperiodic structures. The geometry uses symmetric strain.

`graph_from_atoms` also accepts a list of structures and uses the shared
natural batching schema, including isolated atoms. The apply boundary expects
an unpadded graph in the model dtype with head index zero. `compute=()` returns
energies alone. Padding, compiled inference and an ASE calculator belong to
JAX-2. The loader enables JAX float64 support before constructing arrays;
float32 inference retains float64 E0 storage and model-level accumulation.
Inference uses the highest matmul precision, so GPU float32 does not use TF32.

For XYZ files:

```bash
mace-jax --model model.json --configs input.xyz --output predictions.xyz
```

Use `--dtype float32` or `--device cuda` explicitly. Install the appropriate JAX
GPU runtime separately. The CLI writes `MACE_energy`, `MACE_forces` and
`MACE_stress` on every frame. There is no training command.

## Supported artifacts

| Registry ID | Interaction variants | Additional radial terms | Frozen reference |
| --- | --- | --- | --- |
| `mace-mp-0-small` | residual / residual | Bessel, polynomial cutoff | `mp_small_e3nn_cpu_fp64.json` |
| `mace-mpa-0-medium` | density / density_residual | Agnesi, ZBL | `mpa0_medium_e3nn_cpu_fp64.json` |
| `mace-off23-medium` | plain / residual | Bessel, polynomial cutoff | `off_medium_e3nn_cpu_fp64.json` |

All three use linear and nonlinear energy readouts. They are verified against
their committed references in both float64 and float32. Their converted local
artifacts are available from the FM-2 campaign. Publication and the FM-3 URL
registry are pending, so pass a local path for now; an unresolved registry ID
raises an actionable error. HTTPS pairs use the existing core artifact cache.
The loader never reads a pickle.

OFF23's molecular reference is supplemented by
`off_medium_periodic_e3nn_cpu_fp64.json`, captured independently with frozen
legacy on the three existing periodic fixtures, to check its stress channel.

Fine-tuned descendants with the same architecture can load, but the certified
list contains only these three original artifacts. Headless originals have a
single `default` head after conversion. Multi-head checkpoints require explicit
`mace model select-head` conversion before loading.

| Outside the supported list | Failing support gate |
| --- | --- |
| Large tiers and remaining MP aliases | Outside the three-artifact JAX conformance roster, even where a torch golden exists |
| MH-0, MH-1, OMOL | Multi-head runtime and/or unsupported architecture |
| MDP | Dipole and polarizability outputs |
| Polar and LES | Electrostatic architecture and outputs |
| Magnetic family | Magnetic inputs, outputs and fixed-point architecture |
| ANI-CC | No committed support reference; retired from the model registry |

Unsupported observables, radial kinds, interaction variants, full CG bases and
incompatible canonical layouts fail during loading. The loader reports missing
blocks together. Canonical contraction paths are aligned by coupling-tree name,
independently of the kernel implementation's Python class.

## Implementation and checks

Flax NNX holds the modules and canonical arrays. Its split/merge interface
provides the pure apply function while the equivariant algebra and path names
come from `mace_core`. This follows JAX-1's recommended module library. The
registered reference backend implements the current shared op protocol,
including the radial MLP added after the ticket was written. It declares
`supports_double_backward=False`; all first derivatives use native JAX autodiff.
The package imports neither torch stack nor e3nn/e3nn-jax.

```bash
PYTHONPATH=. python -m pytest packages/mace-jax/tests -m 'not network'
MACE_CI_ALLOW_NETWORK=1 PYTHONPATH=. python -m pytest \
  packages/mace-jax/tests/test_mace_jax_parity_neutral.py \
  -m 'network and not gpu' --jax-model-artifacts /path/to/local-artifacts.json
```

The optional artifact map is a JSON object from the three registry IDs to
converted checkpoint paths. GPU cases carry the existing `gpu` capability
marker. `MACE_REQUIRE_CAPS=gpu,network` makes missing required capabilities fail.
The conformance helper uses the frozen golden harness and tolerance table;
references are never generated by the JAX tests. The per-PR `jax-anchor` job
runs the two tiny-anchor precisions without model downloads or a torch install.
