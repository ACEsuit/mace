# Deployment integration support

Every V1 runtime loads canonical safetensors/JSON through `load_deployed`.
The full-basis weights export is a derived artifact for external contraction
implementations. It is not another model checkpoint.

| Integration | Verified scope | Remaining work |
| --- | --- | --- |
| ASE | Energy, response and magnetic models, units, padding and compiled execution | See the per-family parity records |
| LAMMPS native | Python-free torch.export/AOTInductor, reference/cuEq, changing neighbours and MPI migration on one/two GPUs | Performance results are recorded separately from correctness |
| ML-IAP | Direct callback, reference/cuEq, eager/compiled regions, one/two MPI ranks | Python callback is required by this compatibility interface |
| OpenMM | Direct PyTorch callback on CPU, molecular/periodic trajectories, changing triclinic cells, energy/force parity with ASE | GPU execution and host-transfer performance are not yet measured |
| Symmetrix | Consumer source inspected at the revision below | Whole-model JSON translation and benchmark sweep remain pending; a contraction-only export does not load directly |
| libmace | Consumer source inspected at the revision below | Its published reader requires TorchScript and cannot load V1's neutral or full-basis files; no V1 numerical verification is claimed |
| torch-sim | DEP-4 pending | Adapter, compiled bucket reuse and trajectory parity |

## OpenMM direct PyTorch interface

Verified versions: OpenMM 8.6.1, torch 2.11, OpenMM-Torch commit
`b6532900ad825cb3d12193817a1e34a7f46ae6a3`, built from source. The PyPI
`openmmtorch==1.5` wheel inspected during this work does not expose
`PythonTorchForce`; a matching version string alone is not sufficient.
The pinned build and capability-enforced trajectory tests are in
`.github/workflows/nightly.yaml`. These workflow changes do not schedule runs
on the rewrite branch before publication.

```python
from mace_torch.deploy.openmm import OpenMMEvaluator

callback = OpenMMEvaluator(
    "model.mace-deploy", atomic_numbers=[8, 1, 1], periodic=True,
)
system.addForce(callback.as_force())
```

The species sequence must match OpenMM's particle order, or the selected
particle subset if the caller uses `force.setParticles(...)`. The callback
reads the current box from State, rebuilds neighbours, and evaluates the
shared model. OpenMM supplies nanometres; the adapter converts them to
Angstrom. It returns kJ/mol and kJ/mol/nm. Inputs such as charge and fields
remain in MACE units. `callback.evaluate(positions, cell, inputs=...)` also
allows direct calls with updated declared inputs.

This path executes Python. It uses neither a scripted model nor a serialized
Python model. OpenMM's optional callback pickling is refused: after restoring
simulation state, recreate the callback from the same neutral bundle and
attach the force. The native LAMMPS runtime remains the Python-free route.
The current graph builder prepares host metadata and transfers positions for
neighbour preparation; this implementation makes no claim of zero host-copy
overhead on GPU.

## Full-basis weights export

```bash
mace export full-basis-weights model.mace-deploy --output model-full-basis.json
```

The command writes a JSON sidecar and a safetensors file with format identity
`mace-full-basis-weights`, schema `1.0`. Each `blocks.<index>` tensor has shape
`[Z, A, mul]`. `blocks` records its stable model component, input/output irreps,
correlation, channel count, dtype and species-row mapping. `groups` partitions
`A` by increasing body order, then output declaration order. Every group records
its offset, count, coupling-tree labels, full-basis Frobenius norms and forward
factors. The sidecar states the real-basis change and Wigner phase, including
signs that need not match another implementation's Wigner tables.

For each element and channel, `reduced_to_full` computes the minimum-norm
representative in float64 and writes the source dtype. The forward applies no
additional path factor. A shared-species contraction has one row explicitly
marked `shared`; otherwise rows follow `species`. Bundle runtime precision and
backend pins do not cast or alter the stored weights being exported.

Correctness is checked by full-basis model predictions against both committed
tiny-anchor E/F/stress references, plus full-to-reduced reconstruction under
the existing numerical bounds. Projection is function-preserving; floating
point weight reconstruction is not a bit-exact identity. The source files,
goldens and tolerances remain unchanged. Checkpoint readers reject the exported
format, and `mace model convert --to-basis full` remains refused.

The schema is implemented for review. External maintainer agreement on names
and ordering, requested by DEP-3, has not taken place; this work has not sent
messages or published artifacts. No consumer is declared compatible merely
because the weight export exists.

## External consumer findings

Symmetrix revision `2aae0987ef63cc775f99f75b4fb45d8d0035a3ff` consumes an entire
model JSON: radial splines, embeddings, contractions, residuals and readouts.
Its supplied extractor reads a legacy model and imposes architecture limits,
including two interactions and l_max=3. Matching coupling-tree labels alone is
not a whole-model converter. The planned performance sweep must first verify
that translation, including spline approximation, on the same model.

libmace revision `e3413ae658530654894b6bac6a62a764a74c5115` loads
`torch::jit::Module` in `src/mace.cpp`; its Fortran interface wraps that runtime.
This differs from DEP-3's description of an independent full-basis contraction
reader. It cannot consume the proposed safetensors export. Supporting that
interface under V1's no-TorchScript requirement needs an AOT/native reader,
with an explicitly verified ABI, before adding a passing `bin_libmace` claim.

Sources: [OpenMM-Torch](https://github.com/openmm/openmm-torch/tree/b6532900ad825cb3d12193817a1e34a7f46ae6a3),
[Symmetrix](https://github.com/wcwitt/symmetrix/tree/2aae0987ef63cc775f99f75b4fb45d8d0035a3ff),
[libmace](https://github.com/stenczelt/libmace/tree/e3413ae658530654894b6bac6a62a764a74c5115).
