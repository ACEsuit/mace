# Contracts for performance backends

V1 accepts accelerated implementations through its existing kernel, solver and
neighbour registries. The implementation of each kernel remains in its owning
backend. Checkpoints retain canonical weights and the reduced basis.

## Activation layouts and precision

`BackendCapabilities.activation_layouts` declares accepted activation layouts;
`native_activation_layout` names the preferred one. `KernelSpec.activation_layout`
specifies the ordering of an operation's input and output features.
`BackendCapabilities.weight_layouts` describes parameter storage; every backend
accepts canonical `mul_ir` weights. The composite resolves one activation layout
for the chain at construction. The reference implements both `mul_ir` and `ir_mul`,
so fallback operations follow cuEq's native `ir_mul` layout. Descriptors record
that choice. Physical inputs, public descriptors and observable components retain
their established ordering. Canonical weight indices, parameter names and saved
weights do not depend on the activation layout.

`PrecisionPolicy.product` and `KernelSpec.product` accept `ieee` or `tf32`.
Only float32 linear and fully connected tensor products may request TF32.
`PrecisionConfig.tf32()` requests it for those two operation classes and keeps
the existing float64 accumulation floor. A backend must declare `tf32` in its
`products` capability. Otherwise the effective kernel_spec uses IEEE and the
build report records the degradation. No process-wide matrix-product flag is
changed. A span uses the widest precision required by its members.

## Stateless edge spans

Kernel spec 1.0 defines `make_interaction_layer(kernel_specs)` and its fallback on
`None`. `EdgeConvolutionSpan` describes this ordered tuple:

1. Radial basis and spherical harmonics.
2. Optional distance transform.
3. Radial network and optional density network.
4. Receiver-summed channelwise convolution.

`RadialMLPSpec.channels[0]` is the full input width. With element
conditioning it includes the radial features and two embeddings of
`element_embedding_width` each. `num_elements` describes the tables;
`layer_norm_epsilon` describes the normalization when present. Table parameters
are owned by their original element linear maps, outside the radial MLP's
`weight_numel`.

The call receives the live canonical member tensors in the existing basis and
transform order, followed by radial-network tensors, density-network tensors
when present, then sender and receiver embedding tables when conditioned. The
tables include their linear maps' normalization and retain their gradients.
`EdgeInputs.node_elements` indexes them even when the product basis shares
weights across species.

Without density, the output is the message tensor. With density, it is
`(message, density)`, where density has shape `[n_nodes, 1]` and is the sum of
`tanh(density_network(input)**2)` over each receiver's edges. The interaction
applies its existing normalization outside the span.

`EdgeInputs.receiver_runs` and `sender_runs` coexist when both are requested.
The derivative engine materializes required edge metadata before compilation
and restores exported edge forces to the caller's edge order. A span holds no
parameters or buffers and declares both weight-gradient and double-backward
capabilities. Unsupported training paths use the member operations.

Further radial families can define their own descriptors, shape matcher and
member-tensor order under this tuple protocol. They do not need another backend
registry or a change to the stateless call signature. The shared conformance
suite accepts explicit cases and checks outputs, live-weight gradients,
second derivatives, equivariance, layouts, padding and both edge-run views.

## Prepared inference regions

`mace_core.kernels.inference` adds two optional shapes alongside the existing
live-weight spans:

- `ProductReadoutSpec`: symmetric contraction, channel mixing, optional residual
  addition and optional linear readout. By default it returns node features as
  well as the readout. A terminal caller needing only the readout can explicitly
  set `return_features=False`, allowing the backend to avoid materializing an
  unused intermediate. Keep features enabled for subsequent layers or consumers.
- `InteractionFusionSpec`: input linear, radial/harmonic convolution, output
  linear and normalization, species-dependent skip, product and optional linear
  readout. The skip acts on the message for a plain interaction and on the input
  features for a residual interaction. Density normalization and both cutoff
  placements retain their existing algebra.

These are closed descriptions of existing operations, not a general graph
rewriter. The declarations validate the connected irreps, element tables and
activation layout. Each member retains its explicit precision and product
policy. `interface_dtype` separately records the floating input/output dtype;
internal compute policies do not silently change boundary casts. All weights
enter in canonical reduced form with the existing path
normalization. Nonlinear readouts, conditioned nonlinear interactions and other
unrepresented combinations continue through their existing operations. No
model family loses a path because a backend declines a region.

The optional Torch extension is
`mace_torch.backends.inference.InferenceBackend.prepare_inference(spec, weights)`.
An existing registered kernel backend may implement it without another registry.
It is not a new required method on `KernelBackend`. The public
`prepare_inference(backend, spec, weights)` helper returns `None` when the method
is absent or the backend declines, or a `PreparedInference` snapshot on success.
The caller retains the original operations on `None`; errors are not declines.
Selection is explicit and happens outside execution. Existing model builders
and native exporters do not automatically activate this optional API.
The composite and precision wrappers preserve the optional factory. Build the
request from the loaded model's resolved member specifications and interface
dtype; preparation does not rerun precision negotiation or replace those policies.

### Weight preparation and lifetime

`weights` maps each name in `spec.weight_roles` to the existing
`canonical_state(member)` mapping. This reuses canonical component paths and
tensor names, including model constants. For example:

```python
from mace_torch.backends.inference import prepare_inference
from mace_torch.serialization import canonical_state

weights = {
    "contraction": canonical_state(product.contraction),
    "linear": canonical_state(product.linear),
    "readout": canonical_state(readout),
}
prepared = prepare_inference(backend, product_readout_spec, weights)
# If prepared is None, retain the existing product and readout operations.
# Otherwise: features, values = prepared((message, element, residual))
```

For a complete interaction, `radial_basis` is the radial backend operator's
canonical state; its specification carries the cutoff. `normalization` contains
only the interaction body's root canonical state, `{"": {"neighbours": tensor}}`.
This role is replaced by `density` when normalization is learned. Product roles
are prefixed with `product.`. A distance transform has its own role when present.

The helper gives the backend detached copies of the supplied canonical tensors.
The backend returns `InferencePreparation(operation, tensors)`: a stateless
operation and derived tensors. It can fold compatible linear maps, incorporate
normalization or pack species tables during this preparation. The helper owns
copies of the resulting tensors as nonpersistent buffers. Preparation never
rewrites the canonical source and is not performed in `forward`.

The prepared object is an independent snapshot, not a hidden cache of a live
model. Reprepare after changing weights, device, dtype or specifications. Its
`.train(True)`, device/dtype conversion and canonical checkpoint save are refused
with a message directing the caller to the source model. Continue saving and
training that model normally. A deployment artifact produced with `torch.export`
may contain derived tensors; it is not a second canonical checkpoint format.

### Runtime and validation

The stateless operation receives `(inputs, tensors)` and returns a tensor tuple.
Input and output ordering are declared by the shape. Geometry, atom indices and
connectivity are live inputs, not weight preparation state. Outputs retain all
nodes, including ghost slots; ownership masks, graph reductions and MPI exchanges
stay outside the region. A region cannot cross a required communication boundary.

`SpanCapabilities(False, False)` is sufficient for first-derivative inference;
input adjoints are always required for forces and stress. A backend declaring
double backward must also reproduce those derivatives. Training continues through
the original operations; a caller requesting a Hessian selects a prepared region
only if it declares that capability. No `no_grad` context encloses execution.

GPU implementations must use the framework's current stream, leave tensors on
device and preserve CUDA capture. Prepared weights and inputs are read-only.
Backend execution scratch must not mutate shared prepared weights. The contract
does not add a host callback to each operation, require a CPU transfer or make a
new guarantee of zero overhead.

`run_inference_conformance` compares values, input derivatives and declared
second derivatives with independent reference operations at the existing backend
tolerances. It also checks that prepared buffers stay unchanged. Concrete backends
still need equivariance, compile/export, CUDA capture and native registration
tests, as well as representative changing-neighbour and distributed cases. The
contract fixtures compose existing reference operations; they provide no new
accelerated kernels and establish no GPU performance result.

Electrostatics continues through its separate solver contract below. Neither
shape captures a solver or changes its numerical selection. Sharing GPU buffers
with an external solver and binding a new backend into a native driver require
their own adapter and validation; the contracts do not claim those integrations
are already implemented.

## Long-range preparation

`long_range_plan(graph, kernel_spec, ops)` selects discrete reciprocal indices,
atom counts and periodicity outside the compiled model. An optional
`op.plan(geometry)` contributes solver-specific preparation through a mapping.
The plan travels under `long_range_plan`; it is runtime state, never checkpoint
state. Polar's preparation hook builds it before the model call when absent.

A caller may reuse a plan while cells, periodicity, structure counts and atom
counts remain unchanged. It must rebuild the plan after any of those change.
Positions and source multipoles may change. Reciprocal vectors, cell volumes
and position-dependent solver caches are evaluated from the current
differentiable geometry, so forces and strain derivatives retain their paths.
The per-forward `geometry.prepared` mapping is fresh; `geometry.plans` carries
solver preparation. V1 keeps its Fourier algebra and configured accumulation
precision.

External solvers use `mace.electrostatics_backends.torch`. A solver that is not
bit-identical remains model-affecting state, and checkpoint loading rejects a
silent substitution. Registering a solver does not change a model's selection.

## Device neighbour providers

Providers use `mace.neighbor_backends` or `mace.neighbor_backends.torch` and
return the existing `NeighborList` contract. ASE resolves the provider once and
uses `build_batch` for both host and device implementations. On NVIDIA CUDA,
an installed, importable `nvalchemiops` provider is preferred by default; CPU
and ROCm use vesin. An absent optional provider leaves vesin selected. An
import failure of the automatic CUDA choice is reported. Explicit provider
names and settings are strict and do not fall back.

The accelerated provider belongs to its external package. This contract does
not ship a second neighbour kernel or claim acceleration when that provider is
absent. A provider declaring `sorted_by_sender` must supply that ordering; the
common edge-plan adapter builds any other requested views after collation and
padding.
