# Contracts for performance backends

V1 accepts accelerated implementations through its existing kernel, solver and
neighbour registries. The implementation of each kernel remains in its owning
backend. Checkpoints retain canonical weights and the reduced basis.

## Activation layouts and precision

`BackendCapabilities.activation_layouts` declares accepted activation layouts;
`native_layout` names the preferred one. The composite resolves one layout for
the chain at construction. The reference implements both `mul_ir` and `ir_mul`,
so fallback operations follow cuEq's native `ir_mul` layout. Descriptors record
that choice. Physical inputs, public descriptors and observable components retain
their established ordering. Canonical weight indices, parameter names and saved
weights do not depend on the activation layout.

`PrecisionPolicy.product` and `Descriptor.product` accept `ieee` or `tf32`.
Only float32 linear and fully connected tensor products may request TF32.
`PrecisionConfig.tf32()` requests it for those two operation classes and keeps
the existing float64 accumulation floor. A backend must declare `tf32` in its
`products` capability. Otherwise the effective descriptor uses IEEE and the
build report records the degradation. No process-wide matrix-product flag is
changed. A span uses the widest precision required by its members.

## Stateless edge spans

Kernel spec 1.0 defines `make_interaction_layer(descriptors)` and its fallback on
`None`. `EdgeConvolutionSpan` describes this ordered tuple:

1. Radial basis and spherical harmonics.
2. Optional distance transform.
3. Radial network and optional density network.
4. Receiver-summed channelwise convolution.

`RadialMLPDescriptor.channels[0]` is the full input width. With element
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

## Long-range preparation

`long_range_plan(graph, descriptor, ops)` selects discrete reciprocal indices,
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
