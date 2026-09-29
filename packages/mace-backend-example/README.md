# mace-backend-example

A MACE kernel backend in a distribution of its own, and the template for
writing one. It computes every op MACE dispatches, in plain torch, on kernels
it registers itself. Nothing in `mace-core` or `mace-torch` names it:
installing this distribution is the only step that makes it exist.

It is never published. It lives in the MACE repository so that CI checks it
on every change, which is what keeps the claim that a backend needs no edit
to MACE true rather than historical.

## Writing a backend

A backend is three things: an object that answers the kernel contract, an
entry point that registers it, and a run of the conformance suite that shows
it computes what the reference computes.

### 1. The contract

`mace_core.kernels.protocol.KernelBackend` lists what the object provides:

- `name`, and `capabilities()`: a `BackendCapabilities` saying which ops it
  builds, on which devices, in which dtypes and feature layouts, whether its
  ops can be differentiated twice, and which version of the kernel contract it
  was written against (`spec_version`). Subclass it and override `supports`
  when what the backend can build depends on the shape;
  `ExampleCapabilities` declines a convolution over more than one copy of an
  irrep, and a skip against anything but scalars.
- One `make_<op>(descriptor)` per op in `DISPATCHED_OPS`. A descriptor is the
  op's shape; it carries no weights. Returning `None` from
  `make_spherical_harmonics` or `make_radial_basis` leaves them to the
  reference, and so does `make_interaction_layer` for a fused span.
- For the three ops that own weights, `to_canonical`, `load_canonical` and
  `initialize_weights`, in the layout `mace_core.kernels.canonical` pins. That
  layout is what a checkpoint holds, so a checkpoint written with any backend
  loads into this one and back. A backend that holds its weights in another
  order converts at these three methods and nowhere else; this one holds the
  canonical order and they are views.

Import only the contract. `backend.py` imports `mace_core` and `torch`; the
model calls the backend, never the other way round.

### 2. Your own kernels

`ops.py` registers three `torch.library.custom_op`s, and each needs three
registrations:

- the forward;
- `register_fake`, which returns an empty tensor of the right shape, so that
  `torch.compile` can trace the op without running it. Sizes come from the
  inputs and from host integers such as `num_nodes`; never read a size off a
  device tensor, or every call is a new graph;
- `register_autograd`, whose backward is written in differentiable torch
  operations. Training on forces differentiates the backward, and a backward
  that is right but not differentiable raises nothing: the forces just train
  wrong. `gradgradcheck` in the suite is what catches that.

### 3. The entry point

```toml
[project.entry-points."mace.kernel_backends.torch"]
example = "mace_backend_example:ExampleBackend"
```

A name is registered once. Two distributions registering the same name is an
error, since which one a model got would depend on installation order.

### 4. The conformance suite

```python
from mace_torch.backends.conformance import run_backend_conformance

run_backend_conformance("example", precision="float64", compile_ops=True)
```

It builds each case the backend says it supports next to the reference, and
checks the canonical weights both ways, the values, the first and second
derivatives, equivariance, that a declined case is refused, that the op
compiles whole and does not recompile for a new size, and, on CUDA, that it
replays from a captured graph. The tolerances are fixed; a backend meets them.

```bash
pip install -e packages/mace-backend-example
python -m pytest packages/mace-backend-example/tests
```
