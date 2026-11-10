# A readable MACE energy forward

`mace_forward.py` is an educational, evaluation/inspection-only implementation
in plain JAX. It is not the reference backend, a training path or a mirror of
the newest production architecture. There is no JAX training loop here.

```sh
python -m pip install -r requirements/tutorials.txt
python tutorials/mace_forward.py
python tutorials/notebooks/01_math_walkthrough.py
python tutorials/notebooks/02_forward_demo.py
```

Use `marimo edit tutorials/notebooks/01_math_walkthrough.py` for interaction.
Plain Python executes the notebooks headlessly and exits. The forward imports
only JAX and the standard library, and loads a local JSON of stored weights.
The fixture records cutoff, radial basis, angular order, correlation, layer
count and **full** CG basis. Path normalisations are applied during evaluation.
E0 is added outside the per-node scale and shift.

The fixture is a fully aperiodic water molecule with six directed edges. The
native JSON/safetensors oracle lives in tests/fixtures. Only tests/oracle.py
imports the production stack. `python -m pytest tutorials/tests` checks native
fp64 agreement, invariance, E0 assembly, harmonics and both notebooks. Install
core/torch for that cross-check; standalone execution does not need them.
`MACE_TUTORIAL_ORACLE_PYTHON` optionally points to a separate native interpreter.

## Deliberate scope

Omissions below apply to this teaching file, not to the V1 product.

| Develop capability | `tutorials/mace_forward.py` | Disposition |
|---|---|---|
| E0 term, node embedding, Bessel×cutoff, spherical harmonics, interaction + symmetric contraction, per-layer readout, per-head scale/shift | implemented | COVER |
| Forces, virials, stress, hessians, edge forces, atomic stresses (`get_outputs`, `mace/modules/utils.py:270`) | absent | DROP — needs the autograd/strain glue; reading the model does not (ARCH-4 (#1563) owns them for v1) |
| ZBL pair repulsion (`--pair_repulsion`, `radial.py:149-223`) | absent | DROP — off by default; an additive term that teaches nothing about equivariance |
| Multi-head readouts and head selection | single head only | DROP — the head index is a column selector, not a block (FT-1 (#1580)) |
| `AtomicDipolesMACE`, `AtomicDielectricMACE`, `EnergyDipolesMACE` | absent | DROP — same skeleton, different readout (RET-4 (#1600) / ELEC-2 (#1592)) |
| Magnetic family: `MagneticScaleShiftMACE`, `MagneticSCFMACE`, magmom as an input feature, magforces as an output, the SCF fixed point (`mace/modules/extensions.py:1428-2105`) | absent | DROP — a second equivariant channel and a fixed-point solver; out of a first reading (MAG-1 (#1584)) |
| Electrostatics: `MACELES` (`mace/modules/extensions.py:142-660` — BEC, `latent_charges`/`latent_dipoles`/`latent_alphas`/`latent_kappas`/`latent_quads`, `external_field`) with its calculator-side `eps_infty` / `keep_neutral` handling (`mace/calculators/mace.py:816-845`), and `PolarMACE` k-space | absent | DROP — long-range solvers are their own subject (ELEC-1 (#1591) / ELEC-2 (#1592)) |
| Joint embedding / `embedding_specs` / `use_embedding_readout` graph-level inputs | absent | DROP — an additive embedding term, not a block of the equivariant core (ARCH-3 (#1562)) |
| LoRA (`mace/modules/lora.py`), distance transforms (`AgnesiTransform`, `SoftTransform`), `use_so3`, `use_agnostic_product`, `edge_irreps` | absent | DROP — hyperparameter variants of blocks the tutorial already shows |
| cueq / oeq backend dispatch, `torch.compile`, TorchScript, LAMMPS ghost-atom branches | absent | DROP — the tutorial has no backends and no deployment target |
