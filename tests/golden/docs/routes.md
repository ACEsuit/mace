# The fourth door: models that cannot be driven as calculators

Not a family of goldens and not a regeneration target. `routes.py` is the
adapter layer that several families need to reach the harness at all, and it
is family-agnostic on purpose.

The calculator surface is driven by `get_potential_energy` / `get_forces` /
`get_stress`. **A model with no energy cannot be driven that way at all.**
`MACECalculator(model_type="DipoleMACE")` and `…="DipolePolarizabilityMACE"`
leave no `energy` in `results`, so the first accessor raises
`PropertyNotImplementedError` and no snapshot happens. The schema is not the
problem — every key those families emit resolves to a channel — the *driving
convention* is.

`routes.py` holds the two adapters that follow from this:

* `CalculatorRoute` drives `calculate()` directly and hands back `results`,
  declaring `golden_surface = SURFACE_CALCULATOR` so the calculator's
  vocabulary is used. It forwards every attribute lookup to the wrapped
  calculator, and that part is load-bearing rather than tidy: the harness
  recovers what an evaluation *reads* by inspecting the object it was handed
  (`info_keys`, `arrays_keys`, `charges_key`, …), so an adapter that only
  forwarded outputs would make every reference taken through it record the
  inputs of a reader that never ran.
* `ForwardRoute` presents a `forward` as an evaluation, building the graph
  inside a float64 scope — not casting a float32 graph up afterwards, which
  costs ~2e-8 relative, sits inside the fp64 row, and so reads as agreement
  while making a bit-exact comparison impossible.
