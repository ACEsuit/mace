# OFF23 medium periodic stress

The existing OFF23-medium reference covers three nonperiodic structures and
therefore has no stress channel. JAX-1 also promises stress parity, so this
family captures the same published artifact on the existing `slab_vacuum`,
`slab_zero_vacuum` and `triclinic_bulk` fixtures. Their elements all belong to
the model's original element table. No geometry or existing reference changes.

`targets/off_medium_periodic.py` loads the frozen legacy calculator on CPU in
float64 and verifies the published artifact's original SHA-256 before and
after evaluation. The capture includes its complete calculator surface,
including nonzero periodic stress. It imports neither V1 runtime.

```bash
python tests/golden/regenerate.py --target off-medium-periodic --i-know-what-i-am-doing
```

This target is opt-in because the checkpoint may need downloading. The source
URL, digest, fixtures, dtype and regeneration command are recorded in the JSON.
`test_off_medium_periodic.py` validates the manifest and, when network access is
explicitly enabled, compares the frozen calculator with this reference. The
JAX conformance helper combines this supplemental fixture set with the original
molecular reference, requiring both files to identify the same source bytes.
