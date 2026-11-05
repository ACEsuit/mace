# Native tiny checkpoints

These two safetensors/JSON pairs are converted test inputs, derived from the
committed `tests/golden/models/tiny_scaleshift.model` and `tiny_mace.model`.
They allow the JAX-only CI job to load native weights without installing torch
or reading pickle. Expected outputs still come from the original committed
references under `tests/golden/references`; these pairs define no new oracle.

`provenance.json` records the source and generated hashes. The production
FM-1 reduced-basis converter at local V1 commit `d467adf5` produced both pairs.
To reproduce in the development environment that has both torch trees:

```python
from pathlib import Path
import torch
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.train import write_model
from tests.parity.test_fm_convert_legacy import extract_here

torch.set_default_dtype(torch.float64)
for original, name in (("tiny_scaleshift", "tiny"), ("tiny_mace", "tiny_plain")):
    neutral = extract_here(
        Path("tests/golden/models") / (original + ".model"),
        Path("/tmp/jax-tiny-conversion") / original,
    )
    imported = import_neutral(neutral, DEFAULT_CATALOGUE)
    write_model(
        Path("packages/mace-jax/tests/fixtures") / name,
        imported.engine,
        imported.metadata,
    )
```

Review a regenerated pair as a fixture change. Neither the frozen inputs nor
the numerical references or tolerance table should change with it. The wheel
contains `src/mace_jax` only, so these fixtures are source-tree test assets.
