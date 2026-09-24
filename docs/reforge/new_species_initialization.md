# Adding elements to a trained model

A fine-tune keeps its foundation model's whole element table by default, so a
model fine-tuned on silicon and oxygen can be fine-tuned again on iron. To go
beyond the foundation's table, the model is first extended with the new
elements. Extension is an artifact operation: a checkpoint goes in, a
checkpoint comes out, and nothing is trained.

```python
from mace_torch.finetune.extend import NewSpeciesInit, extend_elements

extend_elements(
    "foundation.json",
    "foundation_with_iron",
    e0s={"default": {26: -3.46}},          # one energy per head, in eV
    init=NewSpeciesInit(kind="fresh", seed=0),
)
```

The result is started from like any other foundation model.

## What changes, and what does not

The model is rebuilt over the larger table. Every canonical tensor that does
not depend on the elements is copied unchanged. The per-element ones are
carried row by row:

| Tensor | Layout | Rows for an old element | Rows for an added element |
|---|---|---|---|
| node embedding | one column of weights per element | copied | from the spec |
| skip connection | `[Z, ...]` | copied | from the spec |
| symmetric contraction | `[Z, A, mul]` | copied | from the spec |
| isolated-atom energies | `[heads, Z]` | copied | the energy given |

Which tensors are per element is read by comparing the canonical shapes of the
model over the old and the new table. A tensor whose shape changes and is not
in this table is refused by name, rather than left at the rebuild's draw.

**A structure holding only old elements computes exactly what it did**, energy
and forces to the last bit. Per-element parameters are gathered by element, so
such a structure reads only old rows, and every other tensor is the parent's.

## The initialization spec

| Spec | New rows | Needs |
|---|---|---|
| `fresh` | what a freshly initialized model over the new table draws there | `seed` |
| `copy` | the rows of a donor element the model already has | `donors`, one per added element |

`fresh` is deterministic: the same seed and the same final table give the same
rows. `copy` starts an element from something trained, typically a chemical
neighbour; its isolated-atom energy is still the one given for it, never the
donor's.

## Energies

An added element's isolated-atom energy is a property of each head's level of
theory, so it is given for every head rather than invented. A fine-tune whose
head copies the foundation's energies then reads it from the extended record.

## The record

The extended checkpoint's metadata records the added elements, the spec, its
seed and its donors (`element_extension`). The parent's record is kept whole
under `parents`, unchanged, and the extended model's own head energies are a
new block that says which elements were given explicitly.
