"""Starting a fine-tune's model from a foundation model's weights.

The two models are built from one architecture and differ in two ways: the
fine-tune may hold fewer elements, and it has its own heads. So most of the
canonical state is copied as it stands, the element-indexed tensors are read
at the fine-tune's elements, and each head's readout is copied from the
foundation head it names.

**The rules are explicit, and a tensor no rule covers is an error.** A shape
that differs where nothing here says why is a model built from a different
architecture, and copying the part that happens to fit would give a model
that runs with some weights at their initial values. The frozen tree's
transfer walks named attributes and copies what it recognizes
(``mace/tools/finetuning_utils.py:57-460``), so an attribute it does not
recognize simply keeps its random start.

Three tensors are indexed by element, each along its first axis or its input:
the node embedding, a linear map out of the element one-hot; the skip
connection, one linear map per element; and the symmetric contraction's
weights, ``[Z, A, mul]``. The isolated-atom energies are indexed by element
too, and they are not copied at all: each head's were resolved by the data
stage, from whatever its declaration names, and that is the point of having a
head. The scale and shift are copied from the foundation head a head's readout
comes from, which is what the frozen tree does with ``use_scale`` and
``use_shift`` on.

A magnetic model adds two more. Its moment saturations are one per element,
and its one-body energy of the moment length is one per element and per head,
so it is read at the kept elements and at each head's source head, like a
readout. The frozen tree does not transfer that term at all, and a fine-tune
of a magnetic model starts it from a random draw instead of the trained one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import Tensor, nn

from mace_torch.models import ObservableHead
from mace_torch.serialization import canonical_state, load_canonical_state

__all__ = ["TransferError", "readout_sources", "transfer_foundation"]


class TransferError(ValueError):
    """A foundation model whose weights do not fit the model being built."""


def readout_sources(
    heads: Sequence[str],
    requested: Mapping[str, str | None],
    foundation_heads: Sequence[str],
) -> dict[str, str]:
    """Which foundation head each head's readout starts from.

    A head that names none takes the foundation model's only head. The frozen
    tree has one ``--foundation_head`` for the whole run and copies it into
    every head; naming it per head is the same thing for a run that names one,
    and what lets two heads start from two different ones.

    Raises:
        TransferError: If a head names a head the foundation model lacks, or
            names none while it has several.
    """
    sources: dict[str, str] = {}
    for head in heads:
        wanted = requested.get(head)
        if wanted is None:
            if len(foundation_heads) != 1:
                raise TransferError(
                    f"head {head!r} does not say which foundation head its "
                    f"readout starts from, and the foundation model has "
                    f"{list(foundation_heads)}. Set its `readout_from`."
                )
            wanted = foundation_heads[0]
        if wanted not in foundation_heads:
            raise TransferError(
                f"head {head!r} starts its readout from {wanted!r}, which the "
                f"foundation model does not have. Its heads are "
                f"{list(foundation_heads)}."
            )
        sources[head] = wanted
    return sources


def _element_rows(
    foundation_elements: Sequence[int], elements: Sequence[int]
) -> list[int]:
    position = {int(z): index for index, z in enumerate(foundation_elements)}
    missing = [int(z) for z in elements if int(z) not in position]
    if missing:
        raise TransferError(
            f"elements {missing} are not among the foundation model's "
            f"{list(foundation_elements)}, so they have no weights to take."
        )
    return [position[int(z)] for z in elements]


def _embedding(weight: Tensor, rows: list[int], count: int) -> Tensor:
    """A linear map out of the element one-hot, read at the kept elements.

    Its input copies are the elements and its output copies the channels, and
    the canonical order puts output copies outermost, so each channel's weights
    are one contiguous run over the elements.
    """
    if weight.numel() % count:
        raise TransferError(
            f"the node embedding holds {weight.numel()} weights, which is not a "
            f"whole number per element for {count} elements."
        )
    return weight.reshape(-1, count)[:, rows].reshape(-1).clone()


def transfer_foundation(
    foundation: nn.Module,
    model: nn.Module,
    *,
    foundation_elements: Sequence[int],
    elements: Sequence[int],
    foundation_heads: Sequence[str],
    heads: Sequence[str],
    readout_from: Mapping[str, str],
    transfer_readout: bool = True,
) -> None:
    """Load a foundation model's weights into a fine-tune's model.

    Args:
        foundation: The foundation model, the module holding the canonical
            state (``engine.backbone``).
        model: The fine-tune's model, built from the same architecture.
        foundation_elements: The foundation's element table, in order.
        elements: The fine-tune's, a subset of it, in order.
        foundation_heads: The foundation's heads, in order.
        heads: The fine-tune's heads, in order.
        readout_from: Each head's source head, as :func:`readout_sources`
            resolves it.
        transfer_readout: Copy the readouts. Off, they keep the fresh draw the
            model was built with, and only the backbone and the scale and shift
            come from the foundation.

    Raises:
        TransferError: If the two models hold different operators, or a tensor
            differs in shape where no rule here explains it.
    """
    source = canonical_state(foundation)
    target = canonical_state(model)
    if set(source) != set(target):
        raise TransferError(
            f"the foundation model and the model being built hold different "
            f"operators. Only in the foundation: {sorted(set(source) - set(target))}. "
            f"Only here: {sorted(set(target) - set(source))}. They were built "
            f"from different architectures."
        )
    rows = _element_rows(foundation_elements, elements)
    head_of = {name: position for position, name in enumerate(foundation_heads)}
    state: dict[str, dict[str, Tensor]] = {}
    for path, tensors in target.items():
        found = source[path]
        if ".readouts." in f".{path}.":
            # Per head, below: a readout is copied one head at a time.
            state[path] = dict(tensors)
            continue
        if path.endswith("energy_head"):
            state[path] = {
                "e0_table": tensors["e0_table"],
                "scale": torch.stack(
                    [found["scale"][head_of[readout_from[head]]] for head in heads]
                ),
                "shift": torch.stack(
                    [found["shift"][head_of[readout_from[head]]] for head in heads]
                ),
            }
            continue
        if path.endswith("node_embedding"):
            state[path] = {
                "weight": _embedding(found["weight"], rows, len(foundation_elements)),
                "bias": found["bias"],
            }
        elif path.endswith(".skip"):
            state[path] = {"weight": found["weight"][rows].clone()}
        elif path.rpartition(".")[2] == "moments":
            state[path] = {"saturation": found["saturation"][rows].clone()}
        elif path.rpartition(".")[2] == "one_body":
            sources = [head_of[readout_from[head]] for head in heads]
            state[path] = {
                "coefficients": found["coefficients"][rows][..., sources].clone(),
                "offset": found["offset"][rows][:, sources].clone(),
            }
        elif path.endswith(".contraction"):
            state[path] = {
                name: value[rows].clone() if name == "weight" else value
                for name, value in found.items()
            }
        else:
            state[path] = {name: value.clone() for name, value in found.items()}
        for name, value in state[path].items():
            if value.shape != tensors[name].shape:
                raise TransferError(
                    f"{path}:{name} is {tuple(value.shape)} from the foundation "
                    f"model and {tuple(tensors[name].shape)} here, and no rule "
                    f"here explains the difference."
                )
    load_canonical_state(model, state)

    if not transfer_readout:
        return
    outputs = model.get_submodule("outputs")
    source_outputs = foundation.get_submodule("outputs")
    for name, head in cast(nn.ModuleDict, outputs.get_submodule("heads")).items():
        donor = cast(ObservableHead, source_outputs.get_submodule(f"heads.{name}"))
        receiver = cast(ObservableHead, head)
        with torch.no_grad():
            for position, fine_tune_head in enumerate(heads):
                receiver.load_head_state(
                    position, donor.head_state(head_of[readout_from[fine_tune_head]])
                )
