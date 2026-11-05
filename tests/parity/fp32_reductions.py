"""Fixed-order scalar-channel reductions for PAR-2's fp32 golden.

Dyadic inputs and scalar (0e) couplings avoid platform-dependent transcendental
or BLAS arithmetic: the golden isolates accumulation order. Tensor-product
physics and full-model E/F/stress use the ordinary tolerance rows elsewhere.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    SegmentReduceDescriptor,
)
from mace_torch.backends.reference import ReferenceBackend

SEED = 20261105


def fixture():
    rng = np.random.default_rng(SEED)
    values = (
        rng.integers(-8, 9, (128, 2)) * np.exp2(rng.integers(-12, 13, (128, 2)))
    ).astype(np.float32)
    # A guaranteed order-sensitive segment, separate from the random inputs.
    values = np.concatenate(
        [values, [[-(2**24), 2**24], [2**24, -(2**24)], [1, 1]]]
    ).astype(np.float32)
    receiver = np.concatenate([rng.integers(0, 3, 128), [3, 3, 3]])
    sender = rng.integers(0, 3, len(values))
    sender[-3:] = 0
    return {
        "seed": SEED,
        "num_nodes": 5,  # includes a node without incoming messages
        "values": values.tolist(),
        "sender": sender.tolist(),
        "receiver": receiver.tolist(),
        "node_features": [[1, 1], [2, -1], [0.5, 4], [1, 1], [1, 1]],
        "edge_attributes": np.ones((len(values), 1)).tolist(),
    }


def evaluate(inputs, *, reverse=False):
    """Run the public reference factories, with an optional order perturbation."""
    values = torch.tensor(inputs["values"], dtype=torch.float32)
    receiver = torch.tensor(inputs["receiver"], dtype=torch.int64)
    sender = torch.tensor(inputs["sender"], dtype=torch.int64)
    nodes = torch.tensor(inputs["node_features"], dtype=torch.float32)
    edges = torch.tensor(inputs["edge_attributes"], dtype=torch.float32)
    if reverse:
        values, receiver, sender, edges = (
            item.flip(0) for item in (values, receiver, sender, edges)
        )
    backend = ReferenceBackend()
    reduce = backend.make_segment_reduce(SegmentReduceDescriptor(precision="float32"))
    convolution = backend.make_channelwise_tp_conv(
        ChannelwiseTPConvDescriptor(
            precision="float32",
            irreps_node="0e",
            irreps_edge="0e",
            irreps_out="0e",
            num_features=2,
        )
    )
    return {
        "segment_reduce": reduce(values, receiver, inputs["num_nodes"]),
        "channelwise_tp_conv": convolution(
            nodes,
            edges,
            values[:, None, :],
            sender,
            receiver,
            inputs["num_nodes"],
        ),
    }


def sequential_reference(inputs):
    """Independent IEEE fp32 left fold, including the scalar tensor product."""
    values = np.asarray(inputs["values"], dtype=np.float32)
    nodes = np.asarray(inputs["node_features"], dtype=np.float32)
    edges = np.asarray(inputs["edge_attributes"], dtype=np.float32)
    reduced = np.zeros((inputs["num_nodes"], 2), dtype=np.float32)
    convolved = reduced.copy()
    for index, (sender, receiver) in enumerate(
        zip(inputs["sender"], inputs["receiver"], strict=True)
    ):
        reduced[receiver] += values[index]
        convolved[receiver] += nodes[sender] * edges[index] * values[index]
    return {"segment_reduce": reduced, "channelwise_tp_conv": convolved}


def assert_golden(outputs, expected):
    """PAR-2's explicit max-abs gate, not a new general physics tolerance."""
    epsilon = np.finfo(np.float32).eps
    for name in ("segment_reduce", "channelwise_tp_conv"):
        actual = outputs[name].detach().numpy()
        reference = np.asarray(expected[name], dtype=np.float32)
        assert actual.dtype == np.float32 and actual.shape == reference.shape, name
        error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        assert error <= epsilon, (
            f"{name}: max abs error {error} exceeds fp32 epsilon {epsilon}"
        )
