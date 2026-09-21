"""That the rewrite adds up a segment in the same order the frozen tree does.

Floating-point addition is not associative, so two reductions over the same
numbers in different orders differ in the last bits. At fp64 on a tiny fixture
that is invisible, which is exactly the problem: the difference is real, it
does not show up where anyone is looking, and it surfaces at fp32 or at scale
as a discrepancy nobody can place.

The frozen tree reduces atoms onto graphs and edges onto atoms with a vendored
scatter, excluded from its own lint and coverage precisely because it is
vendored. The rewrite has its own. They have to agree bit for bit on CPU, and
this is where that is checked rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mace_torch.kernels import segment_sum


def legacy_scatter(values, index, segments):
    from mace.tools.scatter import scatter_sum

    return scatter_sum(src=values, index=index, dim=0, dim_size=segments)


@pytest.mark.parametrize("width", [1, 3])
def test_the_two_reductions_agree_bit_for_bit(fp64, isolated, width):
    """Many edges onto few nodes, which is where the order shows.

    The values are deliberately spread over several orders of magnitude: a sum
    of similar numbers is order-insensitive and would pass whatever the two
    implementations did.
    """
    generator = torch.Generator().manual_seed(4)
    count, segments = 4096, 7
    exponents = torch.randint(-8, 8, (count,), generator=generator).to(torch.float64)
    values = (
        torch.randn(count, width, generator=generator, dtype=torch.float64)
        * (10.0**exponents)[:, None]
    )
    index = torch.randint(0, segments, (count,), generator=generator)

    theirs = legacy_scatter(values, index, segments)
    ours = segment_sum(values, index, segments)

    assert torch.equal(theirs, ours), (
        f"the two reductions differ by "
        f"{float((theirs - ours).abs().max()):.3e}, which at these magnitudes "
        f"is an accumulation-order difference rather than a wrong answer"
    )


def test_the_case_is_hard_enough_to_see_an_order_difference(fp64):
    """Self-test: the same numbers summed in another order really do differ.

    Without this the comparison above would pass just as happily on inputs
    where every order gives the same bits, and would be pinning nothing.
    """
    generator = torch.Generator().manual_seed(4)
    count = 4096
    exponents = torch.randint(-8, 8, (count,), generator=generator).to(torch.float64)
    values = (
        torch.randn(count, generator=generator, dtype=torch.float64) * 10.0**exponents
    )
    forwards = torch.zeros((), dtype=torch.float64)
    for value in values:
        forwards = forwards + value
    backwards = torch.zeros((), dtype=torch.float64)
    for value in reversed(values):
        backwards = backwards + value

    assert not torch.equal(forwards, backwards), (
        "summing these numbers in both directions gave the same bits, so the "
        "comparison above cannot see an ordering difference"
    )


def test_an_empty_segment_is_zero_in_both(fp64, isolated):
    """A node with no edges. Both have to write a zero rather than skip it."""
    values = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    index = torch.tensor([0, 0])
    theirs = legacy_scatter(values, index, 3)
    ours = segment_sum(values, index, 3)
    assert torch.equal(theirs, ours)
    assert float(ours[2]) == 0.0


def test_the_order_matches_on_the_shape_a_model_actually_reduces(fp64, isolated):
    """Site energies onto one graph, which is the reduction the energy takes.

    Separately from the generic case, because this is the one whose ordering
    the two-reduction structure of the energy head depends on.
    """
    generator = np.random.default_rng(11)
    site = torch.tensor(generator.normal(size=(512,)) * 1e-3, dtype=torch.float64)
    isolated_atom = torch.full((512,), -2040.0, dtype=torch.float64)
    index = torch.zeros(512, dtype=torch.long)

    for values in (site, isolated_atom, site + isolated_atom):
        assert torch.equal(
            legacy_scatter(values, index, 1), segment_sum(values, index, 1)
        )
