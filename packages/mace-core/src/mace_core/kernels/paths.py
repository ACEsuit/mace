"""Which couplings a message-passing convolution keeps, and in what order.

A channelwise convolution couples every node irrep with every edge irrep and
keeps whatever the selection rules allow that the layer asked for. Several of
those couplings land on the **same** output irrep, and they are kept apart
rather than summed: the linear map after the convolution mixes them with
independent weights, so one coupling can go where another does not. Summing
them first is a smaller model with the same declarations.

The order is part of the weight format. The radial network produces one weight
per path and channel, and the linear that follows reads the paths in this
order, so another order is another checkpoint. The rule is the frozen tree's:
generate in ``(node term, edge term, output irrep)`` order, keep what the
target declares, then sort by degree and parity, odd first, with a **stable**
sort, so that couplings landing on one irrep stay in the order they were
generated in.
"""

from __future__ import annotations

from dataclasses import dataclass

from mace_core.clebsch_gordan.irreps import Irrep, Irreps

__all__ = ["ConvolutionPath", "channelwise_paths", "path_irreps"]


@dataclass(frozen=True)
class ConvolutionPath:
    """One coupling a channelwise convolution keeps.

    Attributes:
        node_term: Index of the node irreps term it reads.
        edge_term: Index of the edge irreps term it reads.
        irrep: The output irrep it writes.
        multiplicity: The node term's multiplicity, which the path carries
            through unchanged. That is what channelwise means.
    """

    node_term: int
    edge_term: int
    irrep: Irrep
    multiplicity: int


def channelwise_paths(
    irreps_node: str, irreps_edge: str, irreps_target: str
) -> tuple[ConvolutionPath, ...]:
    """Every path the convolution keeps, in the order its weights are stored.

    Args:
        irreps_node: One channel's sender node features.
        irreps_edge: The edge attributes, normally spherical harmonics.
        irreps_target: Which output irreps the layer wants. A coupling landing
            outside it is dropped rather than computed and discarded.
    """
    node = Irreps.parse(irreps_node)
    edge = Irreps.parse(irreps_edge)
    wanted = {irrep for _, irrep in Irreps.parse(irreps_target).terms}
    found = [
        ConvolutionPath(node_index, edge_index, coupled, multiplicity)
        for node_index, (multiplicity, node_irrep) in enumerate(node.terms)
        for edge_index, (_, edge_irrep) in enumerate(edge.terms)
        for coupled in node_irrep.couple(edge_irrep)
        if coupled in wanted
    ]
    # Stable, so that paths onto one irrep keep their generation order, which
    # is the order the stored weights are written against. Odd before even at
    # one degree, the frozen tree's order, which is not `Irrep`'s.
    return tuple(sorted(found, key=lambda path: (path.irrep.degree, path.irrep.parity)))


def path_irreps(paths: tuple[ConvolutionPath, ...]) -> str:
    """One channel's message declaration: one term per path, in path order.

    Repeated irreps stay separate terms on purpose. Merging them would lose
    which weight belongs to which coupling.
    """
    return "+".join(f"{path.multiplicity}x{path.irrep}" for path in paths)
