"""Citations printed by train and evaluation, from the artifact's record."""

from mace_core.metadata import Citation, ModelMetadata

MACE_CITATION = Citation(
    title=(
        "MACE: Higher Order Equivariant Message Passing Neural "
        "Networks for Fast and Accurate Force Fields"
    ),
    authors=[
        "Ilyes Batatia",
        "David Peter Kovacs",
        "Gregor N. C. Simm",
        "Christoph Ortner",
        "Gabor Csanyi",
    ],
    year=2022,
    venue="Advances in Neural Information Processing Systems",
    url="https://openreview.net/forum?id=YPpSngE-ZU",
)


def required_citations(metadata: ModelMetadata) -> list[Citation]:
    citations = [MACE_CITATION, *metadata.citations]
    for parent in metadata.parents:
        if parent.metadata is not None:
            citations.extend(required_citations(parent.metadata))
    return list(
        {
            citation.doi or citation.url or citation.title: citation
            for citation in citations
        }.values()
    )
