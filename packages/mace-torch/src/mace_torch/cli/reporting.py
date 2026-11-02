"""Print the citations required by an evaluated artifact and its parents."""

from mace_core.citations import required_citations
from mace_core.metadata import ModelMetadata, format_citations


def print_citations(metadata: ModelMetadata) -> None:
    print(format_citations(required_citations(metadata)))
