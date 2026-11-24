"""Whether the data carries what a model declares it computes.

An observable that no structure is labelled with trains nothing: its loss term
is absent, and the rest of the loss looks fine. So declaring one is an error
before training starts, not a silent zero.
"""

from __future__ import annotations

from collections.abc import Iterable

from mace_core.data.configuration import Configuration
from mace_core.data.keys import KeySpecification
from mace_core.observables import ObservableCatalogue

__all__ = ["check_observables_labelled"]


def check_observables_labelled(
    catalogue: ObservableCatalogue,
    configurations: Iterable[Configuration],
    keys: KeySpecification,
) -> None:
    """Refuse an observable that no structure carries.

    Args:
        catalogue: What the model declares it computes.
        configurations: The structures it is trained on.
        keys: How their files name each property.

    Raises:
        ValueError: Naming the observable and the file key it is read from,
            if no structure is labelled with it, or if the key specification
            reads no such property at all.
    """
    configurations = list(configurations)
    for spec in catalogue.observables:
        file_key = keys.file_key(spec.name)
        if file_key is None:
            raise ValueError(
                f"the observable {spec.name!r} is declared, and the key "
                f"specification reads no property of that name: it reads "
                f"{sorted(keys.property_names())}. Add a key for it."
            )
        if not any(structure.is_labelled(spec.name) for structure in configurations):
            raise ValueError(
                f"the observable {spec.name!r} is declared, and none of the "
                f"{len(configurations)} structures carries it under the file "
                f"key {file_key!r}. Label the data, or drop the declaration."
            )
