"""Resolve model inputs and dataset keys for training and inference."""

import dataclasses

from mace_core.config.data import DataConfig
from mace_core.config.model import ModelConfig
from mace_core.data import KeySpecification
from mace_core.observables import RequestedOutputs

from mace_torch.data.graphs import target_specs
from mace_torch.models.registry import DeclaresModelInputs, registered_models

__all__ = ["graph_inputs_of", "key_specification"]


def graph_inputs_of(model: ModelConfig) -> tuple[str, ...]:
    """The inputs a model reads, per structure or per atom.

    The charge-aware, dipole and magnetic models', and the ones declared as
    embedded features. Written into every graph built for it, from the file or
    from their defaults.
    """
    inputs: tuple[str, ...] = ()
    registered = registered_models().get(model.model)
    if isinstance(registered, DeclaresModelInputs):
        inputs = tuple(spec.name for spec in registered.declared_inputs(model))
    elif model.model == "les":
        inputs = ("external_field",)
    elif model.model == "polar":
        from mace_torch.models.electrostatics import POLAR_GRAPH_INPUTS

        inputs = POLAR_GRAPH_INPUTS
    elif model.model in ("dipole", "dielectric"):
        from mace_torch.models.dipoles import DIPOLE_GRAPH_INPUTS

        inputs = DIPOLE_GRAPH_INPUTS[
            "fixed" if model.model == "dipole" else "predicted"
        ]
    elif model.model == "magnetic":
        from mace_torch.models.magnetic import MAGNETIC_GRAPH_INPUTS

        inputs = MAGNETIC_GRAPH_INPUTS
    elif "dipole" in model.observables:
        # Declared joint energy/dipole readouts need the same fixed charges as
        # the dipole-only model, through every data and deployment consumer.
        inputs = ("charges",)
    return inputs + tuple(
        feature.name for feature in model.graph_features if feature.name not in inputs
    )


def key_specification(
    data: DataConfig,
    requested: RequestedOutputs | None = None,
    *,
    model: ModelConfig | None = None,
) -> KeySpecification:
    """The default property keys, plus the graph inputs this run renamed."""
    spec = KeySpecification.from_defaults()
    embedding_specs = dict(data.embedding_specs)
    if model is not None:
        for name, storage in model.feature_storages.items():
            declared = embedding_specs.setdefault(name, {"per": storage})
            if declared.get("per") != storage:
                raise ValueError(
                    f"data.embedding_specs.{name}.per disagrees with "
                    "model.graph_features"
                )
    spec.add_embedding_features(embedding_specs)
    if model is not None:
        registered = registered_models().get(model.model)
        if isinstance(registered, DeclaresModelInputs):
            for item in registered.declared_inputs(model):
                keys = spec.atom_keys if item.per_atom else spec.graph_keys
                keys.setdefault(item.name, item.name)
    if requested is not None:
        # Existing labels retain their agreed keys. New properties are read
        # by their declared name, in the declared per-atom/per-graph storage.
        known = set(spec.graph_keys) | set(spec.atom_keys)
        custom = [
            target for target in target_specs(requested) if target.name not in known
        ]
        spec.update(
            graph_keys={
                target.name: target.name for target in custom if not target.per_atom
            },
            atom_keys={
                target.name: target.name for target in custom if target.per_atom
            },
        )
        if custom:
            spec = dataclasses.replace(
                spec,
                label_names=tuple(target.name for target in target_specs(requested)),
            )
    keys = data.graph_input_keys
    return spec.update(
        graph_keys={
            "elec_temp": keys.elec_temp,
            "total_spin": keys.total_spin,
            "total_charge": keys.total_charge,
            "external_field": keys.external_field,
        }
    )
