"""Artifact operations over canonical weights and their recorded configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def select_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_file", nargs="?", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--head", "--head_name", "-n", dest="head")
    parser.add_argument("--list_heads", "-l", action="store_true")
    parser.add_argument("--output", "--output_file", "-o", dest="output", type=Path)
    parser.add_argument(
        "--device", "--target_device", "-d", dest="device", default="cpu"
    )


def convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--to-basis", choices=("full", "reduced"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--head", type=int, default=0)


def _rebuild(source, config, heads):
    from mace_core.data.backend import DatasetStatistics
    from mace_core.elements import ResolvedE0s
    from mace_core.observables import DEFAULT_CATALOGUE

    from mace_torch.models.build import build_model

    return build_model(
        config,
        config.catalogue or DEFAULT_CATALOGUE,
        z_table=source.z_table,
        heads=heads,
        e0s=ResolvedE0s({name: source.e0s[name] for name in heads}),
        statistics=DatasetStatistics(),
        initialize=False,
    )[0]


def select_head(args: argparse.Namespace) -> int:
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.metadata import ConfigRecord, ModelMetadata

    from mace_torch.deploy.loader import load_deployed
    from mace_torch.finetune.transfer import transfer_foundation
    from mace_torch.serialization import write_model

    source_path = args.model or args.model_file
    if source_path is None:
        raise ValueError("give a model path or --model")
    args.model = source_path
    source = load_deployed(source_path)
    if args.head is None or args.list_heads:
        print("Heads: " + ", ".join(source.heads))
        return 0
    if args.head not in source.heads:
        raise ValueError(f"unknown head {args.head!r}; choose from {source.heads}")
    document = source.config.to_resolved_dict()
    document["data"]["heads"] = {args.head: document["data"]["heads"][args.head]}
    document["model"] = source.config.model.model_copy(
        update={
            "normalization": source.config.model.normalization_for_heads(
                {args.head: args.head}
            )
        }
    ).model_dump(mode="json")
    document["data"]["skip_evaluate_heads"] = []
    document["data"]["ratio_guard"] = None
    document["finetune"]["pseudolabels"]["heads"] = []
    document["finetune"]["pseudolabels"]["enabled"] = False
    document["finetune"]["pseudolabels"]["labels_from"] = None
    config = ResolvedConfig.from_dict(document)
    engine = _rebuild(source, config, (args.head,))
    transfer_foundation(
        source.model,
        engine.get_submodule("model"),
        foundation_elements=source.z_table.zs,
        elements=source.z_table.zs,
        foundation_heads=source.heads,
        heads=(args.head,),
        readout_from={args.head: args.head},
    )
    record = source.metadata.model_dump(mode="json")
    record["config"] = ConfigRecord.from_config(config).model_dump(mode="json")
    record["heads"] = {args.head: record["heads"][args.head]}
    if record["training"] is not None:
        record["training"]["validation_errors"] = {
            key: value
            for key, value in record["training"]["validation_errors"].items()
            if key == args.head
        }
    destination = args.output or args.model.with_name(
        f"{args.model.stem}-{args.head}.safetensors"
    )
    if (
        destination.with_suffix(".json").resolve()
        == source_path.with_suffix(".json").resolve()
    ):
        raise ValueError("--output must differ from the source checkpoint")
    print(
        write_model(
            destination, engine.to(args.device), ModelMetadata.model_validate(record)
        )
    )
    return 0


def convert(args: argparse.Namespace) -> int:
    import torch
    from mace_core.clebsch_gordan.conversion import full_to_reduced
    from mace_core.clebsch_gordan.irreps import Irreps
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.kernels.specs import SymmetricContractionSpec
    from mace_core.metadata import BasisConversionRecord, ConfigRecord

    from mace_torch.canonical import canonical_modules
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.deploy.reference import verify_against_reference
    from mace_torch.serialization import (
        canonical_state,
        load_canonical_state,
        write_model,
    )

    if args.to_basis == "full":
        raise ValueError(
            "reduced -> full checkpoint conversion is under-determined; "
            "use the separate full-basis weights export (DEP-3 #1595)"
        )
    if (args.reference is None) != (args.fixtures is None):
        raise ValueError("--reference and --fixtures must be supplied together")
    if (
        args.output.with_suffix(".json").resolve()
        == args.model.with_suffix(".json").resolve()
    ):
        raise ValueError("--output must differ from the source checkpoint")
    source = load_deployed(args.model)
    if source.config.model.clebsch_gordan_basis != "full":
        raise ValueError("the checkpoint is already in the reduced basis")
    document = source.config.to_resolved_dict()
    document["model"]["clebsch_gordan_basis"] = "reduced"
    config = ResolvedConfig.from_dict(document)
    target = _rebuild(source, config, source.heads)
    state = canonical_state(source.engine)
    target_state = canonical_state(target)
    for path, module in canonical_modules(source.engine).items():
        kernel_spec = getattr(module, "kernel_spec", None)
        if not isinstance(kernel_spec, SymmetricContractionSpec):
            continue
        tensors = state[path]
        pieces = iter(
            torch.split(tensors["weight"], tensors["path_counts"].tolist(), dim=1)
        )
        projected = []
        for order in range(1, kernel_spec.correlation + 1):
            for _, irrep in Irreps.parse(kernel_spec.irreps_out):
                weight = next(pieces)
                converted = full_to_reduced(
                    weight.cpu().numpy(), kernel_spec.irreps_in, order, irrep
                )
                projected.append(torch.as_tensor(converted, dtype=weight.dtype))
        state[path] = {
            "weight": torch.cat(projected, dim=1),
            "path_counts": target_state[path]["path_counts"],
        }
    load_canonical_state(target, state)
    report = None
    if args.reference is not None:
        report = verify_against_reference(
            target,
            args.reference,
            args.fixtures,
            z_table=source.z_table,
            cutoff=source.r_max,
            head=args.head,
        )
        if not report.passed:
            raise ValueError(report.describe())
        print(report.describe())
    metadata = source.metadata.model_copy(
        update={
            "config": ConfigRecord.from_config(config),
            "conversion": BasisConversionRecord(
                projection_verified=True,
                reference=str(args.reference) if args.reference else None,
                golden_gate_passed=report.passed if report else None,
            ),
        }
    )
    print(write_model(args.output, target, metadata))
    return 0
