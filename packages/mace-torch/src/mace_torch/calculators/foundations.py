"""Native foundation loaders, sharing identities and downloads with JAX.

The compatibility ``model_type`` spellings are input tokens only. The model's
configuration determines its architecture; no legacy class is imported or built.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from ase import units
from ase.calculators.mixing import SumCalculator
from mace_core.artifacts import ArtifactRef, resolve_checkpoint
from mace_core.config.precision import PrecisionConfig
from mace_core.metadata import ConfigRecord, ModelMetadata
from mace_core.model_registry import (
    Family,
    family_default_dtype,
    published_reference,
    resolve_model,
)

from mace_torch.calculators.ase_calculator import MACECalculator
from mace_torch.deploy.loader import DeployedModel, load_deployed
from mace_torch.serialization import read_sidecar

# Accepted boundary spellings, never a second implementation selector.
_MODEL_TYPES = {
    "mace_mp": "MACE",
    "mace_off": "MACE",
    "mace_omol": "MACE",
    "mace_mdp": "DipolePolarizabilityMACE",
    "mace_polar": "PolarMACE",
}


def load_foundation(
    model: str | Path | None,
    *,
    family: Family = "mace_mp",
    device: str = "",
    default_dtype: str | None = None,
) -> DeployedModel:
    """Load a native checkpoint or registry identity as a structured model.

    Canonical names require published native URLs. Local paths and explicit
    HTTPS URLs allow conversion candidates and fine-tuned descendants; those
    inputs acquire no registry support claim. Metadata versions are checked by
    the shared schema before constructing the model.
    """
    entry = None
    reference: str | Path | ArtifactRef
    if isinstance(model, Path) or (
        isinstance(model, str)
        and (model.startswith("https:") or Path(model).suffix or "/" in model)
    ):
        reference = model
    else:
        entry = resolve_model(model, family=family)
        reference = published_reference(entry)
    dtype = default_dtype or family_default_dtype(family)
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"default_dtype={dtype!r}; use float32 or float64")
    if entry is not None and (notice := entry.license.notice()) is not None:
        print(notice)
    files = resolve_checkpoint(reference)
    document = read_sidecar(files.metadata)
    ModelMetadata.from_json(json.dumps(document["config"]))
    loaded = load_deployed(
        files.metadata,
        tensors_path=files.tensors,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        precision=PrecisionConfig(model=dtype),
    )
    expected = {"mace_mdp": {"dielectric"}, "mace_polar": {"polar"}}.get(
        family, {"plain", "scale_shift"}
    )
    if loaded.config.model.model not in expected:
        raise ValueError(
            f"{family} requires {sorted(expected)}; checkpoint declares "
            f"{loaded.config.model.model!r}"
        )
    config = loaded.config.model_copy(
        update={"precision": PrecisionConfig(model=dtype)}
    )
    loaded = replace(
        loaded,
        config=config,
        metadata=loaded.metadata.model_copy(
            update={
                "config": ConfigRecord(
                    user=loaded.metadata.config.user, resolved=config.to_resolved_dict()
                ),
            }
        ),
    )
    if entry is not None:
        if entry.hyperparameters() != loaded.config.model:
            raise ValueError(
                f"{entry.model_id}: published architecture disagrees "
                f"with its registry record"
            )
        if tuple(entry.metadata.elements) != tuple(loaded.z_table.zs):
            raise ValueError(
                f"{entry.model_id}: published elements disagree "
                "with its registry record"
            )
        if tuple(entry.metadata.heads) != loaded.heads:
            raise ValueError(
                f"{entry.model_id}: published heads disagree with its registry record"
            )
        # Preserve the artifact's training/conversion history; enrich citation
        # discovery from the registry without editing the cached sidecar.
        loaded = replace(
            loaded,
            metadata=loaded.metadata.model_copy(
                update={
                    "doi": entry.metadata.doi or loaded.metadata.doi,
                    "citations": list(
                        {
                            citation.doi or citation.url or citation.title: citation
                            for citation in [
                                *loaded.metadata.citations,
                                *entry.metadata.citations,
                            ]
                        }.values()
                    ),
                }
            ),
        )
    return loaded


def _calculator(
    family: Family,
    model: str | Path | None,
    device: str,
    default_dtype: str | None,
    return_raw_model: bool,
    kwargs: dict[str, Any],
) -> MACECalculator | torch.nn.Module:
    expected = _MODEL_TYPES[family]
    if kwargs.pop("model_type", expected) != expected:
        raise ValueError(f"{family} only supports model_type={expected!r}")
    if family == "mace_mdp":
        warnings.warn(
            "The MACE-MDP model predicts dipoles and polarizabilities "
            "of organic systems only; it does not provide energies or forces.",
            UserWarning,
            stacklevel=3,
        )
    if family == "mace_omol":
        if kwargs.pop("head", "omol") != "omol":
            raise ValueError("mace_omol requires head='omol'")
        kwargs["head"] = "omol"
    loaded = load_foundation(
        model, family=family, device=device, default_dtype=default_dtype
    )
    if return_raw_model:
        return loaded.engine
    return MACECalculator(
        models=loaded,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
    )


def mace_mp(
    model: str | Path | None = None,
    device: str = "",
    default_dtype: str = "float32",
    dispersion: bool = False,
    dispersion_damping: str = "bj",
    dispersion_xc: str = "pbe",
    dispersion_cutoff: float = 40.0 * units.Bohr,
    dispersion_coord_cutoff: float = 40.0 * units.Bohr,
    return_raw_model: bool = False,
    **kwargs: Any,
) -> MACECalculator | torch.nn.Module | SumCalculator:
    """Materials foundation calculator; D3 distances are in Angstrom."""
    damping = kwargs.pop("damping", None)
    if damping is not None:
        dispersion_damping = damping
    calculator = _calculator(
        "mace_mp", model, device, default_dtype, return_raw_model, dict(kwargs)
    )
    if return_raw_model or not dispersion:
        return calculator
    try:
        from torch_dftd.torch_dftd3_calculator import (  # ty: ignore[unresolved-import]
            TorchDFTD3Calculator,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install torch-dftd to use dispersion corrections"
        ) from error
    correction = TorchDFTD3Calculator(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        damping=dispersion_damping,
        dtype=getattr(torch, default_dtype),
        xc=dispersion_xc,
        cutoff=dispersion_cutoff,
        cnthr=dispersion_coord_cutoff,
        **kwargs,
    )
    return SumCalculator([calculator, correction])


def mace_off(
    model: str | Path | None = None,
    device: str = "",
    default_dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs: Any,
) -> MACECalculator | torch.nn.Module:
    """OFF23 organic foundation calculator, with the legacy fp64 default."""
    return _calculator(
        "mace_off", model, device, default_dtype, return_raw_model, kwargs
    )


def mace_omol(
    model: str | Path | None = None,
    device: str = "",
    default_dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs: Any,
) -> MACECalculator | torch.nn.Module:
    """OMOL molecular calculator; the omol head is required."""
    return _calculator(
        "mace_omol", model, device, default_dtype, return_raw_model, kwargs
    )


def mace_mdp(
    model: str | Path | None = None,
    device: str = "",
    default_dtype: str = "float64",
    return_raw_model: bool = False,
    **kwargs: Any,
) -> MACECalculator | torch.nn.Module:
    """MDP dipole/polarizability calculator, without energy or forces."""
    return _calculator(
        "mace_mdp", model, device, default_dtype, return_raw_model, kwargs
    )


def mace_polar(
    model: str | Path,
    device: str = "",
    default_dtype: str = "float32",
    return_raw_model: bool = False,
    **kwargs: Any,
) -> MACECalculator | torch.nn.Module:
    """Polar calculator; select a model explicitly, with the legacy fp32 default."""
    if model is None:
        raise ValueError("mace_polar requires an explicit model")
    return _calculator(
        "mace_polar", model, device, default_dtype, return_raw_model, kwargs
    )
