"""Export contraction weights for consumers of the full coupling basis.

This derived artifact has its own format identity. It contains no model loader
configuration and cannot resume training or be read as a V1 checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mace_core.clebsch_gordan import (
    Irreps,
    full_path_labels,
    full_symmetric_tensor_product_basis,
    reduced_to_full,
)
from mace_core.kernels.descriptors import SymmetricContractionDescriptor
from safetensors.torch import save_file

from mace_torch.canonical import canonical_modules
from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.deploy.manifest import DeployManifest
from mace_torch.serialization import canonical_state

FORMAT = "mace-full-basis-weights"
SCHEMA_VERSION = "1.0"


def export_full_basis_weights(
    bundle: str | Path, output: str | Path | None = None
) -> Path:
    """Write one minimum-norm ``[Z, A, mul]`` tensor per contraction.

    Paths are grouped by increasing body order, then output declaration order.
    Each group names every coupling tree and its normalization. ``Z`` follows
    the checkpoint's species order, or has one shared row as explicitly recorded.
    The result is the JSON sidecar path; neither output file may already exist.
    """
    source = Path(bundle)
    base = (
        Path(output)
        if output is not None
        else source.with_name(f"{source.stem}-full-basis")
    )
    if base.suffix and base.suffix not in {".json", ".safetensors"}:
        raise DeployError(
            "full-basis output must have no suffix, .json or .safetensors"
        )
    sidecar, tensors = base.with_suffix(".json"), base.with_suffix(".safetensors")
    for path in (sidecar, tensors):
        if path.exists():
            raise FileExistsError(f"full-basis output already exists: {path}")
    checkpoint = source
    if source.is_dir():
        DeployManifest.read(source / "deploy.json")
        checkpoint = source / "model.json"
    # Export the stored weights, independent of a bundle's runtime dtype/backend.
    deployed = load_deployed(checkpoint, backend="reference")
    canonical = canonical_state(deployed.derivative_engine)
    weights = {}
    blocks = []
    for identity, module in canonical_modules(deployed.derivative_engine).items():
        descriptor = getattr(module, "descriptor", None)
        if not isinstance(descriptor, SymmetricContractionDescriptor):
            continue
        if descriptor.basis != "reduced":
            raise DeployError(
                "full-basis export requires reduced weights; use "
                "mace model convert --to-basis reduced for a full-basis checkpoint"
            )
        state = canonical[identity]
        parts = iter(torch.split(state["weight"], state["path_counts"].tolist(), dim=1))
        expanded, groups, offset = [], [], 0
        for order in range(1, descriptor.correlation + 1):
            for _, irrep in Irreps.parse(descriptor.irreps_out):
                part = next(parts).detach().cpu().numpy()
                full = reduced_to_full(part, descriptor.irreps_in, order, irrep)
                expanded.append(torch.from_numpy(full))
                target = str(irrep)
                labels = full_path_labels(descriptor.irreps_in, order, target)[target]
                basis = full_symmetric_tensor_product_basis(
                    descriptor.irreps_in, order, target
                )[target]
                groups.append(
                    {
                        "order": order,
                        "target": target,
                        "offset": offset,
                        "count": len(labels),
                        "full_path_labels": [str(label) for label in labels],
                        "basis_frobenius_norms": [
                            float(np.linalg.norm(path)) for path in basis
                        ],
                        "forward_factors": [1.0] * len(labels),
                    }
                )
                offset += len(labels)
        if descriptor.num_elements not in {1, len(deployed.z_table.zs)}:
            raise DeployError(f"{identity}: contraction has an unknown element layout")
        key = f"blocks.{len(blocks)}"
        value = torch.cat(expanded, dim=1).contiguous()
        weights[key] = value
        blocks.append(
            {
                "tensor": key,
                "component": identity,
                "irreps_in": descriptor.irreps_in,
                "irreps_out": descriptor.irreps_out,
                "correlation": descriptor.correlation,
                "num_features": descriptor.num_features,
                "element_rows": (
                    "shared"
                    if descriptor.num_elements == 1 and len(deployed.z_table.zs) > 1
                    else "species_order"
                ),
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "groups": groups,
            }
        )
    if not blocks:
        raise DeployError("model has no symmetric contraction weights to export")
    document = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "export_only": True,
        "representative": "minimum_euclidean_norm_per_element_and_channel",
        "layout": "Z,A,mul",
        "species": list(deployed.z_table.zs),
        "normalization": {
            "basis": "mace-core full coupling trees",
            "construction": (
                "Identity at order 1. Each later step contracts the previous "
                "output with a real unit-Frobenius Wigner 3j tensor in "
                "(output, previous, input) order. No symmetrization or "
                "additional path normalization."
            ),
            "components": "e3nn real spherical components, m=-l..l",
            "wigner_phase": "i**(l_output+l_previous+l_input)",
            "real_basis_change": {
                "negative_m": "i/sqrt(2) * (Y_l^m - (-1)^m * Y_l^-m)",
                "zero_m": "Y_l^0",
                "positive_m": "1/sqrt(2) * (Y_l^-m + (-1)^m * Y_l^m)",
                "complex_basis": "Condon-Shortley",
            },
            "reduced_source": "unit Frobenius norm, first nonzero component positive",
            "weight_projection": (
                "float64 pseudoinverse of the full-to-reduced symmetric projection, "
                "cast to the source weight dtype"
            ),
        },
        "blocks": blocks,
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=sidecar.parent) as staging:
        staged = Path(staging)
        save_file(
            weights, str(staged / "weights.safetensors"), metadata={"format": FORMAT}
        )
        (staged / "weights.json").write_text(json.dumps(document, indent=2) + "\n")
        # Publish the sidecar last: it is the record that the pair is complete.
        (staged / "weights.safetensors").rename(tensors)
        (staged / "weights.json").rename(sidecar)
    return sidecar
