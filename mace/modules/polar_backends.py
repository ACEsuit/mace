"""Optional electrostatics backends used by PolarMACE."""

from functools import lru_cache
from typing import Any, Callable, Optional, Tuple

import torch

from mace.tools.scatter import scatter_sum

GRAPH_LONGRANGE_BACKEND = "graph_longrange"
NVALCHEMIOPS_BACKEND = "nvalchemiops"
SUPPORTED_POLAR_ELECTROSTATICS_BACKENDS = (
    GRAPH_LONGRANGE_BACKEND,
    NVALCHEMIOPS_BACKEND,
)
_NVALCHEMIOPS_DTYPE = torch.float64


def normalize_electrostatics_backend(backend: str) -> str:
    """Validate and normalize a PolarMACE electrostatics backend name."""
    normalized = str(backend).lower().replace("-", "").replace("_", "")
    aliases = {
        "graphlongrange": GRAPH_LONGRANGE_BACKEND,
        "nvalchemiops": NVALCHEMIOPS_BACKEND,
    }
    if normalized not in aliases:
        choices = ", ".join(SUPPORTED_POLAR_ELECTROSTATICS_BACKENDS)
        raise ValueError(
            f"Unknown PolarMACE electrostatics backend {backend!r}. "
            f"Choose one of: {choices}."
        )
    return aliases[normalized]


@lru_cache(maxsize=1)
def _load_nvalchemiops() -> Tuple[Callable, Callable, Callable]:
    try:
        from nvalchemiops.torch.interactions.electrostatics import (
            multipole_scf_step_energy,
            multipole_scf_step_features,
            prepare_multipole_scf_cache,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "The PolarMACE nvalchemiops backend requires "
            "Python>=3.11, PyTorch>=2.8, and "
            "nvalchemi-toolkit-ops>=0.4.0. Install it with "
            "`pip install 'mace-torch[nvalchemiops]'`."
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            "nvalchemiops could not initialize its Warp runtime. Check the Warp "
            "cache permissions and that the toolkit supports this device and CUDA "
            "driver."
        ) from exc
    return (
        prepare_multipole_scf_cache,
        multipole_scf_step_features,
        multipole_scf_step_energy,
    )


def require_nvalchemiops() -> None:
    """Fail early when the optional NVIDIA backend cannot be imported."""
    _load_nvalchemiops()


def prepare_nvalchemiops_scf_cache(
    cell: torch.Tensor,
    batch: torch.Tensor,
    *,
    density_smearing_width: float,
    feature_smearing_widths: Tuple[float, ...],
    kspace_cutoff: float,
    density_max_l: int,
    feature_max_l: int,
) -> Tuple[Any, Optional[torch.Tensor]]:
    """Build one NVIDIA cache for all PolarMACE fixed-point evaluations."""
    prepare_cache, _, _ = _load_nvalchemiops()
    # Toolkit-Ops 0.4.0 multipole kernels use Warp float64 vectors.
    cells = cell.reshape(-1, 3, 3).to(dtype=_NVALCHEMIOPS_DTYPE)
    if cells.shape[0] == 1:
        cache_cell = cells[0]
        batch_idx = None
    else:
        cache_cell = cells
        batch_idx = batch.to(dtype=torch.int32).contiguous()

    cache = prepare_cache(
        cache_cell,
        sigma=float(density_smearing_width),
        receiver_sigmas=feature_smearing_widths,
        k_cutoff=float(kspace_cutoff),
        l_max=int(density_max_l),
        feature_max_l=int(feature_max_l),
        density_normalize="multipoles",
        feature_normalize="receiver",
        device=cell.device,
    )
    return cache, batch_idx


def nvalchemiops_scf_features(
    cache: Any,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    *,
    batch_idx: Optional[torch.Tensor],
    include_self_interaction: bool,
) -> torch.Tensor:
    """Evaluate NVIDIA multipole features with the graph-longrange layout."""
    _, step_features, _ = _load_nvalchemiops()
    features = step_features(
        cache,
        positions.to(dtype=_NVALCHEMIOPS_DTYPE).contiguous(),
        source_feats.to(dtype=_NVALCHEMIOPS_DTYPE).contiguous(),
        batch_idx=batch_idx,
        include_self_interaction=include_self_interaction,
    )
    return features.to(dtype=source_feats.dtype)


def nvalchemiops_scf_energy(
    cache: Any,
    positions: torch.Tensor,
    source_feats: torch.Tensor,
    *,
    batch_idx: Optional[torch.Tensor],
    batch: torch.Tensor,
    num_graphs: int,
    include_self_interaction: bool,
) -> torch.Tensor:
    """Evaluate and reduce NVIDIA per-atom energies to MACE graph energies."""
    _, _, step_energy = _load_nvalchemiops()
    node_energy = step_energy(
        cache,
        positions.to(dtype=_NVALCHEMIOPS_DTYPE).contiguous(),
        source_feats.to(dtype=_NVALCHEMIOPS_DTYPE).contiguous(),
        batch_idx=batch_idx,
        include_self_interaction=include_self_interaction,
    )
    energy = scatter_sum(
        src=node_energy,
        index=batch,
        dim=0,
        dim_size=num_graphs,
    )
    return energy.to(dtype=source_feats.dtype)
