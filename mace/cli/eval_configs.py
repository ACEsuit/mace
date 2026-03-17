#!/usr/bin/env python
###########################################################################################
# Script for multi-GPU evaluation (minimal overhead, robust)
# - Prefer TorchScript compiled model if available (fast per-rank load)
# - Otherwise load model metadata (rank 0), broadcast metadata to ranks
# - Each rank loads/convert model locally (no NCCL broadcast of blobs)
# - Each rank writes per-rank extxyz, rank 0 merges them
#
# Notes:
#  * This script avoids DDP for inference (not needed)
#  * Safe-loading guards for PyTorch 2.6 unpickle changes are included
###########################################################################################

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import ase.io
import numpy as np
import torch

# Allow torch's safe unpickler to accept `slice` (needed for some e3nn pickles)
try:
    from torch.serialization import add_safe_globals

    add_safe_globals([slice])
except Exception as exc:  # pylint: disable=broad-exception-caught
    # Older PyTorch may not expose add_safe_globals; continue and rely on safe_torch_load
    pass

from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument(
        "--model",
        help="path to model (e3nn .model or torchscript compiled)",
        required=True,
    )
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress", help="compute stress", action="store_true", default=False
    )
    parser.add_argument(
        "--compute_bec", help="compute BEC for LES", action="store_true", default=False
    )
    parser.add_argument(
        "--compute_polarization",
        help="compute polarization for MACEField",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_becs",
        help="compute becs for MACEField",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_polarizability",
        help="compute polarizability for MACEField",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--electric_field_key",
        help="Key for the electric field",
        type=str,
        required=False,
        default="REF_electric_field",
    )
    parser.add_argument(
        "--electric-field",
        help="Uniform electric field components in a.u.",
        type=float,
        nargs=3,
        metavar=("Ex", "Ey", "Ez"),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_descriptors",
        help="model outputs MACE descriptors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--descriptor_num_layers",
        help="number of layers to take descriptors from",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--descriptor_aggregation_method",
        help="method for aggregating node features",
        choices=["mean", "per_element_mean", None],
        default=None,
    )
    parser.add_argument(
        "--descriptor_invariants_only",
        help="save invariant (l=0) descriptors only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--return_node_energies",
        help="model outputs MACE node energies",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def safe_torch_load(path_or_buffer, map_location):
    """
    Helper wrapper to call torch.load trying weights_only=False when supported (PyTorch>=2.6).
    """
    try:
        return torch.load(path_or_buffer, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path_or_buffer, map_location=map_location)


def init_distributed_if_needed(args_device: str):
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not is_distributed:
        dev = torch.device(
            "cuda:0" if args_device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        return False, 0, 1, 0, dev

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if args_device == "cuda" and torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if args_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return True, rank, world_size, local_rank, device


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
    compute_polarization: bool,
    compute_becs: bool,
    compute_polarizability: bool,
    electric_field: Optional[torch.Tensor] = None,
):
    forward_args = {"compute_stress": compute_stress}
    if compute_bec:
        forward_args["compute_bec"] = compute_bec
    if compute_polarization:
        forward_args["compute_polarization"] = compute_polarization
        forward_args["training"] = True
    if compute_becs:
        forward_args["compute_becs"] = compute_becs
        forward_args["training"] = True
    if compute_polarizability:
        forward_args["compute_polarizability"] = compute_polarizability
        forward_args["training"] = True
    if electric_field is not None:
        forward_args["electric_field"] = electric_field
    return model(batch, **forward_args)


def try_load_compiled_candidate(model_path: Path, device: torch.device):
    """
    Try candidate compiled filenames next to given model_path:
      - <stem>_compiled.model
      - <stem>_stagetwo_compiled.model
      - the provided path itself
    Returns the ScriptModule if successful, else None.
    """
    candidates = []
    # if user passed a compiled model directly
    candidates.append(str(model_path))
    candidates.append(str(model_path.with_name(model_path.stem + "_compiled.model")))
    candidates.append(
        str(model_path.with_name(model_path.stem + "_stagetwo_compiled.model"))
    )

    for cand in candidates:
        if os.path.exists(cand):
            try:
                sm = torch.jit.load(cand, map_location=device)
                return sm, cand
            except Exception:  # pylint: disable=broad-exception-caught
                # try next
                continue
    return None, None


def main():
    args = parse_args()
    run(args)


def run(args: argparse.Namespace):
    torch_tools.set_default_dtype(args.default_dtype)
    is_distributed, rank, world_size, local_rank, device = init_distributed_if_needed(
        args.device
    )

    if is_distributed:
        if rank == 0:
            print(f"Distributed mode: world_size={world_size}, backend initialized.")
        print(f"Rank {rank}: local_rank={local_rank}, using device {device}")
    else:
        print(f"Running non-distributed on device {device}")

    # read configs (all ranks; we will only write subset per rank)
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    num_configs = len(atoms_list)

    # electric field
    if args.electric_field is not None:
        electric_field = torch.tensor(args.electric_field, dtype=torch.float32).to(
            device
        )
    else:
        electric_field = None

    # Try to use a TorchScript compiled model if present (fast per-rank load).
    model_path = Path(args.model)
    compiled_model, compiled_path = try_load_compiled_candidate(model_path, device)
    model_is_compiled = compiled_model is not None

    # If compiled model used, we still need model metadata (atomic_numbers, r_max, etc.)
    # We'll attempt to load metadata from a .meta.json next to the provided model; else rank 0
    # loads the original saved model and broadcasts small metadata.
    meta_candidates = [
        str(model_path) + ".meta.json",
        str(model_path.with_suffix(".meta.json")),
        str(model_path.with_name(model_path.stem + ".meta.json")),
    ]
    meta = None
    for cand in meta_candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                break
            except Exception:  # pylint: disable=broad-exception-caught
                meta = None

    # If we didn't find a meta file, and we are distributed, let rank 0 load the saved e3nn model
    # (on CPU) to extract minimal metadata then broadcast it. If not distributed, load locally.
    if meta is None:
        if is_distributed:
            if rank == 0:
                try:
                    base_model = safe_torch_load(str(args.model), map_location="cpu")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    raise RuntimeError(
                        f"Rank 0 failed to load model for metadata extraction: {e}"
                    ) from e
                try:
                    atomic_numbers_list = [int(z) for z in base_model.atomic_numbers]
                except Exception:  # pylint: disable=broad-exception-caught
                    atomic_numbers_list = [
                        int(z) for z in getattr(base_model, "atomic_numbers", [])
                    ]
                heads = getattr(base_model, "heads", None)
                r_max = float(getattr(base_model, "r_max", None))
                num_interactions = getattr(base_model, "num_interactions", None)
                # products irreps as string is enough for descriptor layout
                try:
                    products_irreps = str(base_model.products[0].linear.irreps_out)
                except Exception:  # pylint: disable=broad-exception-caught
                    products_irreps = None
                meta = {
                    "atomic_numbers": atomic_numbers_list,
                    "heads": heads,
                    "r_max": r_max,
                    "num_interactions": num_interactions,
                    "products_irreps": products_irreps,
                }
            else:
                meta = None
            # broadcast metadata
            # torch.distributed.broadcast_object_list expects a list
            obj = [meta]
            torch.distributed.broadcast_object_list(obj, src=0)
            meta = obj[0]
        else:
            # non-distributed: load locally
            try:
                base_model = safe_torch_load(str(args.model), map_location="cpu")
            except Exception as e:  # pylint: disable=broad-exception-caught
                raise RuntimeError(
                    f"Failed to load model to extract metadata: {e}"
                ) from e
            try:
                atomic_numbers_list = [int(z) for z in base_model.atomic_numbers]
            except Exception:  # pylint: disable=broad-exception-caught
                atomic_numbers_list = [
                    int(z) for z in getattr(base_model, "atomic_numbers", [])
                ]
            heads = getattr(base_model, "heads", None)
            r_max = float(getattr(base_model, "r_max", None))
            num_interactions = getattr(base_model, "num_interactions", None)
            try:
                products_irreps = str(base_model.products[0].linear.irreps_out)
            except Exception:  # pylint: disable=broad-exception-caught
                products_irreps = None
            meta = {
                "atomic_numbers": atomic_numbers_list,
                "heads": heads,
                "r_max": r_max,
                "num_interactions": num_interactions,
                "products_irreps": products_irreps,
            }

    # Metadata now available in `meta`
    atomic_numbers_list = meta.get("atomic_numbers", [])
    heads = meta.get("heads", None)
    r_max = meta.get("r_max", None)
    num_interactions = meta.get("num_interactions", None)
    products_irreps = meta.get("products_irreps", None)

    # Prepare dataset & sampler. We need AtomicData instances for the dataset.
    z_table = utils.AtomicNumberTable([int(z) for z in atomic_numbers_list])
    # Build dataset: attach original index
    dataset = []
    for idx, atoms in enumerate(atoms_list):
        cfg = data.config_from_atoms(
            atoms,
            key_specification=data.KeySpecification(
                info_keys={"electric_field": args.electric_field_key}, arrays_keys={}
            ),
        )
        ad = data.AtomicData.from_config(
            cfg, z_table=z_table, cutoff=float(r_max), heads=heads
        )
        setattr(ad, "_orig_idx", idx)
        dataset.append(ad)

    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False, drop_last=False
        )
        local_indices = list(iter(sampler))
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None
        local_indices = list(range(len(dataset)))
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )

    # MODEL LOADING (per-rank) -- minimal overhead strategy
    model = None
    if model_is_compiled:
        # If compiled model exists, prefer it: load per-rank directly
        try:
            model = torch.jit.load(compiled_path, map_location=device)
            print(
                f"Rank {rank if is_distributed else 0}: loaded TorchScript compiled model {compiled_path}"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # fallback to loading e3nn model
            print(
                f"Rank {rank if is_distributed else 0}: failed to load compiled model ({compiled_path}): {e}. Falling back to e3nn load."
            )
            model_is_compiled = False

    if not model_is_compiled:
        # Load the full e3nn model per-rank (map_location=cpu then .to(device)); small ranks may avoid loading heavy objects if rank0 already broadcasted metadata,
        # but we still need a runnable model for forward. We'll load from args.model into cpu and then move.
        try:
            # load to CPU then move to device
            base_model = safe_torch_load(str(args.model), map_location="cpu")
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise RuntimeError(f"Failed to load model for inference: {e}") from e
        # Move to device & dtype
        base_model = base_model.to(device)
        model = base_model

        # Convert to CuEq on each rank if requested (robust)
        if args.enable_cueq:
            try:
                print(
                    f"Rank {rank if is_distributed else 0}: converting to CuEq on {device}"
                )
                model = run_e3nn_to_cueq(model, device=device)
                print(f"Rank {rank if is_distributed else 0}: CuEq conversion done")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(
                    f"Rank {rank if is_distributed else 0}: conversion to CuEq failed: {e}. Continuing with E3NN model."
                )

    # Prepare model for inference
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # We'll produce per-rank atoms to write
    local_atoms_out = []

    # Helper to set atoms fields (keeps keys consistent)
    def populate_atoms_fields(
        atoms,
        idx,
        energy,
        forces,
        stress=None,
        bec=None,
        qs=None,
        polarization=None,
        becs=None,
        polarizability=None,
        contributions=None,
        descriptors=None,
        node_energies=None,
    ):
        atoms.calc = None
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces
        if stress is not None:
            atoms.info[args.info_prefix + "stress"] = np.array(stress).reshape(9)
        if bec is not None:
            atoms.arrays[args.info_prefix + "BEC"] = np.asarray(bec).reshape(-1, 9)
        if qs is not None:
            atoms.arrays[args.info_prefix + "latent_charges"] = np.asarray(qs)
        if polarization is not None:
            atoms.info[args.info_prefix + "polarization"] = polarization
        if becs is not None:
            atoms.arrays[args.info_prefix + "becs"] = np.asarray(becs).reshape(-1, 9)
        if polarizability is not None:
            atoms.info[args.info_prefix + "polarizability"] = np.asarray(
                polarizability
            ).reshape(9)
        if contributions is not None:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions
        if descriptors is not None:
            if args.descriptor_aggregation_method:
                if args.descriptor_aggregation_method == "mean":
                    descriptors = np.mean(descriptors, axis=0)
                elif args.descriptor_aggregation_method == "per_element_mean":
                    descriptors = {
                        element: np.mean(
                            descriptors[atoms.symbols == element], axis=0
                        ).tolist()
                        for element in np.unique(atoms.symbols)
                    }
                atoms.info[args.info_prefix + "descriptors"] = descriptors
            else:
                atoms.arrays[args.info_prefix + "descriptors"] = np.array(descriptors)
        if node_energies is not None:
            atoms.arrays[args.info_prefix + "node_energies"] = node_energies
        # record original index to allow rank-0 merging
        atoms.info["orig_idx"] = int(idx)

    # Inference loop
    # For descriptor processing we need product irreps info if descriptors requested
    if args.return_descriptors:
        if products_irreps is None:
            raise RuntimeError(
                "Descriptor requested but product irreps metadata not found."
            )
        irreps_out = o3.Irreps(str(products_irreps))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features_template = [
            irreps_out.dim for _ in range(int(num_interactions))
        ]
        per_layer_features_template[-1] = num_invariant_features

    local_ptr = 0  # index into local_indices
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = get_model_output(
                model,
                batch.to_dict(),
                args.compute_stress,
                args.compute_bec,
                args.compute_polarization,
                args.compute_becs,
                args.compute_polarizability,
                electric_field,
            )

            n_graphs = int(batch.ptr.shape[0] - 1)
            batch_indices = local_indices[local_ptr : local_ptr + n_graphs]
            local_ptr += n_graphs

            energies_batch = torch_tools.to_numpy(output["energy"])
            if energies_batch.ndim == 0:
                energies_batch = np.expand_dims(energies_batch, 0)

            forces_splits = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )[:-1]

            # stress
            if args.compute_stress:
                stresses_batch = torch_tools.to_numpy(output["stress"])
                if stresses_batch.ndim == 1:
                    stresses_batch = stresses_batch.reshape(-1, 9)

            # BEC
            if args.compute_bec:
                becs_batch = np.split(
                    torch_tools.to_numpy(output["BEC"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[:-1]
                qs_batch = np.split(
                    torch_tools.to_numpy(output["latent_charges"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[:-1]

            # MACEField
            if args.compute_polarization:
                pols_batch = torch_tools.to_numpy(output["polarization"])
            if args.compute_polarizability:
                polars_batch = torch_tools.to_numpy(output["polarizability"])
            if args.compute_becs:
                becs_coll_batch = np.split(
                    torch_tools.to_numpy(output["becs"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )
                becs_coll_batch = [b.reshape(-1, 9) for b in becs_coll_batch[:-1]]

            if args.return_contributions:
                contribs_batch = torch_tools.to_numpy(output["contributions"])

            if args.return_descriptors:
                num_layers = args.descriptor_num_layers
                if num_layers == -1:
                    num_layers = int(num_interactions)
                per_layer_features = list(per_layer_features_template)
                per_layer_features[-1] = num_invariant_features

                descriptors = output["node_feats"]
                if args.descriptor_invariants_only:
                    descriptors = extract_invariant(
                        descriptors,
                        num_layers=num_layers,
                        num_features=num_invariant_features,
                        l_max=l_max,
                    )
                to_keep = np.sum(per_layer_features[:num_layers])
                descriptors = descriptors[:, :to_keep].detach().cpu().numpy()
                descriptors_splits = np.split(
                    descriptors, indices_or_sections=batch.ptr[1:], axis=0
                )[:-1]

            if args.return_node_energies:
                node_energies_splits = np.split(
                    torch_tools.to_numpy(output["node_energy"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[:-1]

            # iterate over graphs in this batch and populate atoms
            for j, cfg_idx in enumerate(batch_indices):
                energy = energies_batch[j]
                forces = forces_splits[j]
                stress = stresses_batch[j] if args.compute_stress else None
                bec = becs_batch[j] if args.compute_bec else None
                qs = qs_batch[j] if args.compute_bec else None
                polarization = pols_batch[j] if args.compute_polarization else None
                polarizability = (
                    polars_batch[j] if args.compute_polarizability else None
                )
                becs_coll = becs_coll_batch[j] if args.compute_becs else None
                contributions = contribs_batch[j] if args.return_contributions else None
                descriptors_loc = (
                    descriptors_splits[j] if args.return_descriptors else None
                )
                node_energies_loc = (
                    node_energies_splits[j] if args.return_node_energies else None
                )

                atoms = atoms_list[cfg_idx]
                populate_atoms_fields(
                    atoms,
                    cfg_idx,
                    energy,
                    forces,
                    stress=stress,
                    bec=bec,
                    qs=qs,
                    polarization=polarization,
                    becs=becs_coll,
                    polarizability=polarizability,
                    contributions=contributions,
                    descriptors=descriptors_loc,
                    node_energies=node_energies_loc,
                )
                local_atoms_out.append(atoms)

    # Write per-rank results
    rank_file = f"{args.output}.rank{rank}.extxyz" if is_distributed else args.output
    if is_distributed:
        # write subset only
        ase.io.write(rank_file, images=local_atoms_out, format="extxyz")
        # barrier so other ranks finish writing
        torch.distributed.barrier()

        # rank 0 merges
        if rank == 0:
            merged = [None] * num_configs
            # read each rank file and map by orig_idx
            for r in range(world_size):
                fpath = f"{args.output}.rank{r}.extxyz"
                if not os.path.exists(fpath):
                    raise RuntimeError(f"Expected rank output file {fpath} missing")
                imgs = ase.io.read(fpath, index=":")
                for img in imgs:
                    if "orig_idx" not in img.info:
                        raise RuntimeError(
                            "Missing 'orig_idx' info key in per-rank file"
                        )
                    idx = int(img.info.pop("orig_idx"))
                    merged[idx] = img
            # sanity check
            if any(m is None for m in merged):
                missing_idxs = [i for i, m in enumerate(merged) if m is None]
                raise RuntimeError(f"Missing outputs for indices: {missing_idxs}")
            ase.io.write(args.output, images=merged, format="extxyz")
            # cleanup rank files
            for r in range(world_size):
                try:
                    os.remove(f"{args.output}.rank{r}.extxyz")
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        # final barrier then exit
        torch.distributed.barrier()
    else:
        # single-process: already wrote directly to args.output
        ase.io.write(args.output, images=local_atoms_out, format="extxyz")

    # cleanup process group
    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
