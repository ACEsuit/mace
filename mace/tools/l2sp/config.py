"""L2-SP configuration and optimizer groups for already-selected parameters."""

import json
import logging
import math


def add_arguments(parser, str2bool):
    parser.add_argument(
        "--l2sp",
        type=str2bool,
        default=False,
        help="Regularize fine-tuning towards the pretrained starting weights",
    )
    parser.add_argument(
        "--l2sp_strength",
        type=float,
        default=None,
        help="Non-negative L2-SP coefficient (required with --l2sp)",
    )


def validate_args(args):
    if not getattr(args, "l2sp", False):
        if getattr(args, "l2sp_strength", None) is not None:
            raise ValueError("--l2sp_strength requires --l2sp=True")
        return
    strength = args.l2sp_strength
    if strength is None or not math.isfinite(strength) or strength < 0:
        raise ValueError("--l2sp requires an explicit, finite --l2sp_strength >= 0")
    if args.foundation_model is None:
        raise ValueError("--l2sp requires --foundation_model")
    if args.weight_decay != 0:
        raise ValueError(
            "--l2sp requires --weight_decay=0; use --l2sp_strength for regularization towards starting weights"
        )
    if args.multiheads_finetuning:
        raise ValueError("--l2sp requires --multiheads_finetuning=False")
    if args.lora:
        raise ValueError("--l2sp cannot be combined with --lora")
    if not args.foundation_model_readout:
        raise ValueError("--l2sp requires --foundation_model_readout=True")
    if args.enable_oeq and not args.enable_cueq:
        raise ValueError("--l2sp does not currently support the OEQ backend")
    if args.embedding_specs:
        raise ValueError("--l2sp does not support adding or changing embeddings")


def get_param_options(args, model):
    """Retain LR multipliers and apply L2-SP to all eligible parameters."""
    if type(model).__name__ not in ("MACE", "ScaleShiftMACE") or len(model.heads) != 1:
        raise ValueError("L2-SP currently supports single-head MACE and ScaleShiftMACE")
    if getattr(getattr(model, "oeq_config", None), "enabled", False):
        raise ValueError("L2-SP does not currently support the OEQ backend")
    parameters = dict(model.named_parameters())
    selected = {name for name, param in parameters.items() if param.requires_grad}
    if not selected:
        raise ValueError("L2-SP selected no trainable parameters")
    factors = json.loads(args.lr_params_factors)
    groups = {}
    for name, param in parameters.items():
        if name not in selected:
            continue
        module_name = name.split(".", 1)[0]
        group_name = "embedding" if module_name == "node_embedding" else module_name
        if group_name not in groups:
            factor = factors.get(f"{group_name}_lr_factor", 1.0)
            if not math.isfinite(factor) or factor < 0:
                raise ValueError(f"Invalid L2-SP learning-rate factor for {group_name}")
            groups[group_name] = {
                "name": group_name,
                "params": [],
                "param_names": [],
                "lr": args.lr * factor,
                "weight_decay": 0.0,
            }
        groups[group_name]["params"].append(param)
        groups[group_name]["param_names"].append(name)
    logging.info(
        "L2-SP trainable parameters: %s",
        sum(parameters[name].numel() for name in selected),
    )
    logging.info(
        "L2-SP strength: %s; selected tensors: %s",
        args.l2sp_strength,
        sorted(selected),
    )
    return {
        "params": list(groups.values()),
        "lr": args.lr,
        "betas": (args.beta, 0.999),
        "amsgrad": args.amsgrad,
        "weight_decay": 0.0,
    }
