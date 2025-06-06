import ast
import logging

import numpy as np
from e3nn import o3

from mace import modules
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.utils import AtomicNumberTable


def configure_model(
    args,
    train_loader,
    atomic_energies,
    model_foundation=None,
    heads=None,
    z_table=None,
    head_configs=None,
):
    # Selecting outputs
    compute_virials = args.loss == "virials"
    compute_stress = args.loss in ("stress", "huber", "universal")

    if compute_virials:
        args.compute_virials = True
        args.error_table = "PerAtomRMSEstressvirials"
    elif compute_stress:
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"

    output_args = {
        "energy": args.compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": compute_stress,
        "dipoles": args.compute_dipole,
    }
    logging.info(
        f"During training the following quantities will be reported: {', '.join([f'{report}' for report, value in output_args.items() if value])}"
    )
    logging.info("===========MODEL DETAILS===========")

    if args.scaling == "no_scaling":
        args.std = 1.0
        if head_configs is not None:
            for head_config in head_configs:
                head_config.std = 1.0
        logging.info("No scaling selected")

    if (
        head_configs is not None
        and args.std is not None
        and not isinstance(args.std, list)
    ):
        atomic_inter_scale = []
        for head_config in head_configs:
            if hasattr(head_config, "std") and head_config.std is not None:
                atomic_inter_scale.append(head_config.std)
            elif args.std is not None:
                atomic_inter_scale.append(
                    args.std if isinstance(args.std, float) else 1.0
                )
        args.std = atomic_inter_scale

    elif (args.mean is None or args.std is None) and args.model != "AtomicDipolesMACE":
        args.mean, args.std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )

    # Build model
    if model_foundation is not None and args.model in ["MACE", "ScaleShiftMACE"]:
        logging.info("Loading FOUNDATION model")
        model_config_foundation = extract_config_mace_model(model_foundation)
        model_config_foundation["atomic_energies"] = atomic_energies

        if args.foundation_model_elements:
            foundation_z_table = AtomicNumberTable(
                [int(z) for z in model_foundation.atomic_numbers]
            )
            model_config_foundation["atomic_numbers"] = foundation_z_table.zs
            model_config_foundation["num_elements"] = len(foundation_z_table)
            z_table = foundation_z_table
            logging.info(
                f"Using all elements from foundation model: {foundation_z_table.zs}"
            )
        else:
            model_config_foundation["atomic_numbers"] = z_table.zs
            model_config_foundation["num_elements"] = len(z_table)
            logging.info(f"Using filtered elements: {z_table.zs}")

        args.max_L = model_config_foundation["hidden_irreps"].lmax

        if args.model == "MACE" and model_foundation.__class__.__name__ == "MACE":
            model_config_foundation["atomic_inter_shift"] = [0.0] * len(heads)
        else:
            model_config_foundation["atomic_inter_shift"] = (
                _determine_atomic_inter_shift(args.mean, heads)
            )
        model_config_foundation["atomic_inter_scale"] = [1.0] * len(heads)
        args.avg_num_neighbors = model_config_foundation["avg_num_neighbors"]
        args.model = "FoundationMACE"
        model_config_foundation["heads"] = heads
        model_config = model_config_foundation

        logging.info("Model configuration extracted from foundation model")
        logging.info(f"Using {args.loss} loss function for fine-tuning")
        logging.info(
            f"Message passing with hidden irreps {model_config_foundation['hidden_irreps']})"
        )
        logging.info(
            f"{model_config_foundation['num_interactions']} layers, each with correlation order: {model_config_foundation['correlation']} (body order: {model_config_foundation['correlation']+1}) and spherical harmonics up to: l={model_config_foundation['max_ell']}"
        )
        logging.info(
            f"Radial cutoff: {model_config_foundation['r_max']} A (total receptive field for each atom: {model_config_foundation['r_max'] * model_config_foundation['num_interactions']} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {model_config_foundation['distance_transform']}"
        )
    else:
        logging.info("Building model")
        logging.info(
            f"Message passing with {args.num_channels} channels and max_L={args.max_L} ({args.hidden_irreps})"
        )
        logging.info(
            f"{args.num_interactions} layers, each with correlation order: {args.correlation} (body order: {args.correlation+1}) and spherical harmonics up to: l={args.max_ell}"
        )
        logging.info(
            f"{args.num_radial_basis} radial and {args.num_cutoff_basis} basis functions"
        )
        logging.info(
            f"Radial cutoff: {args.r_max} A (total receptive field for each atom: {args.r_max * args.num_interactions} A)"
        )
        logging.info(
            f"Distance transform for radial basis functions: {args.distance_transform}"
        )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")

        cueq_config = None
        if args.only_cueq:
            logging.info("Using only the backend of the model")
            cueq_config = CuEquivarianceConfig(
                enabled=True,
                layout="ir_mul",
                group="O3_e3nn",
                optimize_all=True,
            )

        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            edge_irreps=o3.Irreps(args.edge_irreps) if args.edge_irreps else None,
            atomic_energies=atomic_energies,
            apply_cutoff=args.apply_cutoff,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            use_reduced_cg=args.use_reduced_cg,
            use_so3=args.use_so3,
            cueq_config=cueq_config,
        )
        model_config_foundation = None

    model = _build_model(args, model_config, model_config_foundation, heads)

    if model_foundation is not None:
        model = load_foundations_elements(
            model,
            model_foundation,
            z_table,
            load_readout=args.foundation_filter_elements,
            max_L=args.max_L,
        )

    return model, output_args


def _determine_atomic_inter_shift(mean, heads):
    if isinstance(mean, np.ndarray):
        if mean.size == 1:
            return mean.item()
        if mean.size == len(heads):
            return mean.tolist()
        logging.info("Mean not in correct format, using default value of 0.0")
        return [0.0] * len(heads)
    if isinstance(mean, list) and len(mean) == len(heads):
        return mean
    if isinstance(mean, float):
        return [mean] * len(heads)
    logging.info("Mean not in correct format, using default value of 0.0")
    return [0.0] * len(heads)


def _build_model(
    args, model_config, model_config_foundation, heads
):  # pylint: disable=too-many-return-statements
    if args.model == "MACE":
        if args.interaction_first not in [
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
        ]:
            args.interaction_first = "RealAgnosticInteractionBlock"
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=[0.0] * len(heads),
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "ScaleShiftMACE":
        return modules.ScaleShiftMACE(
            **model_config,
            pair_repulsion=args.pair_repulsion,
            distance_transform=args.distance_transform,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=args.std,
            atomic_inter_shift=args.mean,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
            heads=heads,
        )
    if args.model == "FoundationMACE":
        return modules.ScaleShiftMACE(**model_config_foundation)
    if args.model == "ScaleShiftBOTNet":
        # say it is deprecated
        raise RuntimeError("ScaleShiftBOTNet is deprecated, use MACE instead")
    if args.model == "BOTNet":
        raise RuntimeError("BOTNet is deprecated, use MACE instead")
    if args.model == "AtomicDipolesMACE":
        assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
        assert (
            args.error_table == "DipoleRMSE"
        ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
        return modules.AtomicDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    if args.model == "EnergyDipolesMACE":
        assert (
            args.loss == "energy_forces_dipole"
        ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
        assert (
            args.error_table == "EnergyDipoleRMSE"
        ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
        return modules.EnergyDipolesMACE(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
        )
    raise RuntimeError(f"Unknown model: '{args.model}'")
