import torch
import numpy as np
import ast
import mace
import logging
from mace import data, modules, tools
from e3nn import o3
from mace.tools import torch_geometric
from mace.tools.scripts_utils import (
    get_dataset_from_xyz
)
from copy import deepcopy

def load_origin_model(model_path):
    model = torch.load(model_path)
    return model.to("cpu")

def load_wrapper_model():
    args = tools.build_default_arg_parser().parse_args()
    tag = tools.get_tag(name=args.name, seed=args.seed)

    # Setup
    tools.set_seeds(args.seed)
    tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
    try:
        logging.info(f"MACE version: {mace.__version__}")
    except AttributeError:
        logging.info("Cannot find MACE version, please install MACE via pip")
    logging.info(f"Configuration: {args}")
    device = tools.init_device(args.device)
    tools.set_default_dtype(args.default_dtype)

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )
    logging.info(z_table)
    if args.model == "AtomicDipolesMACE":
        atomic_energies = None
        dipole_only = True
        compute_dipole = True
        compute_energy = False
        args.compute_forces = False
        compute_virials = False
        args.compute_stress = False
    else:
        dipole_only = False
        if args.model == "EnergyDipolesMACE":
            compute_dipole = True
            compute_energy = True
            args.compute_forces = True
            compute_virials = False
            args.compute_stress = False
        else:
            compute_energy = True
            compute_dipole = False
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            if args.E0s is not None:
                logging.info(
                    "Atomic Energies not in training file, using command line argument E0s"
                )
                if args.E0s.lower() == "average":
                    logging.info(
                        "Computing average Atomic Energies using least squares regression"
                    )
                    atomic_energies_dict = data.compute_average_E0s(
                        collections.train, z_table
                    )
                else:
                    try:
                        atomic_energies_dict = ast.literal_eval(args.E0s)
                        assert isinstance(atomic_energies_dict, dict)
                    except Exception as e:
                        raise RuntimeError(
                            f"E0s specified invalidly, error {e} occurred"
                        ) from e
            else:
                raise RuntimeError(
                    "E0s not found in training file and not specified in command line"
                )
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_max)
            for config in collections.train
        ],
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    loss_fn: torch.nn.Module
    if args.loss == "weighted":
        loss_fn = modules.WeightedEnergyForcesLoss(
            energy_weight=args.energy_weight, forces_weight=args.forces_weight
        )
    elif args.loss == "forces_only":
        loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
    elif args.loss == "virials":
        loss_fn = modules.WeightedEnergyForcesVirialsLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            virials_weight=args.virials_weight,
        )
    elif args.loss == "stress":
        loss_fn = modules.WeightedEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
        )
    elif args.loss == "huber":
        loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            stress_weight=args.stress_weight,
            huber_delta=args.huber_delta,
        )
    elif args.loss == "dipole":
        assert (
            dipole_only is True
        ), "dipole loss can only be used with AtomicDipolesMACE model"
        loss_fn = modules.DipoleSingleLoss(
            dipole_weight=args.dipole_weight,
        )
    elif args.loss == "energy_forces_dipole":
        assert dipole_only is False and compute_dipole is True
        loss_fn = modules.WeightedEnergyForcesDipoleLoss(
            energy_weight=args.energy_weight,
            forces_weight=args.forces_weight,
            dipole_weight=args.dipole_weight,
        )
    else:
        # Unweighted Energy and Forces loss by default
        loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    logging.info(loss_fn)

    if args.compute_avg_num_neighbors:
        args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
    logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

    # Selecting outputs
    compute_virials = False
    if args.loss in ("stress", "virials", "huber"):
        compute_virials = True
        args.compute_stress = True
        args.error_table = "PerAtomRMSEstressvirials"
    
    output_args = {
        "energy": compute_energy,
        "forces": args.compute_forces,
        "virials": compute_virials,
        "stress": args.compute_stress,
        "dipoles": compute_dipole,
    }
    logging.info(f"Selected the following outputs: {output_args}")

    # Build model
    logging.info("Building model")
    if args.num_channels is not None and args.max_L is not None:
        assert args.num_channels > 0, "num_channels must be positive integer"
        assert args.max_L >= 0, "max_L must be non-negative integer"
        args.hidden_irreps = o3.Irreps(
            (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
            .sort()
            .irreps.simplify()
        )

    assert (
        len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
    ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

    logging.info(f"Hidden irreps: {args.hidden_irreps}")
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_radial_basis,
        num_polynomial_cutoff=args.num_cutoff_basis,
        max_ell=args.max_ell,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=args.num_interactions,
        num_elements=len(z_table),
        hidden_irreps=o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=args.avg_num_neighbors,
        atomic_numbers=z_table.zs,
    )
    if args.scaling == "no_scaling":
        std = 1.0
        logging.info("No scaling selected")
    else:
        mean, std = modules.scaling_classes[args.scaling](
            train_loader, atomic_energies
        )
        model = modules.ScaleShiftMACEWrapper(
            **model_config,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticInteractionBlock"
            ],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            radial_MLP=ast.literal_eval(args.radial_MLP),
            radial_type=args.radial_type,
        )
    return train_loader, model

def copy_weights(old, new):
    new.model.load_state_dict(old.state_dict())
    new.model.interactions = deepcopy(old.interactions)
    new.model.products = deepcopy(old.products)
    new.model.readouts = deepcopy(old.readouts)

def verify_models(old, new, loader, output_args):
    for batch in loader:
        batch_dict = batch.to_dict()
        test_dict = deepcopy(batch_dict)
        new_output = new(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        
        old_output = old(
            test_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        # compare outputs
        for newOutput, oldOutput in zip(new_output, old_output):
            print(
                newOutput,
                np.allclose(
                    new_output[newOutput].detach().numpy(), 
                    old_output[oldOutput].detach().numpy()
                )
            )
        break
        
if __name__=='__main__':
    model = load_origin_model("mace.model")
    train_loader, wrapper_model = load_wrapper_model()
    copy_weights(model, wrapper_model)
    verify_models(
        model,
        wrapper_model,
        train_loader,
        {'energy': True, 'forces': True, 'virials': False, 'stress': True, 'dipoles': False}
    )
    torch.save(wrapper_model.model, "test.model")
