######################################################################
# Script to calibrate the LLPR uncertainties of a trained MACE model
# Authors: Sanggyu Chong, Filippo Bigi
# This program is distributed under the MIT License (see MIT.md)
######################################################################

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brute, root_scalar
from scipy.stats import norm

from mace import data
from mace.modules import LLPRModel
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.scripts_utils import get_config_type_weights, get_dataset_from_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        help="path to model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_configs",
        help="path to XYZ configurations used for model training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--calib_configs",
        help="path to XYZ configurations used for calibration",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--calib_model_output",
        help="output path for the calibrated model",
        type=str,
        default="MACE_LLPR_calib.model",
    )
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default="REF_energy",
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default="REF_forces",
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default="REF_virials",
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default="REF_stress",
    )
    parser.add_argument(
        "--batch_size",
        help="batch size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--head",
        help="Model head targeted for UQ calibration",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--include_forces",
        help="include forces in the covariance (and evaluation)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_virials",
        help="include virials in the covariance (and evaluation)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--include_stress",
        help="include stresses in the covariance (and evaluation)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--recompute_covariance",
        help="recompute the covariance matrix even if it already is computed",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--alpha",
        help="alpha parameter of LLPR UQ (can be user-specified)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--sigma",
        help="sigma parameter of LLPR UQ (can be user-specified)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--calib_bin_size",
        help="bin size to use during sum of squared losses calibration",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--calib_bound",
        help="bounds of alpha screened during calibration, 10**( -/+ val)",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--calib_delta",
        help="delta used to define alpha screening range for calibration",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--eval_configs",
        help="invoke UQ evaluation by providing an evaluation dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--eval_forces",
        help="include forces in UQ evaluation (overridden by '--include_forces')",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_virials",
        help="include forces in UQ evaluation (overridden by '--include_virials')",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_stress",
        help="include forces in UQ evaluation (overridden by '--include_stress')",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:

    torch_tools.set_default_dtype(args.default_dtype)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(args.device)

    # Check if model is already wrapped with LLPR class
    if model.__class__.__name__ == "LLPRModel":
        model_llpr = model
        z_table = utils.AtomicNumberTable(
            [int(z) for z in model_llpr.orig_model.atomic_numbers]
        )
        r_max = float(model_llpr.orig_model.r_max)
    else:
        # Wrap model in LLPR class
        model_llpr = LLPRModel(model)
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        r_max = float(model.r_max)

    # atomic_energies_dict = model.atomic_energies_dict
    config_type_weights = get_config_type_weights(args.config_type_weights)

    # Load datasets
    collections, _ = get_dataset_from_xyz(
        work_dir="",
        train_path=args.train_configs,
        valid_path=args.calib_configs,  # calib saved as valid
        test_path=args.eval_configs,  # eval saved as test
        valid_fraction=0.0,
        config_type_weights=config_type_weights,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
    )

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=r_max,
                heads=heads,
            )
            for config in collections.train
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    calib_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=r_max,
                heads=heads,
            )
            for config in collections.valid
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    if args.eval_configs is not None:
        eval_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=r_max,
                    heads=heads,
                )
                for config in collections.test
            ],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

    # Collect targets for covariance computation and evaluation based on flags
    if args.include_forces:
        eval_forces = True
    else:
        eval_forces = args.eval_forces
    if args.include_virials:
        eval_virials = True
    else:
        eval_virials = args.eval_virials
    if args.include_stress:
        eval_stress = True
    else:
        eval_stress = args.eval_stress

    eval_targets = ["energy"]
    if eval_forces:
        eval_targets.append("forces")
    if eval_virials:
        eval_targets.append("virials")
    if eval_stress:
        eval_targets.append("stress")

    # Check weights to make sure all targets are read properly.
    # (Catches target key issues)
    batch = next(iter(calib_loader))
    batch.to(args.device)
    batch_dict = batch.to_dict()
    for target in eval_targets:
        if batch_dict[target + "_weight"].min() == 0:
            raise RuntimeError(
                f"Zero weight for requested target '{target}'"
                + "routine cannot be executed. Check that you"
                + "are supplying the correct target keys"
            )

    # Compute covariance matrix
    if (not model_llpr.covariance_computed) or args.recompute_covariance:

        # Reset LLPR matrices if covariance recomputation is requested
        if args.recompute_covariance:
            model_llpr.reset_matrices()

        model_llpr.compute_covariance(
            train_loader,
            include_forces=args.include_forces,
            include_virials=args.include_virials,
            include_stresses=args.include_stress,
        )

    # Make sure covariance matrix is available before proceeding
    assert model_llpr.covariance_computed

    # Compute inverse covariance if user provides their own alpha value
    if args.alpha is not None:
        model_llpr.compute_inv_covariance(args.alpha, args.sigma)
        # Terminate routine if further evaluation is not needed
        if args.eval_configs is None:
            torch.save(model_llpr, args.calib_model_output)
            return

    # Perform an energy-based auto LLPR UQ calibration (alpha-only)
    if not model_llpr.inv_covariance_computed:

        errors = []
        ll_feats = []

        for batch in iter(calib_loader):
            batch.to(args.device)
            batch_dict = batch.to_dict()
            outputs = model_llpr(
                batch_dict,
                compute_force=False,
                compute_energy_uncertainty=False,
            )
            cur_ref = batch_dict["energy"].detach()
            cur_pred = outputs["energy"].detach()
            errors.append((cur_ref - cur_pred) ** 2)

            num_graphs = batch_dict["ptr"].numel() - 1
            ll_feats.append(
                model_llpr.aggregate_ll_features(
                    outputs["node_feats"], batch_dict["batch"], num_graphs
                ).detach()
            )

        errors = torch.cat(errors, dim=0)
        ll_feats_all = torch.cat(ll_feats, dim=0)

        # Define range of alpha values to consider
        calib_slice = slice(
            -1 * args.calib_bound,
            args.calib_bound + 0.01,
            args.calib_delta,
        )

        # Auto-calibrate alpha value and compute inverse covariance
        opt_alpha = brute(
            _ssl_wrapper,
            ranges=[calib_slice],
            args=(model_llpr, ll_feats_all, errors, args.sigma, args.calib_bin_size),
        )
        print(f"alpha auto-calibrated to: {10 ** opt_alpha[0]:.4E}".format())
        model_llpr.compute_inv_covariance(10 ** opt_alpha[0], args.sigma)

    # Make sure inverse covariance matrix is available before proceeding
    assert model_llpr.inv_covariance_computed

    # Save calibrated model for the user
    torch.save(model_llpr, args.calib_model_output)

    # Evaluate UQ quality for the calibration dataset when requested
    if args.eval_configs is not None:

        # Compute actual and predicted errors on the calibration dataset
        errors = defaultdict(lambda: [])
        pred_errors = defaultdict(lambda: [])

        for batch in iter(eval_loader):
            batch.to(args.device)
            batch_dict = batch.to_dict()
            outputs = model_llpr(
                batch,
                compute_force=eval_forces,
                compute_force_uncertainty=eval_forces,
                compute_virials=eval_virials,
                compute_virial_uncertainty=eval_virials,
                compute_stress=eval_stress,
                compute_stress_uncertainty=eval_stress,
            )
            for target in eval_targets:
                cur_ref = batch_dict[target].cpu().detach().numpy().flatten()
                cur_pred = outputs[target].cpu().detach().numpy().flatten()
                cur_error = (cur_ref - cur_pred) ** 2
                errors[target].append(cur_error)
                pred_errors[target].append(
                    outputs[target + "_uncertainty"].cpu().detach().numpy().flatten()
                )

        for target in eval_targets:
            errors[target] = np.concatenate(errors[target]).flatten()
            pred_errors[target] = np.concatenate(pred_errors[target]).flatten()

        # Routines to obtain isolines in the log-log scale
        desired_fractions = [
            norm.cdf(1, 0.0, 1.0) - norm.cdf(-1, 0.0, 1.0),  # 1 sig
            norm.cdf(2, 0.0, 1.0) - norm.cdf(-2, 0.0, 1.0),  # 2 sig
            norm.cdf(3, 0.0, 1.0) - norm.cdf(-3, 0.0, 1.0),  # 3 sig
        ]
        err_sigs = np.linspace(1e-8, 5e0, 5)

        lower_bounds = []
        upper_bounds = []

        for desired_fraction in desired_fractions:
            lower_bounds.append([])
            upper_bounds.append([])
            for sig in err_sigs:
                isoline_value = _find_fraction(sig, desired_fraction)
                x1, x2 = _find_where_pdf_is_c(isoline_value, sig)
                lower_bounds[-1].append(x1)
                upper_bounds[-1].append(x2)

            additional_sigma = 100.0
            lower_bounds[-1].append(
                lower_bounds[-1][-1]
                + (lower_bounds[-1][-1] - lower_bounds[-1][-2])
                / (err_sigs[-1] - err_sigs[-2])
                * additional_sigma
            )
            upper_bounds[-1].append(
                upper_bounds[-1][-1]
                + (upper_bounds[-1][-1] - upper_bounds[-1][-2])
                / (err_sigs[-1] - err_sigs[-2])
                * additional_sigma
            )

            lower_bounds[-1] = np.array(lower_bounds[-1])
            upper_bounds[-1] = np.array(upper_bounds[-1])

        err_sigs = np.concatenate([err_sigs, np.array([100.0])])

        # Gerenate UQ parity plots for each evaluation target
        for target in eval_targets:
            plt.plot(
                np.sqrt(pred_errors[target]), np.sqrt(errors[target]), ".", alpha=0.25
            )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(
                np.sqrt(pred_errors[target]).min() / 5,
                np.sqrt(pred_errors[target]).max() * 5,
            )
            plt.ylim(
                np.sqrt(errors[target]).min() / 5, np.sqrt(errors[target]).max() * 5
            )
            plt.plot([1e-50, 1e10], [1e-50, 1e10], "k--", zorder=0, lw=1)
            for i, desired_fraction in enumerate(desired_fractions):
                plt.plot(err_sigs, lower_bounds[i], color="gray", zorder=0, lw=0.5)
                plt.plot(err_sigs, upper_bounds[i], color="gray", zorder=0, lw=0.5)
            plt.title(f"LLPR UQ Plot for {target}")
            plt.xlabel("Predicted Errors")
            plt.ylabel("Actual Errors")
            plt.savefig(args.model.split(".model")[0] + "_" + target + "_LLPR_UQ.pdf")
            plt.clf()


def _ssl_wrapper(x, model_llpr, ll_feats_all, errors, sigma, bin_size):
    model_llpr.compute_inv_covariance(10 ** x[0], sigma)  # needs to be tested
    pred_errors = torch.einsum(
        "ij, jk, ik -> i", ll_feats_all, model_llpr.inv_covariance, ll_feats_all
    )
    obj_value = _ssl(errors, pred_errors, bin_size)
    return obj_value


def _ssl(errors, pred_errors, bin_size=1):
    # Calculates sum of squared log errors on the energy for a dataset
    sort_indices = torch.argsort(pred_errors)
    errors_sorted = errors[sort_indices]
    pred_errors_sorted = pred_errors[sort_indices]

    n_samples = len(errors)

    error_bins = []
    pred_error_bins = []

    for i_bin in range(n_samples // bin_size - 1):
        error_bins.append(errors_sorted[i_bin * bin_size : (i_bin + 1) * bin_size])
        pred_error_bins.append(
            pred_errors_sorted[i_bin * bin_size : (i_bin + 1) * bin_size]
        )

    error_bins = torch.stack(error_bins)
    pred_error_bins = torch.stack(pred_error_bins)

    error_means = error_bins.mean(dim=1)
    pred_error_means = pred_error_bins.mean(dim=1)

    squared_log_errors = torch.log(error_means / pred_error_means) ** 2

    sum_squared_log = squared_log_errors.sum().item()

    return sum_squared_log


def _pdf(x, sigma):
    return x * np.exp(-(x**2) / (2 * sigma**2)) * 1.0 / (sigma * np.sqrt(2 * np.pi))


def _find_where_pdf_is_c(c, sig):
    mode_value = _pdf(sig, sig)
    if c > mode_value:
        raise ValueError("c must be less than mode_value")
    where_below_mode = root_scalar(
        lambda x: _pdf(x, sig) - c,
        bracket=[0, sig],
    ).root
    where_above_mode = root_scalar(
        lambda x: _pdf(x, sig) - c,
        bracket=[sig, 100],
    ).root
    return where_below_mode, where_above_mode


def _pdf_integral(sig, c):
    x1, x2 = _find_where_pdf_is_c(c, sig)
    return np.exp(-(x1**2) / (2 * sig**2)) - np.exp(-(x2**2) / (2 * sig**2))


def _find_fraction(sig, fraction):
    mode_value = _pdf(sig, sig)
    return root_scalar(
        lambda x: _pdf_integral(sig, x) - fraction,
        x0=mode_value - 0.01,
        x1=mode_value - 0.02,
    ).root


if __name__ == "__main__":
    main()
