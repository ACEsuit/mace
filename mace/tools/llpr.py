import torch
import numpy as np
from scipy.optimize import brute
import math


def calibrate_llpr_params(
    model,
    validation_loader,
    function="ssl",
    calib_bound=5,
    calib_delta=0.1,
    **kwargs,
):
    # This function optimizes the calibration parameters for LLPR on the validation set
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>

    if function == "ssl":
        obj_function = _sum_squared_log
    elif function == "nll":
        obj_function = _avg_nll_regression
    else:
        raise RuntimeError("Unsupported objective function type for LLPR uncertainty calibration!")

    actual_errors = []
    ll_feats = []
    num_atoms = []
    # Compute model predictions, actual errors, ll feats
    for batch in validation_loader:
        batch = batch.to(next(model.parameters()).device)
        batch_dict = batch.to_dict()
        y = batch_dict['energy']
        num_graphs = batch_dict["ptr"].numel() - 1
        num_atoms.append(batch_dict['ptr'][1:] - batch_dict['ptr'][:-1])
        model_outputs = model(batch_dict)
        predictions = model_outputs['energy'].detach()
        cur_ll_feats = model.aggregate_ll_features(
            model_outputs["node_feats"], batch_dict["batch"], num_graphs
        ).detach()
        ll_feats.append(cur_ll_feats)
        actual_errors.append((y - predictions)**2)

    actual_errors = torch.cat(actual_errors, dim=0)
    ll_feats_all = torch.cat(ll_feats, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)

    def obj_function_wrapper(x):
        x = _process_inputs(x)
        try:
            model.compute_inv_covariance(*x)
            predicted_errors = torch.einsum(
                "ij, jk, ik -> i",
                ll_feats_all,
                model.inv_covariance,
                ll_feats_all
            )
            obj_value = obj_function(actual_errors, predicted_errors, **kwargs)
        except torch._C._LinAlgError:
            obj_value = 1e10
        if math.isnan(obj_value):
            obj_value = 1e10
        return obj_value
    calib_slice = slice(-1*calib_bound, calib_bound+0.01, calib_delta)
    result = brute(obj_function_wrapper, ranges=[calib_slice, calib_slice])

    # warn if we hit the edge of the parameter space
    if result[0] <= -5 or result[0] >= 5 or result[1] <= -5 or result[1] >= 5:
        print("Optimal parameters found beyond the designated parameter space!")

    print(f"Calibrated LLPR parameters:\tC = {10**result[0]:.4E}\tsigma = {10**result[1]:.4E}")
    model.compute_inv_covariance(*(_process_inputs(result)))


def _process_inputs(x):
    x = list(x)
    x = [10**single_x for single_x in x]
    return x


def _avg_nll_regression(actual_errors, predicted_errors, energy_shift=0.0, energy_scale=1.0):
    # This function calculates the negative log-likelihood on the energy for a dataset
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>
    total_nll = (
        actual_errors / predicted_errors + torch.log(actual_errors) + np.log(2*np.pi)
        ).sum().item() * 0.5
    return total_nll / len(actual_errors)


def _sum_squared_log(actual_errors, predicted_errors, n_samples_per_bin=1):
    # This function calculates the sum of squared log errors on the energy for a dataset
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>
    sort_indices = torch.argsort(predicted_errors)
    actual_errors_sorted = actual_errors[sort_indices]
    predicted_errors_sorted = predicted_errors[sort_indices]

    n_samples = len(actual_errors)

    actual_error_bins = []
    predicted_error_bins = []

    # skip the last bin for incompleteness
    for i_bin in range(n_samples // n_samples_per_bin - 1):
        actual_error_bins.append(
            actual_errors_sorted[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )
        predicted_error_bins.append(
            predicted_errors_sorted[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )

    actual_error_bins = torch.stack(actual_error_bins)
    predicted_error_bins = torch.stack(predicted_error_bins)

    # calculate means:
    actual_error_means = actual_error_bins.mean(dim=1)
    predicted_error_means = predicted_error_bins.mean(dim=1)

    # calculate squared log errors:
    squared_log_errors = (
        torch.log(actual_error_means / predicted_error_means)**2
    )

    # calculate the sum of squared log errors:
    sum_squared_log = squared_log_errors.sum().item()

    return sum_squared_log
