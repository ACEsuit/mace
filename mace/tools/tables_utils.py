import logging
from typing import Dict, List, Optional

import torch
from prettytable import PrettyTable

from mace.tools import evaluate


def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == "train":
        return (0, key)
    if key == "valid":
        return (1, key)
    return (2, key)


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
    skip_heads: Optional[List[str]] = None,
) -> PrettyTable:
    if log_wandb:
        import wandb
    skip_heads = skip_heads or []
    table = PrettyTable()
    if table_type == "TotalRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSEstressvirials":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "PerAtomMAEstressvirials":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
            "MAE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "TotalMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "PerAtomMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / mDebye / atom",
            "relative MU RMSE %",
        ]
    elif table_type == "DipoleMAE":
        table.field_names = [
            "config_type",
            "MAE MU / mDebye / atom",
            "relative MU MAE %",
        ]
    elif table_type == "DipolePolarRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / me A / atom",
            "relative MU RMSE %",
            "RMSE ALPHA e A^2 / V / atom",
        ]
    elif table_type == "EnergyDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "rel F RMSE %",
            "RMSE MU / mDebye / atom",
            "rel MU RMSE %",
        ]

    for name in sorted(all_data_loaders, key=custom_key):
        if any(skip_head in name for skip_head in skip_heads):
            logging.info(f"Skipping evaluation of {name} (in skip_heads list)")
            continue
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()

        del data_loader
        torch.cuda.empty_cache()
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)
        if table_type == "TotalRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                    f"{metrics['rmse_stress'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                    f"{metrics['rmse_virials'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomMAEstressvirials"
            and metrics["mae_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                    f"{metrics['mae_stress'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomMAEstressvirials"
            and metrics["mae_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                    f"{metrics['mae_virials'] * 1000:8.1f}",
                ]
            )
        elif table_type == "TotalMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.2f}",
                    f"{metrics['rel_rmse_mu']:8.1f}",
                ]
            )
        elif table_type == "DipoleMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_mu_per_atom'] * 1000:8.2f}",
                    f"{metrics['rel_mae_mu']:8.1f}",
                ]
            )
        elif table_type == "DipolePolarRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:.2f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                    f"{metrics['rmse_polarizability_per_atom'] * 1000:.2f}",
                ]
            )
        elif table_type == "EnergyDipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.1f}",
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_mu']:8.1f}",
                ]
            )
    return table
