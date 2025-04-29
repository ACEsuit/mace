###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, ncols=55)

import dataclasses
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .torch_tools import to_numpy
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    LRScheduler
)


def print_gpu_memory():
    if torch.cuda.is_available():
        # Current memory stats
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Peak memory stats
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2
        
        print(f"Current Allocated: {allocated:.2f} MB")
        print(f"Current Reserved/Cached: {reserved:.2f} MB")
        print(f"Peak Allocated: {max_allocated:.2f} MB")
        print(f"Peak Reserved/Cached: {max_reserved:.2f} MB")

def detect_nan_gradients(model):
    """
    Check all parameters in the model for NaN gradients and report the layer names.
    
    Args:
        model: PyTorch model
        
    Returns:
        bool: True if NaN gradients were found, False otherwise
    """
    found_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient found in layer: {name}")
            found_nan = True
    return found_nan

def detect_nan_parameters(model):
    found_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN param found in layer: {name}")
            found_nan = True
    return found_nan

def detect_nan_output(output):
    found_nan = False

    for k, v in output.items():
        if torch.isnan(v).any():
            print(f"NaN param found in output: {k}")
            found_nan = True
    return found_nan
    

@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress={error_stress:.1f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomRMSE+EMAEstressvirials" and eval_metrics["rmse_stress"] is not None
    ):  
        error_e_rmse = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f_rmse = eval_metrics["rmse_f"] * 1e3
        error_stress_rmse = eval_metrics["rmse_stress"] * 1e3
        error_e_mae = eval_metrics["mae_e_per_atom"] * 1e3
        error_f_mae = eval_metrics["mae_f"] * 1e3
        error_stress_mae = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, \t RMSE_E_per_atom={error_e_rmse:.1f} meV, RMSE_F={error_f_rmse:.1f} meV / A, RMSE_stress={error_stress_rmse:.1f} meV / A^3"
        )
        logging.info(
            f"                                                                 \t MAE_E_per_atom={error_e_mae:.1f} meV, MAE_F={error_f_mae:.1f} meV / A, MAE_stress={error_stress_mae:.1f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} mDebye"
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"head: {valid_loader_name}, Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
        )


def evaluate_on_train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    logger: MetricsLogger,
    distributed: bool = False,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
):
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    for batch in train_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux
    

    logging.info("Training complete")

def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler, 
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    restart: Optional[str] = None,
    log_opt: Optional[bool] = False,
    async_update: Optional[bool] = False,
    #refit_e0s: Optional[bool] = False,
    rsmooth: Optional[float] = None,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    #if restart is not None:
    #    loss_history = {}  # loss history per batch

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")
    logging.info("Started training")
    epoch = start_epoch

    # log validation loss before _any_ training
    valid_loss = 0.0
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch and lr_scheduler.step_unit == "epoch": # only when epoch based LRScheduler
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
            logging.info(f"latest lr --> {lr_scheduler.get_last_lr()}")
        else:
            if swa_start:
                logging.info("Changing loss based on SWA")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch and swa.scheduler.step_unit == "epoch":
                swa.scheduler.step()
        

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed_model=distributed_model,
            rank=rank,
            restart=restart,
            log_opt=log_opt,
            async_update=async_update,
            rsmooth=rsmooth,
            lr_scheduler=lr_scheduler,
        )

        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            with param_context:
                valid_loss = 0.0
                wandb_log_dict = {}
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                    )
                    valid_loss += valid_loss_head

                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics["rmse_e_per_atom"],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }

            if rank == 0:
                if log_wandb:
                    wandb.log(wandb_log_dict)

                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience and (
                        swa is not None and epoch < swa.start
                    ):
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement and starting swa"
                        )
                        epoch = swa.start
                    elif patience_counter >= patience and (
                        swa is None or epoch >= swa.start
                    ):
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters() if ema is not None else nullcontext()
                        )
                        with param_context:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    restart: Optional[str] = None,
    log_opt: Optional[bool] = False,
    async_update: Optional[bool] = False,
    rsmooth: Optional[float] = None, 
    lr_scheduler: Optional[LRScheduler] = None,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model
    

    if rank == 0:
        data_iter = tqdm(data_loader)
    else:
        data_iter = data_loader

    for idx, batch in enumerate(data_iter):
        _, opt_metrics = take_step(
            model=model_to_train,
            loss_fn=loss_fn,
            batch=batch,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            restart=restart,
            log_opt=log_opt,
            async_update=async_update,
            rsmooth=rsmooth,
        )
        if lr_scheduler and lr_scheduler.step_unit == "step":
            lr_scheduler.step()
            #print(lr_scheduler.lr_scheduler.get_last_lr())

        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)

def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    restart: Optional[str] = None,
    log_opt: Optional[bool] = False,
    async_update: Optional[bool] = False,
    rsmooth: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:

    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()

       

    if rsmooth is not None:
        rsmooth = float(rsmooth)
        batch_dict['positions'] = batch_dict['positions'] + torch.rand_like(batch_dict['positions']) * rsmooth
    
    if detect_nan_parameters(model):
        print(f"NaN params detected at iteration")
        # Optional: you can break or take other actions here


    #with torch.autograd.detect_anomaly():
    if async_update:
        with model.no_sync():
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            loss = loss_fn(pred=output, ref=batch)
            loss.backward()
    else:
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )

        loss = loss_fn(pred=output, ref=batch)
        loss.backward()

        #unused = []
        #for name, param in model.named_parameters():
        #    if param.grad is None:
        #        unused.append(name)

        #import ipdb; ipdb.set_trace()


    #if restart == "batch":
    #    import ipdb; ipdb.set_trace()
    #    loss_history = None
    #    spike_flag = detect_spike(loss.item(), loss_history)

    #    if spike_flag:
    #        print(f"loss: {loss.item()}; loss history {loss_history} --> spike detected")
    #        print(f"skip batch")
    #        
    #        loss_dict = {
    #            "loss": to_numpy(loss),
    #            "time": time.time() - start_time,
    #        }
    #        return loss, loss_dict

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    
    #import ipdb; ipdb.set_trace()

    optimizer.step()

    

    if async_update:
        # sync optimizer
        optimizer = sync_optimizer(optimizer)

        # sync weights
        model = sync_model(model)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    #print(to_numpy(loss))
    #print_gpu_memory()

    if log_opt:
        elem_count = torch.count_nonzero(batch.node_attrs, dim=0)
        torch.distributed.all_reduce(elem_count, op=torch.distributed.ReduceOp.SUM)
        total_graphs = (len(batch.ptr) - 1) * torch.distributed.get_world_size()
        total_nodes = torch.tensor(batch.batch.size(0), device=batch.batch.device)
        torch.distributed.all_reduce(total_nodes, op=torch.distributed.ReduceOp.SUM)
        total_nodes = total_nodes.item()

        
        exp_avg =  {k:v['exp_avg'] for k,v in optimizer.state.items()}
        exp_avg_sq = {k:v['exp_avg_sq'] for k,v in optimizer.state.items()}
        eps = optimizer.defaults['eps']
        
        if optimizer.defaults['amsgrad']:
            max_exp_avg_sq = {k:v['max_exp_avg_sq'] for k,v in optimizer.state.items()}
            update = {k:exp_avg[k] / (max_exp_avg_sq[k] + eps) for k,v in optimizer.state.items()}
        else:
            update = {k:exp_avg[k] / (exp_avg_sq[k] + eps) for k,v in optimizer.state.items()}

        exp_avg_stats = {}
        max_exp_avg_sq_stats = {}
        exp_avg_sq_stats = {}
        update_stats = {}

        for k,v in model.named_parameters():
            exp_avg_sq_stats[k] = {"mean": exp_avg_sq[v].mean().item(), "std": exp_avg_sq[v].std().item()}
            if optimizer.defaults['amsgrad']:
                max_exp_avg_sq_stats[k] = {"mean": max_exp_avg_sq[v].mean().item(), "std": max_exp_avg_sq[v].std().item()}
            exp_avg_stats[k] = {"mean": exp_avg[v].mean().item(), "std": exp_avg[v].std().item()}
            update_stats[k] = {"mean": update[v].mean().item(), "std": update[v].std().item()}

        loss_dict['total_graphs'] = total_graphs
        loss_dict['total_nodes'] = total_nodes
        loss_dict['elem_count'] = elem_count
        loss_dict['exp_avg_sq_stats'] = exp_avg_sq_stats
        if optimizer.defaults['amsgrad']:
            loss_dict['max_exp_avg_stats'] = max_exp_avg_sq_stats
        loss_dict['exp_avg_stats'] = exp_avg_stats
        loss_dict['update_stats'] = update_stats

    return loss, loss_dict

def sync_optimizer(optimizer):
    state_dict = optimizer.state
    for k, v in state_dict.items():
        sync_state_dict(v)

def sync_model(model):
    state_dict = model.state_dict()
    sync_state_dict(state_dict)

def sync_state_dict(state_dict):
    for k,v in state_dict.items():
        if v.device.type.startswith("cpu"):
            continue
        if not v.is_contiguous():
            continue
        try:
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.AVG)
        except:
            import ipdb; ipdb.set_trace()

def detec_spike(loss, loss_history):
    pass

def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        avg_loss, aux = metrics(batch, output)



    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):
        aux = {}
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)

        return aux["loss"], aux
