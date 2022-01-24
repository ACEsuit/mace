import dataclasses
import logging
import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch.nn.functional
import torch_geometric
from e3nn import o3
from torch.optim.swa_utils import AveragedModel, SWALR
from torch_ema import ExponentialMovingAverage

from LieACE import data, tools, modules


@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


def get_dataset(downloads_dir: str, dataset: str, subset: Optional[str], split: Optional[int]) -> SubsetCollection:
    if dataset == 'iso17':
        ref_configs, test_within, test_other = data.load_iso17(directory=downloads_dir)
        train_size, valid_size = 5000, 500
        train_valid_configs = np.random.default_rng(1).choice(ref_configs, train_size + valid_size)
        train_configs, valid_configs = train_valid_configs[:train_size], train_valid_configs[train_size:]
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[('test_within', test_within), ('test_other', test_other)])

    if dataset == 'rmd17':
        if not subset or not split:
            raise RuntimeError('Specify subset and split')
        train_valid_configs, test_configs = data.load_rmd17(directory=downloads_dir, subset=subset, split=split)
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs,
                                                                     valid_fraction=0.05,
                                                                     seed=1)
        return SubsetCollection(train=train_configs, valid=valid_configs, tests=[('test', test_configs)])

    if dataset == '3bpa':
        if not subset:
            raise RuntimeError('Specify subset')
        configs_dict = data.load_3bpa(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs,
                                                                     valid_fraction=0.10,
                                                                     seed=1)
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[(key, configs_dict[key])
                                       for key in ['test_300K', 'test_600K', 'test_1200K', 'test_dih']])

    if dataset == 'acac':
        if not subset:
            raise RuntimeError('Specify subset')
        configs_dict = data.load_acac(directory=downloads_dir)
        train_valid_configs = configs_dict[subset]
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs,
                                                                     valid_fraction=0.10,
                                                                     seed=1)
        return SubsetCollection(train=train_configs,
                                valid=valid_configs,
                                tests=[(key, configs_dict[key]) for key in ['test_MD_300K', 'test_MD_600K']])

    if dataset == 'ethanol':
        configs_dict = data.load_ethanol(directory=downloads_dir)
        train_valid_configs = configs_dict['train']
        train_configs, valid_configs = data.random_train_valid_split(items=train_valid_configs,
                                                                     valid_fraction=0.05,
                                                                     seed=1)
        return SubsetCollection(train=train_configs, valid=valid_configs, tests=[('test_MD', configs_dict['test_MD'])])

    raise RuntimeError(f'Unknown dataset: {dataset}')


def main() -> None:
    
    args = Config.from_file(r'D:\Equivarient_GNN\ACE_recu_torch\configs\example.yaml') #yaml file
    
    torch.set_default_dtype(torch.float64)


    device = tools.init_device(args.device)
    

    tag = tools.get_tag(name=args.name, seed=args.seed)
    
    train_valid_configs = tools.config.load_xyz(args.dataset_file_name)
    train_configs, valid_configs = train_valid_configs[:args.n_train], train_valid_configs[args.n_val:]
    
    test_configs = []
    if 'dataset_file_name_test' in args.keys() :
        test_configs = tools.config.load_xyz(args.dataset_file_name_test)
    
    logging.info(f'Number of configurations: train={len(train_configs)}, valid={len(valid_configs)}, '
                 f'test={len(test_configs)}')
    
    z_table = data.get_atomic_number_table_from_zs(
        z
        for configs in (train_configs, valid_configs)
        for config in configs
        for z in config.atomic_numbers
    )
    
    logging.info(z_table)
    
    degrees = data.species_degrees(args.degrees)
    
    atomic_energies = np.array(args.atomic_energies)
    
    train_loader, valid_loader = (
        data.get_data_loader(
            dataset=[data.AtomicData.from_config(config, z_table=z_table, cutoff=args.r_cut) for config in configs],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        for configs in (train_configs, valid_configs)
    )
    
    
    loss_fn: torch.nn.Module
    if args.no_forces:
        loss_fn = modules.loss.EnergyLoss()
    else:
        loss_fn = modules.loss.EnergyForcesLoss(energy_weight=args.energy_weight, forces_weight=args.forces_weight)
    logging.info(loss_fn)
    
    
     # Build model
    logging.info('Building model')
    model = models.invariant_multi_ACE(
        r_cut=args.r_cut,
        degrees = degrees,
        num_polynomial_cutoff=args.num_cutoff_basis,
        num_layers=args.num_layers,
        num_elements=len(z_table),
        atomic_energies=atomic_energies,
        include_forces=True,
        non_linear = args.non_linear,
        device = device,
    )
    
    model.to(device)
    
    
    logging.info(model)
    logging.info(f'Number of parameters: {utils.count_parameters(model)}')
    
    optimizer = tools.get_optimizer(name=args.optimizer, learning_rate=args.lr, parameters=model.parameters())
    logger = tools.ProgressLogger(directory=args.results_dir, tag=tag)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lr_scheduler_gamma,verbose=True)
    #lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = 100, eta_min=5e-6, last_epoch=-1, verbose=True)
    checkpoint_handler = tools.checkpoint.CheckpointHandler(directory=args.checkpoints_dir, tag=tag, keep=args.keep_checkpoints)

    
    start_epoch = 0
        
    if args.restart_latest:
        start_epoch = checkpoint_handler.load_latest(state=tools.CheckpointState(model, optimizer, lr_scheduler))
    
    logging.info(f'Optimizer: {optimizer}')
    
    if device.type == 'cuda' : 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    utils.train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_handler=checkpoint_handler,
        eval_interval=args.eval_interval,
        start_epoch=start_epoch,
        max_num_epochs=args.max_num_epochs,
        logger=logger,
        patience=args.patience,
        device=device,
    )

    # Evaluation on test dataset
    epoch = checkpoint_handler.load_latest(state=utils.CheckpointState(model, optimizer, lr_scheduler))
    """if args.dataset_file_name_test:
        test_loss, test_metrics = utils.evaluate(model, loss_fn=loss_fn, data_loader=test_loader, device=device)
        test_metrics['mode'] = 'test'
        test_metrics['epoch'] = epoch
        logger.log(test_metrics)
        logging.info(f'Test loss (epoch {epoch}): {test_loss:.3f}')"""

    # Save entire model
    model_path = os.path.join(args.checkpoints_dir, tag + '.model')
    logging.info(f'Saving model to {model_path}')
    torch.save(model, model_path)

    logging.info('Done')
    
    
    
if __name__ == '__main__':
    main()    


