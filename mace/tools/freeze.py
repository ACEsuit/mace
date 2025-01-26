import torch
import logging

# change: freeze layers
def freeze_layers(model, n):
    layers = list(model.children())
    num_layers = len(layers)
    logging.info(f"N of model layers: {num_layers}")
    if n < 0:
        frozen_layers = layers[n:]
    else:
        frozen_layers = layers[:n]

    logging.info(f"N of frozen layers: {len(frozen_layers)}")

    for layer in frozen_layers:
        for param in layer.parameters():
            param.requires_grad = False

# or freeze parameter tensors 
def freeze_param(model, n):
    par_ten = list(model.named_parameters())
    num_par = len(par_ten)
    logging.info(f"N of named parameters: {num_par}")
    if n < 0: 
        index = n + num_par - 1
        for idx, (name, param) in enumerate(model.named_parameters()):
            if idx > index:
                    param.requires_grad = False
            else:
                param.requires_grad = True     
    else: 
        index = n - 1
        for idx, (name, param) in enumerate(model.named_parameters()):
            if idx > index:
                param.requires_grad = True
            else:
                param.requires_grad = False
