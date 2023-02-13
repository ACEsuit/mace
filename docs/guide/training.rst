.. _training:

========
Training
========

Script
------

To train a MACE model, you can use the `run_train.py` script:

.. code-block:: bash

    python ./mace/scripts/run_train.py \
        --name="MACE_model" \
        --train_file="train.xyz" \
        --valid_fraction=0.05 \
        --test_file="test.xyz" \
        --config_type_weights='{"Default":1.0}' \
        --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
        --model="MACE" \
        --hidden_irreps='128x0e + 128x1o' \
        --r_max=5.0 \
        --batch_size=10 \
        --max_num_epochs=1500 \
        --swa \
        --start_swa=1200 \
        --ema \
        --ema_decay=0.99 \
        --amsgrad \
        --restart_latest \
        --device=cuda \


Set validation
--------------

To give a specific validation set, use the argument `--valid_file`. 
To set a larger batch size for evaluating the validation set, specify `--valid_batch_size`. 

Model size
----------

To control the model's size, you need to change `--hidden_irreps`. 
For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'`. If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`. 

Reference energies
------------------

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above. 
To label them in the training set, set `config_type=IsolatedAtom` in their info fields. 
If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression. 

SWA and EMA
-----------

If the keyword `--swa` is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_swa` epochs). 
This setting usually helps lower the energy errors. 

Float precision
---------------

The precision can be changed using the keyword ``--default_dtype``, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).


Set batch size
--------------

The keywords ``--batch_size`` and ``--max_num_epochs`` should be adapted based on the size of the training set. 
The batch size should be increased when the number of training data increases, and the number of epochs should be decreased. 
An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

Heterogeneous labels
--------------------

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules. 
In this example, to make the code ignore stress on molecules, append to your molecules configuration a ``config_stress_weight = 0.0``.


Devices
-------

To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify ``--device=mps``. 