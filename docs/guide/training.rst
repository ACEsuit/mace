.. _training:

========
Training
========

Script
------

To train a MACE model, you can use the `run_train.py` script (note that if you used `pip install` to install
mace, you can also use the executable `mace_run_train` entry point which should be in your path).

.. code-block:: bash

    python <mace_repo_dir>/mace/cli/run_train.py \
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

Training files
---------------

To train a MACE model, you will use the `run_train.py` command which takes the following arguments:

First specify the name of your model and final log file using the `--name` flag.

You can specify the training file with the `--train_file` flag.
The validation set can either be specified as a separate file using the `--valid_file` keyword, or it can be specified as a fraction of the training set using the `--valid_fraction` keyword.
The validation set is not used for optimizing the model but to estimate the model accuracy during training.

It is also possible to provide a test set using the `--test_file` keyword. This set is entirely independent and only gets evaluated at the end of the training process.


Model
-----

Model options
^^^^^^^^^^^^^^

The `--model` flag specifies the type of model to be trained. The vanilla MACE model is specified by `--model="MACE"`.
The `--model="ScaleShiftMACE"` model includes a residual connection at first, which will usually improve the model's accuracy but will make the model output incorrect isolated atoms energies.
Use this model if you are not interested in bond-breaking energies.
To train a model on dipole moments, use `--model="AtomicDipolesMACE"`. If you want to also train simultaneously on energies, use `--model="EnergyDipolesMACE"`.

The Messages
^^^^^^^^^^^^

Change `-hidden_irreps` to control the model size. For most applications, the recommended default model size is `--hidden_irreps='256x0e'` (meaning 256 invariant messages) or `--hidden_irreps='128x0e + 128x1o'` (meaning 128 equivariant messages). If the model is not accurate enough, you can include higher order features, e.g., `128x0e + 128x1o + 128x2e`, or increase the number of channels to `256`.
The number of message passing layers can be controlled via the `--num_ineractions` parameter. **Increasing the model size and the number of layers will lead to more accurate but slower models.**

Correlation order
^^^^^^^^^^^^^^^^^

MACE uses a body order expansion on the site energy:

:math:`E_{i} = E^{(0)}_{i} + \sum_{j} E_{ij}^{(1)} + \sum_{jk} E_{ijk}^{(2)} + ...`

The correlation order corresponds to the order that MACE induces at each layer. Choosing `--correlation=3` will create basis function of up to 4-body (ijke) indices, for each layer. Because of the multiple layers of MACE, the total correlation order is much higher. A two layers mace, with `--correlation=3` has a total body order of 13.

Angular resolution
^^^^^^^^^^^^^^^^^^

The angular resolution describes how precise the model can identify angles. This is controled by `l_max`. The higher this integer, more precise is the angular resolution. Larger value will result in more accurate but slower models. The default is `l_max=3`.

Cutoff radius
^^^^^^^^^^^^^

The cutoff radius controls the locality of the model. A `--r_max=3.0` means that the model assumes atoms seperated by a distance of more than 3.0 A do not directly `communicate`. Because the model has two layers, atoms further than 3.0 A can still `communicate` by proxy. The actual receptive field of the model is the number of layers times the cutoff distance.

Reference energies
------------------

It is usually preferred to add the isolated atoms to the training set, rather than reading in their energies through the command line like in the example above.
To label them in the training set, set `config_type=IsolatedAtom` in their info fields.
If you prefer not to use or do not know the energies of the isolated atoms, you can use the option `--E0s="average"` which estimates the atomic energies using least squares regression.

SWA and EMA
-----------

If the keyword `--swa` is enabled, the energy weight of the loss is increased for the last ~20% of the training epochs (from `--start_swa` epochs).
This setting usually helps lower the energy errors.


Data keys
---------

When parsing the data files, the energies are read using the keyword `energy` and the forces using the keyword `forces`. To change that, specify the `--energy_key` and `--forces_key`.
You can also specify `--stress_key` to read the stress tensor, `--virials_key` to read the virial tensor, and `--dipole_key` to read the dipole moments.

Float precision
---------------

The precision can be changed using the keyword ``--default_dtype``, the default is `float64` but `float32` gives a significant speed-up (usually a factor of x2 in training).


Set batch size
--------------

The keywords ``--batch_size`` and ``--max_num_epochs`` should be adapted based on the size of the training set.
The batch size should be increased when the number of training data increases, and the number of epochs should be decreased.
An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $\text{max-num-epochs}*\frac{\text{num-configs-training}}{\text{batch-size}}$.

Validation parameters
---------------------

The validation set controls the stopping of the training. At each `--eval_interval` the model is tested on the validation set. We also evaluate the set by batch size, controlled by `--valid_batch_size`. If the accuracy of the model stops improving on the validation set for `--patience` number of epochs. This is called **early stopping**.


Heterogeneous labels
--------------------

The code can handle training set with heterogeneous labels, for example containing both bulk structures with stress and isolated molecules.
In this example, to make the code ignore stress on molecules, append to your molecules configuration a ``config_stress_weight = 0.0``.


Devices
-------

To use GPUs, specify ``--device=cuda``.
To use CPUs, specify ``--device=cpu``.
To use Apple Silicon GPU acceleration make sure to install the latest PyTorch version and specify ``--device=mps``.

Checkpoints
-----------

For trainings that require restarting, you can continue the fitting from the last checkpoint by using the flag `--restart_latest`. The checkpoint saves the best model that currently has been trained. All checkpoints are saved in ./checkpoints folder. We can also continue from a restart when extending the dataset or changing any hyperparameters that do not affect the model size.
