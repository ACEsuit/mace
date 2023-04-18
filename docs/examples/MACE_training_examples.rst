==============================================
Example scripts for training MACE models
==============================================

This page collects various training scripts for training MACE models in the paper "Evaluation of the MACE Force Field Architecture: from Medicinal Chemistry to Materials Science"

######################
MD22: large molecules
######################

The MD22 dataset (http://www.sgdml.org) contains configurations with energies and forces computed at the DFT level of QM for 7 different large molecular systems. In the paper we used a very large MACE model to showcase primarily the achievable accuracy with a model that is still usefully fast. 

An example script for training the 2-layer model on the carbon nanotube (largest) system is given below

.. code-block:: shell 

    python /PATH/TO/MACE/mace/scripts/run_train.py \
        --name="nanotube_large_r55" \
        --train_file="nanotube_large.xyz" \
        --valid_fraction=0.05 \
        --test_file="nanotube_test.xyz" \
        --E0s="average" \
        --model="MACE" \
        --num_interactions=2 \
        --num_channels=256 \
        --max_L=2 \
        --correlation=3 \
        --r_max=5.0 \
        --forces_weight=1000 \
        --energy_weight=10 \
        --batch_size=2 \
        --valid_batch_size=2 \
        --max_num_epochs=650 \
        --start_swa=450 \
        --scheduler_patience=5 \
        --patience=15 \
        --eval_interval=3 \
        --ema \
        --swa \
        --swa_forces_weight=10 \
        --error_table='PerAtomMAE' \
        --default_dtype="float64"\
        --device=cuda \
        --seed=123 \
        --restart_latest \
        --save_cpu 

In comparison, the single layer model uses ``max_L=0``, because there is no equivariant message to be passed. this model is considerably (ca 10x) faster, and somewhat less accurate as shown in the paper. 

.. code-block:: shell

    python /PATH/TO/MACE/mace/scripts/run_train.py \
        --name="nano_large_r6" \
        --train_file="nanotube_large.xyz" \
        --valid_fraction=0.05 \
        --test_file="nanotube_test.xyz" \
        --E0s="average" \
        --model="MACE" \
        --num_interactions=1 \
        --num_channels=256 \
        --max_L=0 \
        --correlation=3 \
        --r_max=6.0 \
        --forces_weight=1000 \
        --energy_weight=10 \
        --batch_size=4 \
        --valid_batch_size=8 \
        --max_num_epochs=1000 \
        --start_swa=600 \
        --scheduler_patience=5 \
        --patience=15 \
        --eval_interval=3 \
        --ema \
        --swa \
        --swa_forces_weight=10 \
        --error_table='PerAtomMAE' \
        --default_dtype="float64"\
        --device=cuda \
        --seed=123 \
        --restart_latest \
        --save_cpu

###########################################
ANI-1x dataset: H, C, N, O transferable FF
###########################################

We used the subset of the ANI-1x datatset (https://www.nature.com/articles/s41597-020-0473-z) that also has couple cluster reference data to train 3 transferable MACE models of increaing accuracy. 

To train MACE on large datasets one can preprocess the data and use on the fly data loading. This option is currently available on the multi-GPU branch of MACE. 

.. code-block:: shell
    python /PATH/TO/MACE/mace/scripts/preprocess_data.py \
        --train_file="ani1x_cc_dft.xyz" \
        --valid_fraction=0.03 \
        --energy_key="DFT_energy" \
        --forces_key="DFT_forces" \
        --r_max=5.0 \
        --h5_prefix="ANI1x_cc_DFT_rc5_" \
        --compute_statistics \
        --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}" \
        --seed=12345

This produces 3 files: ANI1x_cc_DFT_rc5_train.h5, ANI1x_cc_DFT_rc5_valid.h5, ANI1x_cc_DFT_rc5_statistics.json. The statistics file contains the mean and standard deviation of the energies and forces, which are used to normalize the data as well as other statistics like the cutoff and average number of neighbours used for internal normailsation of the model. For the smallest model we used ``r_max=4.5`` and for the medium and large models ``r_max=5.0``.

The training script for the smallest model is given below.
.. code-block:: shell
    python /PATH/TO/MACE/mace/scripts/run_train.py \
        --name="ani500k_small" \
        --train_file="ANI1x_cc_DFT_rc5_train.h5" \
        --valid_file="ANI1x_cc_DFT_rc5_valid.h5" \
        --statistics_file="ANI1x_cc_DFT_rc5_statistics.json" \
        --E0s="{1: -13.62222753701504, 6: -1029.4130839658328, 7: -1484.8710358098756, 8: -2041.8396277138045}" \
        --model="MACE" \
        --num_interactions=2 \
        --num_channels=64 \
        --max_L=0 \
        --correlation=3 \
        --r_max=5.0 \
        --forces_weight=1000 \
        --energy_weight=40 \
        --weight_decay=1e-7 \
        --clip_grad=1.0 \
        --batch_size=128 \
        --valid_batch_size=128 \
        --max_num_epochs=500 \
        --scheduler_patience=20 \
        --patience=50 \
        --eval_interval=1 \
        --ema \
        --swa \
        --start_swa=250 \
        --swa_lr=0.00025 \
        --swa_forces_weight=10 \
        --num_workers=32 \
        --error_table='PerAtomMAE' \
        --default_dtype="float64"\
        --device=cuda \
        --seed=123 \
        --restart_latest \
        --save_cpu \

The model can easily be transfer learned to CC level of theory. For this the preprocesing has to be repated with the CC energies. Than the training can simply be continued. Suince the CC data does not have forces it is crucial to deactivate scaling by the RMS of the forces by setting ``scaling=no_scaling``. 