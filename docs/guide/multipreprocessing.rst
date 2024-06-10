.. _multipreprocessing:

============
Large Dataset Pre-processing
============


If you have a large dataset that might not fit into the GPU memory it is recommended to preprocess the data on a CPU and use on-line dataloading for training the model. 
To preprocess your dataset specified as an xyz file run the `preprocess_data.py` script. 
An example is given here:

.. code-block:: bash
    mkdir processed_data
    python <mace_repo_dir>/mace/cli/preprocess_data.py \
        --train_file="/path/to/train_large.xyz" \
        --valid_fraction=0.05 \
        --test_file="/path/to/test_large.xyz" \
        --atomic_numbers="[1, 6, 7, 8, 9, 15, 16, 17, 35, 53]" \
        --r_max=4.5 \
        --h5_prefix="processed_data/" \
        --compute_statistics \
        --E0s="average" \
        --seed=123 \ preprocess_data.py --data_path path_to_your_data --out_path path_to_save_preprocessed_data

This will create a directory processed_data with the preprocessed data as h5 files. 
There will be one folder for training, one for validation and a separate one for each config_type in the test set.
To see all options and a little description of them run ``python ./mace/scripts/preprocess_data.py --help`` . 
The statistics of the dataset will be saved in a json file in the same directory.

The preprocessed data can be used for training the model using the on-line dataloader as shown in the example below.

.. code-block:: bash
    python <mace_repo_dir>/mace/cli/run_train.py \
    --name="MACE_on_big_data" \
    --num_workers=16 \
    --train_file="./processed_data/train.h5" \
    --valid_file="./processed_data/valid.h5" \
    --test_dir="./processed_data" \
    --statistics_file="./processed_data/statistics.json" \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=100 \
    --swa \
    --start_swa=60 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --error_table='PerAtomMAE' \
    --device=cuda \
    --seed=123 \