.. _multihead:

****************************
Multihead Training for MACE
****************************

MACE supports training on multiple level of theories simultaneously. This is called multihead training.
In order to do that, MACE constructs as many readout functions as the number of level of theories.
The rest of the weights are all shared. 
This enables the model to learn the different level of theories simultaneously, that are potentially inconsistent with each other.

To train a multihead model, you can use the mace_run_train script with the following yaml file in the --config argument:

.. codeblock:: yaml

    heads: 
        3bpa_wb97x_d3bj:
            train_file: train_3bpa_wb97x_d3bj.xyz
            valid_file: val_3bpa_wb97x_d3bj.xyz
            E0s: e0s_wb97x_d3bj.json
            energy_key: dft_energy
            forces_key: dft_forces

        3bpa_ccdt:
            train_file: train_3bpa_ccdt.xyz
            valid_file: val_3bpa_ccdt.xyz
            E0s: e0s_ccdt.json
            energy_key: ccdt_energy
            forces_key: ccdt_forces