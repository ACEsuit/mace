## Wrapper for mace.cli.run_train.main ##

import yaml
import os
from mace.cli.run_train import run
from mace import tools

import warnings

warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system doesn't support instance-level annotations",
)

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True

train_xyz = "/home/king1305/Apps/les_fit/data-benchmark/train-H2O_RPBE-D3.xyz"
test_xyz = "/home/king1305/Apps/les_fit/data-benchmark/test-H2O_RPBE-D3.xyz"
if on_cluster:
    train_xyz = "/global/scratch/users/king1305/data/train-H2O_RPBE-D3.xyz"
    test_xyz = "/global/scratch/users/king1305/data/test-H2O_RPBE-D3.xyz"

les_arguments = {
        "use_dipole": True,
        "use_quads": False,
        "use_induced_dipole": True,
        "use_anisotropic_polarizability": True,
        "make_alpha_positive":True,
}

if __name__ == "__main__":
    with open("les.yaml", "w") as f:
        yaml.dump(les_arguments, f)

    args = tools.build_default_arg_parser().parse_args([
        "--name=H2O",
        f"--train_file={train_xyz}",
        "--valid_fraction=0.05",
        f"--test_file={test_xyz}",
        "--energy_key=energy",
        "--forces_key=forces",
        "--E0s=average",
        "--model=MACELES",
        "--les_arguments=les.yaml",
        # "--hidden_irreps=128x0e + 128x1o + 128x2e",
        "--hidden_irreps=128x0e + 128x1o",
        "--r_max=4.5",
        "--num_interactions=1",
        "--batch_size=4",
        "--max_num_epochs=1000",
        "--ema",
        "--ema_decay=0.99",
        "--amsgrad",
        "--restart_latest",
        "--device=cuda",
        "--default_dtype=float32",
        "--save_cpu",
    ])

    run(args)
