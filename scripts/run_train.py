###########################################################################################
# Training script for MACE
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from mace import tools
from mace.model_training import train_mace_model


def main() -> None:
    args = tools.build_default_arg_parser().parse_args()
    train_mace_model(args)


if __name__ == "__main__":
    main()
