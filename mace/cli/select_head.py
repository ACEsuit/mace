from argparse import ArgumentParser

import torch

from mace.tools.scripts_utils import remove_pt_head


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--head_name",
        "-n",
        help="name of the head to extract",
        default=None,
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="name for output model, defaults to model_file.target_device",
    )
    parser.add_argument("model_file", help="input model file path")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.model_file + "." + args.target_device

    model = torch.load(args.model_file)
    model_single = remove_pt_head(model, args.head_name)
    torch.save(model_single, args.output_file)


if __name__ == "__main__":
    main()
