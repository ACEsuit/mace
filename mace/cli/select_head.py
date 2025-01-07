from argparse import ArgumentParser

import torch

from mace.tools.scripts_utils import remove_pt_head


def main():
    parser = ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--head_name",
        "-n",
        help="name of the head to extract",
        default=None,
    )
    grp.add_argument(
        "--list_heads",
        "-l",
        action="store_true",
        help="list names of the heads",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="name for output model, defaults to model_file.target_device",
    )
    parser.add_argument("model_file", help="input model file path")
    args = parser.parse_args()

    model = torch.load(args.model_file)
    if args.list_heads:
        print("Available heads:")
        print("\n".join(["  " + h for h in model.heads]))
    else:
        if args.output_file is None:
            args.output_file = args.model_file + "." + args.head_name + "." + str(next(model.parameters()).device)
        model_single = remove_pt_head(model, args.head_name)
        torch.save(model_single, args.output_file)


if __name__ == "__main__":
    main()
