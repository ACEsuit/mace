"""Console entry points, and nothing else.

Each module here parses, builds the configuration and calls the stages. No
logic lives in this package: what a run does belongs to
:mod:`mace_torch.train`, and a command line that carried behaviour of its own
would be a second place for it that only the command line can reach.
"""
