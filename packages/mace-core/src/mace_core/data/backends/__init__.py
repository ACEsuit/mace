"""Dataset backends shipped with mace-core.

A format is registered through the ``mace.data_backends`` entry point group and
resolved by :func:`mace_core.data.registry.open_dataset`; nothing imports a
format module by name.
"""
