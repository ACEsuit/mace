.. _descriptors:

================
MACE descriptors
================

MACE descriptors are atomic features obtained after each message passing block of a MACE model.
To extract these descriptors directly from an ase.Atoms object, you can use the convenient function provided in the MACECalculator class.

Here's an example in Python:

.. code-block:: python

    from ase.io import read
    import numpy as np

    from mace.calculators import MACECalculator

    calculator = MACECalculator(model_path='/content/checkpoints/MACE_model_run-123.model', device='cuda')
    init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')

    descriptors = calculator.get_invariant_descriptors(init_conf)

Please note that only the invariant part of the node features is extracted, and currently, only the base `MACE` and `ScaleShiftMACE` models are supported.

Alternatively, if you're using a `MACE` model directly, you can utilize the `get_node_invariant_descriptors` method.
This allows you to track the gradients of the descriptors with respect to the input atomic positions.

.. code-block:: python

    from ase.io import read
    import torch
    from mace.tools.torch_geometric.data_loader import DataLoader
    from mace import data
    from mace.data.utils import config_from_atoms

    model = torch.load(model_path)
    model.eval()
    atoms = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
    config = data.config_from_atoms(atoms)
    data_loader = DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    descriptors = model.get_node_invariant_descriptors(batch, track_gradient_on_positions=True)

You can now use the ``descriptors`` in a downstream `nn.Module` for further processing.