.. _descriptors:

================
MACE descriptors
================

MACE descriptors are atomic features obtained after each message passing block of a MACE model.
As MACE is a very expressive model, these descriptors are rich and can be used for various tasks, such as classification or analysis.
To extract these descriptors directly from an ase.Atoms object, you can use the convenient function provided in the MACECalculator class.

Here is a simple example:

.. code-block:: python

    from ase.io import read
    import numpy as np
    from mace.calculators import MACECalculator
    calculator = MACECalculator(model_path='/content/checkpoints/MACE_model_run-123.model', device='cuda')
    init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
    descriptors = calculator.get_descriptors(init_conf)

Currently only the base `MACE` and `ScaleShiftMACE` models are supported.
By default, only the invariant part of the descriptors is returned.
To get the full descriptors, you can set the `invariants_only` argument to `False`.

Also, by default, the descriptors for each layer are returned.
To get the descriptors of the first `n` layers, you can set the `num_layers` argument to `n`.

.. note::

   The descriptors are returned in the form of numpy arrays, structured as follows:

   - **First Dimension**: Corresponds to the number of atoms in the system.
   - **Second Dimension**: Relates to the number of descriptors, dependent on the model used.

   Depending on the value of the `invariants_only` argument, the number of descriptors, :math:`N_{\text{descriptors}}`, is calculated as follows:

   - If `invariants_only=True`: :math:`N_{\text{descriptors}} = \text{nchannels} \times \text{nlayers}`.
   - If `invariants_only=False`: :math:`N_{\text{descriptors}} = \text{nchannels} \times (\text{nlayers} - 1) \times (\text{L_{max}} + 1)^{2} + \text{nchannels}`.
