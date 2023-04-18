.. _ase:

================
ASE calculator
================

MACE models can run molecular dynamics or geometry optimisation through the ASE calculator.
The ASE calculator is a Python module that can be used to run MD simulations or geometry optimisations.

The native ASE neighbour list being slow, it is recommended to use the matscipy `branch <https://github.com/ACEsuit/mace/tree/52-matscipy-neighbour-list-as-default>`_.
**Note** that the matscipy branch is not compatible with non-periodic systems.

Running MD simulations
----------------------

The ASE calculator can be used to run MD simulations with MACE. 
The following example shows how to run a MD simulation of a 3BPA molecule in vacuum.

.. code-block:: python

    from ase import units
    from ase.md.langevin import Langevin
    from ase.io import read, write
    import numpy as np
    import time

    from mace.calculators import MACECalculator

    calculator = MACECalculator(model_path='/content/checkpoints/MACE_model_run-123.model', device='cuda')
    init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
    init_conf.set_calculator(calculator)

    dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
    def write_frame():
            dyn.atoms.write('md_3bpa.xyz', append=True)
    dyn.attach(write_frame, interval=50)
    dyn.run(100)
    print("MD finished!")

To get the model and the data associated with this example, please run the colab tutorial `here <https://colab.research.google.com/drive/1D6EtMUjQPey_GkuxUAbPgld6_9ibIa-V?authuser=1#scrollTo=wfCwdnaWv9rd>`_.