.. _finetuning:

*******************
Analytical Hessians
*******************

The analytical hessian can be computed using ASE calculator, as in the following example:

.. code-block:: python

    from mace.calculators import mace_mp
    from ase import build

    atoms = build.molecule('H2O')
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda')
    atoms.calc = calc
    hessian=calc.get_hessian(atoms=atoms)