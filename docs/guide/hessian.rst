.. _hessian:

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
    hessian = calc.get_hessian(atoms=atoms)

**Note:**  
The implementation of analytical Hessians in MACE is based on the methodology described in [1]_.  
Users are encouraged to cite this paper when using Hessians in their work.

.. [1] Nils Gönnheimer, Karsten Reuter, and Johannes T. Margraf,  
   *Journal of Chemical Theory and Computation* **2025**, 21 (9), 4742–4752.  
   `DOI:10.1021/acs.jctc.4c01790 <https://doi.org/10.1021/acs.jctc.4c01790>`_