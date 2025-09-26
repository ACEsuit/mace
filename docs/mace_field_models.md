# Basic structure

The archtiecture is still adapting, but a rough overview of the current electrostatic OMOL models is:

![fukui mace models](./field_mace_fukui_steps.png)

A few points about the model:
- This model is not 'self-consistent', since there is no conevrgence criterion. The number of SC-like steps ($N$) is normally just set to 3.
- One can apply external fields, since the electric potential features $v_{nlm}^i$ are proejctions of the actual electric potential, from both the model's charges and from aplied fields. 

# Requirements

- graph_longrange. Contact `WillBaldwin0` and use the `develop` branch. 

# Models
The available models can be found in https://github.com/ilyes319/mace-field/releases/tag/fukui-1.
There are three models with 1, 2, and 3 mace layers, all trained on 100 millions OMOL.


# How to use the model

set up and use a calculator via

```python
from mace.calculators import MACECalculator

model_path = "mace-field-fukui-spin-2L.model""
calc = MACECalculator(
    model_path,
    device='cuda',
)

atoms.calc = calc

# with specified total charge, spin, field:
atoms.info['charge'] = 1.0              # (e)
atoms.info['spin'] = 1                  # multiplicity, default=1 (not total spin but (number of unpaired electrons + 1))
atoms.info['external_field'] = 0.01     # V/\AA. note that sign is REVERSED. fields point uphill, consistent with fhi-aims, vasp, ...
atoms.get_forces()
```

To change the keys for charge, spin, external fields, use:
```
calc = MACECalculator(
    model_path,
    device='cuda',
    info_keys={
        'total_charge': 'custom_charge_key',
        'external_field': 'other_field_key',
        'total_spin': 'my_spin_key',
    }
)
```
In this case, any keys which are not specified will take the default values of `charge`, `spin`, `external_field`. 


# What does the model output

- Energy/forces (as usual)
- Total dipole `calc.results['dipole']`. this is only a well defined quantity for non-periodic systems, and is a meaningless value for periodic systems.
- Partial charges and partial dipoles. These are the final values of $q_{lm}^i$ in the figure above. We store atomic multipoles as spherical (not cartesian) multipoles, and use the Condon-Shortley Phase convention. This can be accessed via `calc.results['density_coefficients']` which is `(n_atoms, 4)`. One can extract partial charges and (cartesian) partial dipoles via 
    - `atomic_charges = calc.results['density_coefficients'][:,0]`
    - `atomic_dipoles = calc.results['density_coefficients'][:,[3,1,2]]`  
- Partial charges and dipoles are **not well defined**. Futhermore, sums of these quantities onto clusters or molecules are also not defined, except one sums the totals on **isolated** fragments, where isolated means that the fragment to be summed doesnt come within ~6 Angstrom of any other atoms.
- Partial spins (some models). We also store a spin density array `calc.results['spin_charge_density']` which is `(n_atoms, 2, 4)`. This also contains atomic multipoles, resolved to two different spin channels. The sum across the second index is the atomic multipoes above.

