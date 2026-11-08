# Local-source extension references

`elec3-local-extensions` captures the initial reference family for the optional
LSC oxidation mixers, oxidation-dependent bond transfer and direct polarizability.
The [base family](elec3.md) and its pinned external revisions stay unchanged.

Seven configurations cross monopole/dipole densities and the six existing
geometry/batch profiles. The combined per-atom case uses fractional formal
charges and a non-default oxidation range. Every case records energies, forces,
stress, virials, dipole, density and a canonical node-embedding gradient of the
force/stress loss, including polarizability loss where the readout is enabled.
The model has two channels and two interactions, with upstream seed 123.

The direct polarizability is a separate learned observable. It does not modify
the electrostatic energy or represent a derivative of the model's dipole with
respect to its external field. Rotations act on it as a Cartesian rank-two
tensor. Formal-charge sources and oxidation polynomials retain the existing
declared-input and canonical-checkpoint contracts.

```bash
# Initial capture; recapture requires separate explicit authorization.
python tests/golden/regenerate.py --target elec3-local-extensions --i-know-what-i-am-doing
python -m pytest tests/parity/test_elec3_local_extensions_goldens.py -m 'not gpu'
MACE_REQUIRE_CAPS=gpu,cueq python -m pytest tests/parity/test_elec3_local_extensions_goldens.py -m gpu
```
