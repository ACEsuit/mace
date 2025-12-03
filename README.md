# MACE-Field: Field-Aware MACE Models

## Overview

**MACE-Field** is a field-aware extension of the **MACE** (Message Passing Atomic Cluster Expansion) architecture that enables learning **electric-field–dependent energy functionals** for molecules and periodic materials. From a *single scalar electric enthalpy*, MACE-Field exposes physically consistent dielectric response properties via **automatic differentiation**:

- **Polarization**  
  $\mathbf{P} = -\frac{1}{\Omega}\,\frac{\partial E}{\partial \mathbf{E}}$

- **Born effective charges (BECs)**  
  $Z^\ast_{\kappa,\alpha\beta} = \frac{\partial P_\alpha}{\partial R_{\kappa,\beta}}$

- **Polarisability/susceptibility**  
  $\chi_{\alpha\beta} = \frac{\partial P_\alpha}{\partial E_\beta}$

All quantities are **derivative-consistent** (Maxwell relations, acoustic sum rule) by construction.

MACE-Field can be:
- trained **from scratch**, or
- used to **fine-tune existing MACE foundation models** to become *field-aware*.

---

## Installation

```bash
git clone https://github.com/mdi-group/mace-field.git
pip install ./mace
```

> **Note**  
> MACE-Field is in the process of being upstreamed into the main MACE repository  
> https://github.com/ACEsuit/mace/pull/1177

---

## Architecture summary

![MACE-Field architecture](macefield_architecture.png)

MACE-Field preserves the standard MACE backbone and readout, but **injects a uniform electric field** (an O(3) irrep `1o`) into the *latent equivariant features* at each interaction layer.

Conceptually:
1. Atomic neighbourhoods are expanded into equivariant “multipole-like” features (as in MACE / ACE).
2. A **global electric field** couples to these features through symmetry-allowed Clebsch–Gordan tensor products.
3. The final readout remains a scalar energy (electric enthalpy).
4. Dielectric observables are obtained by *exact differentiation* of this scalar.

At zero field, MACE-Field **reduces exactly to standard MACE**, enabling foundation-weight reuse.

*(See Fig. 1 in Martin et al., 2025 for an architectural schematic.)*

---

## Data format

MACE-Field uses ASE-readable datasets (typically extended XYZ).

### Required configuration-level fields (`atoms.info`)

| Key | Shape | Units |
|---|---|---|
| `REF_energy` | scalar | eV |
| `REF_stress` or `REF_virials` | (6,) or (3,3) | eV/Å³ |
| `REF_electric_field` | (3,) | V/Å |
| `REF_polarization` | (3,) | e/Å² |
| `REF_polarizability` | (3,3) or (9,) | e/(V·Å) |

### Required per-atom arrays (`atoms.arrays`)

| Key | Shape | Units |
|---|---|---|
| `REF_forces` | (N,3) | eV/Å |
| `REF_becs` | (N,3,3) | e |

Key names can be overridden via CLI flags.

---

## Training a MACE-Field model

Training uses the standard MACE CLI with:
- `--model MACEField`
- `--loss universal_field`

```bash
python -m mace.scripts.run_train \
  --model MACEField \
  --name MACEField_model \
  --train_file data/field_train.xyz \
  --valid_fraction 0.2 \
  --device cuda \
  --r_max 5.0 \
  --num_interactions 2 \
  --num_channels 128 \
  --max_L 1 \
  --loss universal_field \
  --energy_weight 1.0 \
  --forces_weight 100.0 \
  --polarization_weight 1.0 \
  --becs_weight 100.0 \
  --polarizability_weight 100.0
```

### Polarization branch folding

Polarization is multi-valued in periodic systems. During training, MACE-Field compares **folded polarization differences**, ensuring a branch-invariant loss:
- avoids discontinuities,
- supports ferroelectric distortion paths,
- preserves a conservative derivative definition.

---

## Fine-tuning a MACE foundation model (field-aware)

One of MACE-Field’s key strengths is **foundation-model inheritance**.

You can fine-tune a pretrained MACE model (e.g. `mace-mp-0b3`) to add polarization, BEC, and polarisability heads.

### Multi-head fine-tuning example

```yaml
name: mace-field-mp-0b3-mh
foundation_model: mace-mp-0b3-medium.model
model: MACEField
loss: universal_field
multiheads_finetuning: true

heads:
  mp-dielectric:
    train_file: becs-polarizabilities-train.xyz
    valid_file: becs-polarizabilities-valid.xyz
  mp-polarization:
    train_file: polarizations-train.xyz
    valid_file: polarizations-valid.xyz

pt_train_file: mp_replay.xyz
compute_forces: true
compute_polarization: true
compute_becs: true
compute_polarizability: true
```

Run:

```bash
torchrun --standalone --nproc_per_node=gpu \
  python -m mace.scripts.run_train --config config.yaml
```

This adds dielectric response **without degrading** the original energy/force accuracy.

---

## Inference

### ASE calculator

```python
from mace.calculators import MACECalculator

calc = MACECalculator(
    model_path="MACEField.model",
    model_type="MACEField",
    electric_field=[0.0, 0.0, 0.02],  # overrides per-structure field
)

atoms.calc = calc
E = atoms.get_potential_energy()
P = atoms.calc.results["polarization"]
Z = atoms.calc.results["becs"]
alpha = atoms.calc.results["polarizability"]
```

### Batch inference (XYZ)

```bash
mace_eval_configs \
  --configs input.xyz \
  --model MACEField.model \
  --output output.xyz \
  --compute_polarization \
  --compute_becs \
  --compute_polarizability
```

---

## Finite-field molecular dynamics

MACE-Field integrates seamlessly with ASE MD drivers.

```python
atoms.info["REF_electric_field"] = [0.0, 0.0, 0.1]

# Update dynamically if required:
calc.electric_field = [0.0, 0.0, Ez_t]
```

This enables:
- ferroelectric hysteresis loops,
- IR / Raman spectra,
- finite-field response at finite temperature.

---

## Units and conventions

- Energy: eV
- Force: eV/Å
- Electric field: V/Å
- Polarization: e/Å²
- BECs: |e|
- Polarisability: e/(V·Å) (often reported in units of ε₀)

---

## References

If you use this code, please cite:

```bibtex
@misc{martin2025generallearningelectricresponse,
  title={General Learning of the Electric Response of Inorganic Materials},
  author={Martin, Bradley A. A. and Ganose, Alex M. and Kapil, Venkat and Li, Tingwei and Butler, Keith T.},
  year={2025},
  eprint={2508.17870},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci}
}
```

and the main MACE papers:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kovacs, David Peter and Simm, Gregor N. C. and Ortner, Christoph and Csanyi, Gabor},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{'a}cs, D{'a}vid P{'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{'a}nyi, G{'a}bor},
  year = {2022},
  eprint = {2205.06643},
  archiveprefix = {arXiv}
}
```

---

## Contact

- **MACE-Field**: bradley.martin@ucl.ac.uk  
- **MACE core**: ilyes.batatia@ens-paris-saclay.fr  
- Issues & feature requests: <https://github.com/mdi-group/mace-field/issues>

---

## License

MACE-Field is released under the **MIT License**.  
(Some upstream MACE models may use different licenses.)
