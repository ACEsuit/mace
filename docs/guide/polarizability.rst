.. _polarizability:

=====================================================================
Polarizability with MACE: How to Train an AtomicDielectric MACE Model
=====================================================================

Training Example
================

A typical training command for an AtomicDielectric MACE model looks like:

.. code-block:: bash

  python /../mace/mace/cli/run_train.py \
       --name="mace_mu_alpha" \
       --train_file="train.xyz" \
       --valid_file="val.xyz" \
       --test_dir="test.xyz" \
       --model="AtomicDielectricMACE" \
       --E0s="average" \
       --num_interactions=2 \
       --num_channels=128 \
       --max_L=2 \
       --correlation=3 \
       --MLP_irreps="16x0e+16x1o+16x2e" \
       --dipole_key="REF_dipole" \
       --polarizability_key="REF_polarizability" \
       --loss="dipole_polar" \
       --weight_decay=5e-10 \
       --polarizability_weight=2000 \
       --dipole_weight=1000 \
       --clip_grad=1.0 \
       --batch_size=128 \
       --valid_batch_size=128 \
       --max_num_epochs=40 \
       --scheduler_patience=15 \
       --patience=15 \
       --eval_interval=1 \
       --ema \
       --error_table="DipolePolarRMSE" \
       --default_dtype="float64" \
       --device=cuda \
       --seed=123 \
       --restart_latest \
       --save_cpu

Setting --MLP_irreps="16x0e+16x1o+16x2e" and --max_L=2 are crutial for predicting polarizability correctly.
Compared to a MACE - MLIP these models usually need less epochs to converge.

Extracting Polarizability Using ASE
===================================

Once training is complete and you have your model file (e.g., `mace_mu_alpha.model`), you can extract polarizability tensors from a trajectory.

Example extraction script:

.. code-block:: python
    
   import numpy as np
   from ase.io import Trajectory
   from mace.calculators.mace import MACECalculator

   # Setup calculator
   polar_calc = MACECalculator(
       model_paths="mace_mu_alpha.model",
       model_type="DipolePolarizabilityMACE",
       device="cuda",
       default_dtype="float64"
   )

   traj = Trajectory("test.traj", "r")
   n_frames=len(traj)
   alpha = np.empty((n_frames, 3, 3), dtype=float)

   for i, atoms in enumerate(traj):
       atoms.calc = polar_calc
       alpha[i] = np.asarray(atoms.calc.get_property("polarizability", atoms)).reshape(3, 3)
       print(f"Frame {i} Polarizability: ", alpha[i])
   traj.close()


**Tip:**  
To get the **spherical polarizability** (e.g. for Raman spectra), use the property `"polarizability_sh"` with `get_property`:

.. code-block:: python

   spherical_alpha = atoms.calc.get_property("polarizability_sh", atoms)