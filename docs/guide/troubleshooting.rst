.. _troubleshooting:

=============================================================
MACE: Troubleshooting Guide
=============================================================

When getting started with MACE, users often encounter similar challenges. This guide addresses the most common issues to help you successfully fit your first MACE model.

Data Loading Issues
-------------------

**Q:** Why doesn't MACE recognize my energy and forces data?

**A:** MACE looks for specific keys in your XYZ files. By default, it searches for ``REF_energy`` and ``REF_forces``. Check your command-line arguments:

.. code-block:: bash

    --energy_key="REF_energy" --forces_key="REF_forces"

If you see a warning like *"standard deviation is zero,"* your data might not be loading correctly. Double-check your keys in both your data files and MACE arguments.

**Q:** How can I verify my data is loaded correctly?

**A:** Look at the initial training output, which displays the number of configurations, energies, and forces. For example:

.. code-block:: text

    Training set [100 configs, 100 energy, 30000 forces] loaded from 'training.xyz'

If any numbers are unexpectedly zero, your data isn't being read properly. 
If your model is not training properly, check the data loading step first. 
A hint that the data is not loaded correctly is unusually high initial loss values and a rapid plateau in training loss.

Energy Offset (E0s) Problems
----------------------------

**Q:** What values should I use for atomic reference energies (E0s)?

**A:** You have two main options:

- **DFT reference energies:** Use isolated atom energies calculated with exactly the same DFT settings as your dataset.
- **Average E0s:** Use ``--E0s=average`` to let MACE determine these values from your dataset.

Using mismatched E0s is a common source of large errors. For systems with highly different electronic states (e.g., mixed oxidation states), carefully computed reference E0s are recommended.

**Q:** How do I know if my E0s are causing problems?

**A:** Check the initial validation loss. For energy, expected values are:

- **Good E0s:** Initial RMSE of 0.1–4 eV/atom
- **Problematic E0s:** Initial RMSE >5 eV

Initial Metrics Interpretation
------------------------------

**Q:** What initial metrics should I expect for a healthy training run?

**A:** For initial metrics if using DFT reference energies:

- **Energy RMSE:** 0.1–4 eV/atom
- **Forces RMSE:** 0.5–4 eV/Å

If your initial energy error is in the 10 of eV/atom, there's likely a problem with your data loading, your E0s or your data.

Dataset Quality Issues
----------------------

It is recommended to not use configurations of too high energies or forces in the training set, as they can lead to bad training dynamics.
A sensible cutoff on the forces is about 200 eV/Å.
If data is of high energy or forces are required, it recommended to use the a Huber loss function by specifying the `--loss="Universal"` argument.

Multi-head Finetuning Issues
----------------------------

**Q:** What should I check when using multi-head finetuning?

**A:** The initial loss for finetuning should be relatively small:

- **Energy:** 40–300 meV/atom
- **Forces:** 100–600 meV/Å

If you see significantly larger values, check:

- Data parsing (keys match both pretraining and finetuning datasets)
- E0s (should be recomputed with your finetuning DFT settings)
- Spin polarization (use spin-polarized calculations if your system requires it)

Remember to use the ``--foundation_model`` flag to specify your base model.

Cutoff Radius Selection
-----------------------

**Q:** What cutoff radius should I use?

**A:** The optimal cutoff is system-dependent, but:

- **Recommended range:** 4–7 Å
- **Standard starting value:** 6 Å
- **Minimum recommended:** 4 Å (smaller values significantly reduce accuracy)

After having done an initial fit with the recommended values, reduce the cutoff if you have memory or speed constraints.

Memory Issues
-------------

**Q:** I'm getting *"CUDA out of memory"* errors. How can I fit my model?

**A:** Try these solutions in order:

1. Enable CUEQ acceleration with ``--enable_cueq=True``
2. Reduce batch size with ``--batch_size=4``
3. Reduce the cutoff radius (e.g., from 6Å to 5Å)
4. Decrease model size using ``--num_channels=64`` (default is 128) or ``--max_L=0`` (default is 1)
5. Try training on CPU first to verify your setup works

Remember that any reduction in model size or cutoff may affect accuracy. Reducing the number of channels is usually the least impactful change.

Getting More Help
-----------------

**Q:** Where can I find more comprehensive guidance on using MACE?

**A:** Check out the tutorials in the MACE documentation:

- `Tutorial 1: Introduction to MACE training and evaluation <https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX>`_
- `Tutorial 2: MACE active learning and fine-tuning <https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb>`_
- `Tutorial 3: MACE theory and code (advanced) <https://colab.research.google.com/drive/1AlfjQETV_jZ0JQnV5M3FGwAM2SGCl2aU>`_

The GitHub repository also has example scripts for common use cases and detailed explanations in the documentation.

If you still encounter issues, check the GitHub Issues and GitHub Discussions sections for similar problems or open a new issue/discussions with details of your specific case.

Issues are meant for actual bugs reports or feature requests, while Discussions are for general questions or more specific help requests.
