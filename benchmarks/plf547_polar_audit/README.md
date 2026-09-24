# PLF547 POLAR correction audit

This isolated branch repurposes the existing dispatchable workflow path only on this branch. No default-branch CI configuration is changed. It targets the still-registered bluesky A100 runner, evaluates the two released models sequentially in float64, and uploads partial evidence on failure.

The runtime archive contains the exact installed Python sources used for the CPU audit (MACE 0.3.16, graph-longrange 0.4.0), with per-file SHA256 and original distribution metadata/licenses. It takes precedence over pip-installed MACE. Checkpoints are downloaded from the public release and checked against the CPU checkpoint hashes. Inputs and frozen public predictions are from ML-PEG PLF547. Original upstream input revision: 025fae1c73c43a0366035b472defa0103df44e1a. Public prediction snapshot: 2026-09-24, https://ml-peg.stfc.ac.uk/assets/supramolecular/PLF547/figure_plf547.json.

The source is not edited. A process-local hook bypasses only the reciprocal-grid allocation that is unused by isolated real-space calculations. It rejects PBC and forced periodic evaluation. Controls compare a bounded native GPU water evaluation with the bypass, three public nonchlorine predictions, and the saved corrected-Cl CPU component energies and interaction energy. Float64 GPU comparisons allow 1e-5 kcal/mol to accommodate backend reductions; no empirical energy adjustment is made.

Smoke runs evaluate 2OBF_01 in both interpretations. Full runs evaluate all 129 chlorine cases in both interpretations, and retain 418 unaffected public predictions. A full run is a controlled correction of this public snapshot, not a fresh full547 benchmark. Wrong-C predictions must reproduce each public prediction within 1e-5 kcal/mol; any failed control or comparison stops the job. The local script supports explicit --resume of a matching output, saves atomically after each case and refuses implicit overwrite.

GPU allocator is capped at 60% of VRAM. Native-grid validation is capped at two million candidate vectors. Per-model timeout is 45 minutes. One model is active at a time. No inference runs on the local laptop.
