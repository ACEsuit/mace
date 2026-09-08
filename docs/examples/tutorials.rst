.. _tutorials:

============================================
Tutorials on MACE training and architecture
============================================

We have built a series of tutorials to help you get started with MACE. 
These tutorials are designed to help you understand the basics of MACE, how to train a model, and how to use it in your own projects.
Tutorials 1–3 were made by Ioan Magdău, Ilyes Batatia and Will Baldwin.
Tutorial 4 was made by Eszter Varga-Umbrich and Tamás Lajos Tompa.
All tutorials are available on Google Colab, so you can run them in your browser without any setup.
If you want to run the tutorials locally, you can download the notebooks from Google Colab.


########################################################
Tutorial 1: Introduction to MACE training and evaluation
########################################################

In this tutorial, we will introduce you to the basics of MACE training and evaluation.
We cover the construction of a dataset, the basic hyperparameters of a MACE model, and how to train and evaluate a model.

Link to tutorial in colab: https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX

################################################
Tutorial 2: MACE active learning and fine-tuning
################################################

In this tutorial, we will show you how to use MACE for active learning and fine-tuning.

Link to tutorial in colab: https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb

###########################################
Tutorial 3: MACE theory and code (advanced)
###########################################

In this tutorial, we will dive into the theory and code of MACE.
Each section of the code is explained in detail, and we reference the corresponding equations in the manuscript.

Link to tutorial in colab: https://colab.research.google.com/drive/1AlfjQETV_jZ0JQnV5M3FGwAM2SGCl2aU

##########################################################
Tutorial 4: Reliable fine-tuning of MACE foundation models
##########################################################

This tutorial explains how atomic reference energies, method-specific
optimiser settings and deployment-aware validation affect fine-tuned MACE
models. It provides commands for naive fine-tuning, layer freezing, LoRA and
pseudolabel replay, and uses released results from `Fine-tuning MLIP foundation
models: strategies for accuracy and transferability
<https://arxiv.org/abs/2606.12704>`_.

The default CPU path downloads less than 10 MB and reproduces the diagnostic
plots without training. GPU training is an explicit opt-in.
For a concise decision guide, see :ref:`finetuning_guidance`.

`Open the tutorial in Google Colab
<https://colab.research.google.com/github/ACEsuit/mace/blob/docs/docs/examples/reliable_finetuning_of_mace_foundation_models.ipynb>`_
