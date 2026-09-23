"""Model weights in a form no framework owns.

:mod:`mace_core.weights.neutral_format` is the interchange format between a
trained model and whatever reads it next: the v1 torch stack, the JAX one, or a
converter carrying a legacy checkpoint across.
"""
