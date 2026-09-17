"""The reduced symmetric tensor-product basis, and the coefficients under it.

Imports neither ``e3nn`` nor ``cuequivariance``. The basis is model state, one
value for every device and every backend: nothing here branches on what happens
to be installed. That is the defect this package exists to remove, since on the
frozen tree a host without ``cuequivariance`` silently trains a different, more
heavily parametrized network for the same hyperparameters.
"""

from mace_core.clebsch_gordan.coefficients import clebsch_gordan, wigner_3j_complex

__all__ = ["clebsch_gordan", "wigner_3j_complex"]
