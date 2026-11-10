"""The differentiable physics glue, characterized against finite differences.

**These docstrings are normative.** CORE-3 copies the three sign conventions
below verbatim into the rewrite's units module, and every one of them is
asserted by a test in this file rather than assumed:

* **forces** = -dE/d(positions), in eV/Ang.
* **stress** = (1/V) dE/d(strain), in eV/Ang^3, with V = |det(cell)|.
* **virials** = -stress * V = -dE/d(strain), in eV.

All three come out of one function, `compute_forces_virials`
(mace/modules/utils.py:52-80): it takes the raw autograd gradients of the
energy with respect to the positions and to the injected strain, divides the
strain gradient by |det(cell)| to form the stress, and negates *both* raw
gradients on the way out. So the returned stress is the only one of the three
that is **not** negated, and the virial is the negative of the same quantity
the stress is built from. That asymmetry is the single most likely thing for
a port to normalise away, which is why the relation is pinned twice here --
once against a finite-difference derivative and once against a synthetic
energy whose gradient is known exactly and by hand.

Everything runs on the committed fp64 anchors on CPU, with no capability
marker: this is the specification the new derivative code is written against,
so it may not be a test that can skip.

Tolerances come from the one table in `tests/golden/harness.py`. The
finite-difference comparisons use the fp64 CPU row (1e-6 absolute), and they
clear it by four to five orders of magnitude -- measured on this tree, the
worst deviation is 2.2e-11 eV/Ang for the forces and 2.5e-13 eV/Ang^3 for the
stress. The budget at h = 1e-4 Ang is dominated by cancellation, not by the
5-point stencil's O(h^4) truncation: roughly eps * |E| / h ~ 1e-16 * 40 / 1e-4
= 4e-11, which is what is actually observed.
"""
