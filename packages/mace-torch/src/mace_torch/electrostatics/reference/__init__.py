"""The reference long-range solver: Gaussian multipoles, summed in k-space.

These modules are ``graph_longrange`` by Will Baldwin, from
https://github.com/WillBaldwin0/graph_electrostatics at commit
``6a86de5e3ed35fd86a55fc046aa085fe48a72764`` (version 0.4.4), the commit the
``develop`` branch pins for its polar models. They are distributed under the
MIT License, whose text is in ``LICENSE.md`` beside them, and they keep their
authorship.

Only the part the long-range energy and the electrostatic features need is
here: the energy, the features, the Gaussian basis, the k-vectors, the
real-space sum, the slab and molecule corrections, and the helpers. What was
changed on the way in, and nothing else:

* the spherical harmonics come from this package's reference backend instead
  of ``e3nn``, divided by the square root of the sphere's area to give the
  integral normalisation ``e3nn`` was asked for;
* feature dimensions are counted directly instead of through ``e3nn`` irreps;
* the scatter helpers are the ones in ``utils`` instead of the frozen tree's,
  which were the same code;
* the two TorchScript decorators are gone, since nothing here is scripted;
* the k-space search bounds are read as integers, the only thing
  ``torch.arange`` does with them;
* the style: formatting, unused locals, names and annotations.

Against the original, with ``e3nn``, energies agree bit for bit in the
periodic, slab and open cases and to 1.8e-15 in a batch that mixes them, and
their derivatives against positions, multipoles and the cell to 3.6e-15.
"""
