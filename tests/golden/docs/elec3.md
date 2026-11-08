# LC-MACE and LSC-MACE references

The explicit `elec3` target captures the first two families of ELEC-3. It is
excluded from `all`; ordinary tests read committed JSON and canonical
safetensors without importing the external packages. Four tiny anchors cover
the two families with monopoles alone and with monopoles plus dipoles.

The oracle runs unmodified, in a child process with three clean checkouts:

| Directory under `MACE_ELEC3_SOURCES` | Repository | Commit |
| --- | --- | --- |
| `elec3-reference` | [ACEsuit/mace-scf](https://github.com/ACEsuit/mace-scf) | `784a510339aacc4412b61b6926ef1a92bca387c0` |
| `elec3-legacy` | [ACEsuit/mace](https://github.com/ACEsuit/mace) | `9e223a1dd2e20eacd8e6bd551775adc7bd3a0682` (v0.3.14) |
| `elec3-longrange` | [WillBaldwin0/graph_electrostatics](https://github.com/WillBaldwin0/graph_electrostatics) | `e0eb8945c21e203656f4e0bbece5d1da488b6753` |

The pins live in `targets/elec3.py`. The external version assertion remains
intact, and the repository's frozen `mace/` is untouched. The test-only bridge
projects backbone weights into the agreed reduced basis and copies the source
readouts. LSC omits the unused monopole readout and tensor-product paths that
cannot reach the scalar transfer; these contribute nothing in the oracle.
This bridge is not a converter for public checkpoints.

Each family reference contains twelve cases: orders zero/one crossed with an
isolated molecule, its rotated copy, a triclinic crystal, a slab, a mixed batch
using `partial`, and a mixed batch using `per_structure`. Fields are nonzero.
Every case pins energy, short-range site energies, forces, stress, virials,
dipole and density coefficients. The capture compares V1 to the oracle before
writing. CPU tests use the existing `fp64_cpu_reference` and `fp32` rows;
GPU tests use `fp64_accelerated_backend` and `fp32`. No bound is widened.

Three existing contracts matter when interpreting these references:

- Aperiodic stress is masked to zero in V1. Its physical virial is compared
  without masking. The reference stress is adapted only for those structures.
- Upstream `auto` selects a regime for an entire batch. V1 `per_structure`
  preserves independent physical systems, so its reference is assembled from
  separate oracle evaluations with identical weights.
- The finite-difference dipole solver displaces charges along laboratory axes.
  Its molecular energy has a small orientation error, already part of the
  solver descriptor contract. The rotated fixture pins that same function;
  the learned charges and dipoles have separate equivariance tests.

Float32 means float32 model parameters with V1's existing float64 accumulation
floor and the corresponding electrostatics boundary. It does not promise a
wholly float32 solve. An exploratory all-float32 run failed the existing virial
bound for LC dipoles; it is not included in the validated precision claim.
The two pre-existing Polar float32 gaps remain unchanged.

```bash
# Only when initial capture or recapture is explicitly authorized:
MACE_ELEC3_SOURCES=/path/to/checkouts python tests/golden/regenerate.py \
  --target elec3 --i-know-what-i-am-doing

python -m pytest tests/parity/test_elec3_goldens.py -m 'not gpu'
MACE_REQUIRE_CAPS=gpu,cueq python -m pytest \
  tests/parity/test_elec3_goldens.py -m gpu
```

The default source directory is the ignored repository `scripts/` directory.
QEq, FixedPointSCF and EnergyFunctionalSCF have no goldens from this target yet.
The [implementation record](../../../docs/reforge/gate3/2026-11-08/elec3-local-sources.md)
lists supported options and the remaining ELEC-3 scope.
