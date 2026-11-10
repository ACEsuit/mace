// Kokkos owns atom and neighbour buffers; the native graph remains on CUDA.
#pragma once
#include <cstdint>
namespace LAMMPS_NS {
class LAMMPS;
class Pair;
class NeighList;
class NeighRequest;
} // namespace LAMMPS_NS
namespace mace_native {
void configure_kokkos(LAMMPS_NS::LAMMPS *, LAMMPS_NS::Pair &);
void request_kokkos_neighbors(LAMMPS_NS::NeighRequest &);
void pack_kokkos_graph(LAMMPS_NS::LAMMPS *, LAMMPS_NS::NeighList *,
                       const int64_t *types, int capacity, double cutoff,
                       int64_t padding_shift, double *positions,
                       int64_t *elements, int64_t *index, double *shifts);
void add_kokkos_forces(LAMMPS_NS::LAMMPS *, const double *forces);
} // namespace mace_native
