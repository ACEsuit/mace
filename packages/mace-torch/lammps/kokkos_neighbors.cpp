#include "kokkos_neighbors.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "pair.h"
#include <Kokkos_UnorderedMap.hpp>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace mace_native {
using namespace LAMMPS_NS;
namespace {
void sync_torch() {
  const auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace
void configure_kokkos(LAMMPS *lmp, Pair &pair) {
  if (!lmp->kokkos)
    throw std::runtime_error("mace/native/kk requires LAMMPS -k on g 1");
#ifndef KOKKOS_ENABLE_CUDA
#error "The native Kokkos adapter requires a CUDA-enabled Kokkos build"
#endif
  pair.kokkosable = 1;
  pair.execution_space = Device;
  pair.datamask_read = X_MASK | F_MASK | TYPE_MASK | TAG_MASK;
  pair.datamask_modify = F_MASK;
}
void request_kokkos_neighbors(NeighRequest &request) {
  request.set_kokkos_host(0);
  request.set_kokkos_device(1);
  request.enable_full();
}
void pack_kokkos_graph(LAMMPS *lmp, NeighList *list, const int64_t *types,
                       int capacity, double cutoff, int64_t padding_shift,
                       double *positions, int64_t *elements, int64_t *index,
                       double *shifts) {
  sync_torch();
  auto *atoms = static_cast<AtomKokkos *>(lmp->atom);
  atoms->sync(Device, X_MASK | TYPE_MASK | TAG_MASK);
  const auto x = atoms->k_x.view<LMPDeviceType>();
  const auto tags = atoms->k_tag.view<LMPDeviceType>();
  const auto atom_types = atoms->k_type.view<LMPDeviceType>();
  auto *gpu = static_cast<NeighListKokkos<LMPDeviceType> *>(list);
  const auto ilist = gpu->d_ilist, counts = gpu->d_numneigh;
  const auto neighbors = gpu->d_neighbors;
  const int n = atoms->nlocal;
  using Range = Kokkos::RangePolicy<LMPDeviceType>;
  Kokkos::UnorderedMap<tagint, int, LMPDeviceType> owners(2 * n);
  Kokkos::parallel_for(
      "mace native atom inputs", Range(0, n), KOKKOS_LAMBDA(int i) {
        if (owners.insert(tags(i), i).failed())
          Kokkos::abort("MACE could not map local atom tags");
        elements[i] = types[atom_types(i)];
        for (int a = 0; a < 3; ++a)
          positions[3 * i + a] = x(i, a);
      });
  Kokkos::parallel_for(
      "mace native edge padding", Range(0, capacity), KOKKOS_LAMBDA(int e) {
        index[e] = index[capacity + e] = n - 1;
        shifts[3 * e] = padding_shift;
        shifts[3 * e + 1] = shifts[3 * e + 2] = 0;
      });
  const double cutoff2 = cutoff * cutoff;
  const double lx = lmp->domain->xprd, ly = lmp->domain->yprd,
               lz = lmp->domain->zprd, xy = lmp->domain->xy,
               xz = lmp->domain->xz, yz = lmp->domain->yz;
  Kokkos::parallel_scan(
      "mace native Kokkos edges", Range(0, list->inum),
      KOKKOS_LAMBDA(int ii, int &offset, bool final) {
        const int i = ilist(ii);
        for (int jj = 0; jj < counts(i); ++jj) {
          const int j = neighbors(i, jj) & NEIGHMASK;
          double r2 = 0;
          for (int a = 0; a < 3; ++a) {
            const double dx = x(i, a) - x(j, a);
            r2 += dx * dx;
          }
          if (r2 >= cutoff2)
            continue;
          if (final) {
            if (offset >= capacity)
              Kokkos::abort("MACE neighbour capacity exceeded; re-export with "
                            "more edges");
            const auto slot = owners.find(tags(j));
            if (!owners.valid_at(slot))
              Kokkos::abort("MACE periodic neighbour has no local owner");
            const int real = owners.value_at(slot);
            index[offset] = real;
            index[capacity + offset] = i;
            const double sz = (x(real, 2) - x(j, 2)) / lz;
            const double sy = (x(real, 1) - x(j, 1) - sz * yz) / ly;
            const double sx = (x(real, 0) - x(j, 0) - sy * xy - sz * xz) / lx;
            shifts[3 * offset] = round(sx);
            shifts[3 * offset + 1] = round(sy);
            shifts[3 * offset + 2] = round(sz);
          }
          ++offset;
        }
      });
  Kokkos::fence();
}
void add_kokkos_forces(LAMMPS *lmp, const double *forces) {
  sync_torch();
  auto *atoms = static_cast<AtomKokkos *>(lmp->atom);
  atoms->sync(Device, F_MASK);
  const auto f = atoms->k_f.view<LMPDeviceType>();
  Kokkos::parallel_for(
      "mace native force accumulation",
      Kokkos::RangePolicy<LMPDeviceType>(0, atoms->nlocal),
      KOKKOS_LAMBDA(int i) {
        for (int a = 0; a < 3; ++a)
          f(i, a) += forces[3 * i + a];
      });
  atoms->modified(Device, F_MASK);
  Kokkos::fence();
}
} // namespace mace_native
