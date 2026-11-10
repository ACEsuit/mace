// Native LAMMPS plugin. The MPI pack/unpack contract follows the LAMMPS
// many-body pair styles; gradients require reverse summation, not a second
// gather.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#ifdef MACE_HAS_CUEQ
#include <cuda_runtime_api.h>
#endif
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/library.h>

#include "../native/package_metadata.h"
#ifdef MACE_HAS_KOKKOS
#include "kokkos_neighbors.h"
#endif
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "lammpsplugin.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "version.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using namespace LAMMPS_NS;
using Loader = torch::inductor::AOTIModelPackageLoader;

class NativePair final : public Pair {
public:
  explicit NativePair(LAMMPS *lmp, bool kokkos = false)
      : Pair(lmp), use_kokkos(kokkos) {
#ifdef MACE_HAS_KOKKOS
    if (use_kokkos) {
      try {
        mace_native::configure_kokkos(lmp, *this);
      } catch (const std::exception &error_message) {
        error->all(FLERR, "Cannot initialize mace/native/kk: {}",
                   error_message.what());
      }
    }
#endif
    manybody_flag = 1;
    single_enable = restartinfo = 0;
    centroidstressflag = CENTROID_NOTAVAIL;
  }
  ~NativePair() override {
    if (allocated) {
      memory->destroy(setflag);
      memory->destroy(cutsq);
    }
  }
  void settings(int argc, char **argv) override {
    if (argc != 1)
      error->all(FLERR, "pair_style mace/native requires the native.pt2 path");
    try {
      auto meta = Loader::load_metadata_from_package(argv[0], "model");
      this->package = std::make_unique<mace_native::PackageMetadata>(meta);
      const auto &package = *this->package;
      no_virial_fdotr_compute = package.periodic_system;
      nodes = package.nodes;
      edges = package.edges;
      cutoff = package.cutoff;
      comm_forward = comm_reverse = package.comm_width;
      dtype = package.dtype == "float64" ? at::kDouble : at::kFloat;
      species = package.species;
      device = at::Device(package.device);
      if (use_kokkos && (!package.periodic_system || !device.is_cuda()))
        throw std::runtime_error(
            "mace/native/kk currently requires a periodic CUDA package");
      if (device.is_cuda()) {
        const int count = at::getNumGPUs();
        if (!count)
          throw std::runtime_error("CUDA package requires a visible GPU");
        MPI_Comm shared;
        MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me,
                            MPI_INFO_NULL, &shared);
        int rank;
        MPI_Comm_rank(shared, &rank);
        MPI_Comm_free(&shared);
        device = at::Device(at::kCUDA, rank % count);
#ifdef MACE_HAS_CUEQ
        cudaDeviceProp properties{};
        const auto status =
            cudaGetDeviceProperties(&properties, device.index());
        if (status != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(status));
        const std::string sm =
            std::to_string(properties.major) + std::to_string(properties.minor);
        const auto compiled = meta.at("AOTI_COMPUTE_CAPABILITY");
        if (compiled != sm)
          throw std::runtime_error("package targets SM" + compiled +
                                   " but this GPU is SM" + sm +
                                   "; re-export on the target GPU");
#endif
      }
      loader =
          std::make_unique<Loader>(argv[0], "model", false, 1, device.index());
    } catch (const std::exception &e) {
      // A device or loader failure can be local to a single rank.
      error->one(FLERR, "Cannot load mace/native package: {}", e.what());
    }
  }
  void coeff(int argc, char **argv) override {
    if (argc != atom->ntypes + 2 || std::string(argv[0]) != "*" ||
        std::string(argv[1]) != "*")
      error->all(
          FLERR,
          "Use pair_coeff * * followed by one element symbol per atom type");
    if (!allocated) {
      memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1,
                     "mace:setflag");
      memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "mace:cutsq");
      allocated = 1;
    }
    types.resize(atom->ntypes + 1);
    for (int i = 1; i <= atom->ntypes; ++i) {
      auto found = std::find(species.begin(), species.end(), argv[i + 1]);
      if (found == species.end())
        error->all(FLERR, "Unknown mace/native element {}", argv[i + 1]);
      types[i] = found - species.begin();
      for (int j = i; j <= atom->ntypes; ++j)
        setflag[i][j] = 1;
    }
  }
  double init_one(int, int) override { return cutoff; }
  void init_style() override {
    // Complete periodic evaluation returns forces on real atoms directly;
    // Kokkos full lists require newton off and need no ghost force reduction.
    if (!force->newton_pair && !package->periodic_system)
      error->all(FLERR, "mace/native requires newton pair on");
    if (std::string(update->unit_style) != "metal")
      error->all(FLERR, "mace/native requires units metal (eV and Angstrom)");
    if (atom->molecular != Atom::ATOMIC)
      error->all(FLERR, "mace/native requires atomic neighbor lists without "
                        "bonded exclusions");
    if (package->periodic_system && (comm->nprocs != 1 || !atom->tag_enable))
      error->all(FLERR, "Polar mace/native requires one MPI rank and atom IDs");
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
#ifdef MACE_HAS_KOKKOS
    if (use_kokkos)
      mace_native::request_kokkos_neighbors(*request);
#else
    (void)request;
#endif
  }

  void compute(int eflag, int vflag) override;
  at::Tensor exchange(const at::Tensor &, bool);
  int pack_forward_comm(int n, int *indices, double *buf, int, int *) override {
    for (int j = 0; j < n; ++j)
      std::copy_n(row(indices[j]), width, buf + j * width);
    return n * width;
  }
  void unpack_forward_comm(int n, int first, double *buf) override {
    std::copy_n(buf, n * width, row(first));
  }
  int pack_reverse_comm(int n, int first, double *buf) override {
    std::copy_n(row(first), n * width, buf);
    return n * width;
  }
  void unpack_reverse_comm(int n, int *indices, double *buf) override {
    for (int j = 0; j < n; ++j) {
      auto dest = row(indices[j]);
      for (int k = 0; k < width; ++k)
        dest[k] += buf[j * width + k];
    }
  }

private:
  bool use_kokkos;
  std::unique_ptr<Loader> loader;
  std::unique_ptr<mace_native::PackageMetadata> package;
  at::Device device{at::kCPU};
  at::ScalarType dtype = at::kDouble;
  int nodes = 0, edges = 0, width = 0;
  double cutoff = 0;
  std::vector<std::string> species;
  std::vector<int> types;
  at::Tensor buffer;
  double *row(int index) { return buffer.data_ptr<double>() + index * width; }
};

// Custom operators in a package can execute only during its owning LAMMPS call.
thread_local NativePair *active = nullptr;

at::Tensor NativePair::exchange(const at::Tensor &values, bool reverse) {
  TORCH_CHECK(values.dim() == 2,
              "expected [nodes, features] at the MPI exchange");
  width = values.size(1);
  TORCH_CHECK(values.dim() == 2 && values.size(0) == nodes &&
                  width <= comm_forward,
              "unexpected mace/native halo shape");
  // Even CPU doubles must be copied: the dispatcher operator is out-of-place.
  buffer = values.to(at::kCPU, at::kDouble, false, true).contiguous();
  if (reverse) {
    comm->reverse_comm(this, width);
    buffer.narrow(0, atom->nlocal, nodes - atom->nlocal).zero_();
  } else {
    comm->forward_comm(this, width);
  }
  return buffer.to(values.options());
}

void NativePair::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  try {
    c10::DeviceGuard device_guard(device);
    at::NoGradGuard guard;
    int local = atom->nlocal;
    const bool periodic = package->periodic_system;
    int total = periodic ? local : local + atom->nghost;
    std::array<std::array<double, 3>, 3> cell{
        {{domain->xprd, 0, 0},
         {domain->xy, domain->yprd, 0},
         {domain->xz, domain->yz, domain->zprd}}};
    package->validate_system(cell,
                             {bool(domain->xperiodic), bool(domain->yperiodic),
                              bool(domain->zperiodic)},
                             local);
    if (periodic && (eflag_atom || vflag_atom))
      error->one(FLERR, "Polar mace/native does not partition long-range "
                        "energy or virial per atom");
#ifdef MACE_HAS_KOKKOS
    if (use_kokkos) {
      auto longs = at::TensorOptions().device(device).dtype(at::kLong);
      auto doubles = longs.dtype(at::kDouble);
      auto positions = at::empty({nodes, 3}, doubles);
      auto elements = at::empty({nodes}, longs);
      auto index = at::empty({2, edges}, longs);
      auto shifts = at::empty({edges, 3}, doubles);
      auto mapping = at::tensor(types, at::TensorOptions().dtype(at::kInt))
                         .to(device, at::kLong);
      mace_native::pack_kokkos_graph(
          lmp, list, mapping.data_ptr<int64_t>(), edges, cutoff,
          package->padding_unit_shift(), positions.data_ptr<double>(),
          elements.data_ptr<int64_t>(), index.data_ptr<int64_t>(),
          shifts.data_ptr<double>());
      auto box =
          at::from_blob(cell.data(), {3, 3}, at::kDouble).to(device, dtype);
      auto out = loader->run(
          {shifts.to(dtype), index, elements, positions.to(dtype), box});
      const auto forces = out.at(2).to(at::kDouble).contiguous();
      TORCH_CHECK(forces.sizes() == at::IntArrayRef({nodes, 3}),
                  "invalid periodic CUDA forces");
      mace_native::add_kokkos_forces(lmp, forces.data_ptr<double>());
      if (eflag_global)
        eng_vdwl += out.at(0).item<double>();
      if (vflag_global) {
        const auto w = out.at(3).to(at::kCPU, at::kDouble).contiguous();
        constexpr int left[6] = {0, 1, 2, 0, 0, 1},
                      right[6] = {0, 1, 2, 1, 2, 2};
        const auto tensor = w.accessor<double, 2>();
        for (int i = 0; i < 6; ++i)
          virial[i] += tensor[left[i]][right[i]];
      }
      return;
    }
#endif
    std::unordered_map<tagint, int> owner;
    if (periodic)
      for (int i = 0; i < local; ++i)
        owner.emplace(atom->tag[i], i);
    if (!periodic && total >= nodes)
      error->one(
          FLERR,
          "mace/native needs {} node slots including padding; exported {}",
          total + 1, nodes);
    auto elements = at::zeros({nodes}, at::kLong);
    auto owned = at::zeros({nodes}, at::kDouble);
    auto index = at::full({2, edges}, nodes - 1, at::kLong);
    auto vectors = at::zeros({edges, 3}, at::kDouble);
    // Periodic packages take lattice shifts; local packages take edge vectors.
    // Both pad self edges beyond the radial cutoff.
    vectors.select(1, 0).fill_(periodic ? package->padding_unit_shift()
                                        : 2 * cutoff);
    owned.narrow(0, 0, local).fill_(1);
    for (int i = 0; i < total; ++i)
      elements.data_ptr<int64_t>()[i] = types[atom->type[i]];
    auto edge = index.accessor<int64_t, 2>();
    auto vec = vectors.accessor<double, 2>();
    int count = 0;
    for (int ii = 0; ii < list->inum; ++ii) {
      const int i = list->ilist[ii], numneigh = list->numneigh[i];
      const int *row = list->firstneigh[i];
      for (int jj = 0; jj < numneigh; ++jj) {
        int j = row[jj] & NEIGHMASK;
        double dx[3], rsq = 0;
        for (int k = 0; k < 3; ++k) {
          dx[k] = atom->x[i][k] - atom->x[j][k];
          rsq += dx[k] * dx[k];
        }
        if (rsq >= cutoff * cutoff)
          continue;
        if (count == edges)
          error->one(FLERR,
                     "mace/native exceeds {} edge slots; re-export with a "
                     "larger capacity",
                     edges);
        edge[0][count] = periodic ? owner.at(atom->tag[j]) : j;
        edge[1][count] = i;
        if (periodic) {
          // The neighbour may be an image of a real atom. Pass integer image
          // offsets so the package forms vectors in the model's own precision.
          const int real = edge[0][count];
          const double sz = (atom->x[real][2] - atom->x[j][2]) / domain->zprd;
          const double sy =
              (atom->x[real][1] - atom->x[j][1] - sz * domain->yz) /
              domain->yprd;
          const double sx = (atom->x[real][0] - atom->x[j][0] -
                             sy * domain->xy - sz * domain->xz) /
                            domain->xprd;
          vec[count][0] = std::round(sx);
          vec[count][1] = std::round(sy);
          vec[count][2] = std::round(sz);
        } else {
          std::copy_n(dx, 3, vec[count].data());
        }
        ++count;
      }
    }
    if (periodic) {
      auto positions = at::empty({nodes, 3}, at::kDouble);
      auto p = positions.accessor<double, 2>();
      for (int i = 0; i < local; ++i)
        for (int j = 0; j < 3; ++j)
          p[i][j] = atom->x[i][j];
      auto box =
          at::from_blob(cell.data(), {3, 3}, at::kDouble).to(device, dtype);
      const auto out =
          loader->run({vectors.to(device, dtype), index.to(device),
                       elements.to(device), positions.to(device, dtype), box});
      const auto energy = out.at(0).to(at::kCPU, at::kDouble).contiguous();
      const auto forces = out.at(2).to(at::kCPU, at::kDouble).contiguous();
      const auto w = out.at(3).to(at::kCPU, at::kDouble).contiguous();
      TORCH_CHECK(energy.numel() == 1 &&
                      forces.sizes() == at::IntArrayRef({nodes, 3}) &&
                      w.sizes() == at::IntArrayRef({3, 3}),
                  "invalid periodic output shapes");
      if (eflag_global)
        eng_vdwl += energy.item<double>();
      const auto f = forces.accessor<double, 2>();
      for (int i = 0; i < local; ++i)
        for (int j = 0; j < 3; ++j)
          atom->f[i][j] += f[i][j];
      if (vflag_global) {
        constexpr int left[6] = {0, 1, 2, 0, 0, 1},
                      right[6] = {0, 1, 2, 1, 2, 2};
        const auto tensor = w.accessor<double, 2>();
        for (int i = 0; i < 6; ++i)
          virial[i] += tensor[left[i]][right[i]];
      }
      return;
    }
    struct Scope {
      NativePair *before;
      explicit Scope(NativePair *current) : before(active) { active = current; }
      ~Scope() { active = before; }
    } scope(this);
    auto out = loader->run({vectors.to(device, dtype), index.to(device),
                            elements.to(device), owned.to(device, dtype)});
    auto site = out.at(0).to(at::kCPU, at::kDouble).contiguous();
    auto grad = out.at(1).to(at::kCPU, at::kDouble).contiguous();
    TORCH_CHECK(site.dim() == 1 && site.size(0) == nodes && grad.dim() == 2 &&
                    grad.size(0) == edges && grad.size(1) == 3,
                "invalid native.pt2 output shape");
    auto g = grad.accessor<double, 2>();
    for (int i = 0; i < local; ++i) {
      if (eflag_global)
        eng_vdwl += site.data_ptr<double>()[i];
      if (eflag_atom)
        eatom[i] += site.data_ptr<double>()[i];
    }
    for (int p = 0; p < count; ++p) {
      int j = edge[0][p], i = edge[1][p];
      for (int k = 0; k < 3; ++k) {
        atom->f[j][k] += g[p][k];
        atom->f[i][k] -= g[p][k];
      }
      if (vflag_either)
        ev_tally_xyz(i, j, local, 1, 0, 0, -g[p][0], -g[p][1], -g[p][2],
                     vec[p][0], vec[p][1], vec[p][2]);
    }
    if (vflag_fdotr)
      virial_fdotr_compute();
  } catch (const std::exception &e) {
    error->one(FLERR, "mace/native evaluation failed: {}", e.what());
  }
}

at::Tensor forward_halo(const at::Tensor &v) {
  TORCH_CHECK(active, "halo outside a mace/native step");
  return active->exchange(v, false);
}
at::Tensor reverse_halo(const at::Tensor &v) {
  TORCH_CHECK(active, "halo adjoint outside a mace/native step");
  return active->exchange(v, true);
}
TORCH_LIBRARY(mace_native, m) {
  m.def("halo(Tensor features) -> Tensor", forward_halo);
  m.def("halo_adjoint(Tensor adjoint) -> Tensor", reverse_halo);
}
Pair *create_pair(LAMMPS *lmp) { return new NativePair(lmp); }
#ifdef MACE_HAS_KOKKOS
Pair *create_kokkos_pair(LAMMPS *lmp) { return new NativePair(lmp, true); }
#endif
} // namespace

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "mace/native";
  plugin.info = "Native AOTInductor MACE energy and forces";
  plugin.author = "MACE developers";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_pair);
  plugin.handle = handle;
  reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&plugin, lmp);
#ifdef MACE_HAS_KOKKOS
  plugin.name = "mace/native/kk";
  plugin.info = "Native Polar with Kokkos CUDA neighbours and device forces";
  plugin.creator.v1 =
      reinterpret_cast<lammpsplugin_factory1 *>(&create_kokkos_pair);
  reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&plugin, lmp);
#endif
}
