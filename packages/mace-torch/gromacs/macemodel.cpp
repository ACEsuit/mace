// The INNPotModel boundary and rank-local force mapping follow the public
// GROMACS interface at 121090014570a53a17ea391bcddae45e5ea05eb4. Model
// execution and neighbours belong to V1 libmace; no scripted model or Python is
// loaded.
#include "macemodel.h"
#include "gmxpre.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/embedded_system_preprocessing.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/mpicomm.h"
#include "nnpotoptions.h"
#include <filesystem>
#include <set>

namespace gmx {
void validateMaceParameters(const NNPotParameters &parameters) {
  if (parameters.embeddingScheme_ != NNPotEmbedding::Mechanical ||
      !parameters.mmIndices_.empty() || !parameters.linkFrontier_.empty() ||
      parameters.nnpCharge_ != 0) {
    GMX_THROW(InconsistentInputError(
        "MACE native currently requires the whole System group and no charged "
        "or hybrid embedding; export a supported energy model."));
  }
  std::set<std::string> inputs;
  for (const auto &input : parameters.modelInput_)
    if (!input.empty() && !inputs.insert(input).second)
      GMX_THROW(InconsistentInputError("Duplicate MACE model input: " + input));
  if (inputs !=
      std::set<std::string>{"atom-positions", "atom-numbers", "box", "pbc"})
    GMX_THROW(InconsistentInputError(
        "MACE native requires exactly atom-positions, atom-numbers, box and "
        "pbc; "
        "neighbours are built from the native model cutoff."));
  const std::filesystem::path path(parameters.modelFileName_);
  if (path.extension() != ".pt2" || !std::filesystem::is_regular_file(path))
    GMX_THROW(InconsistentInputError("nnpot-modelfile must name the existing "
                                     "native.pt2 from a V1 native export."));
}

NativeMaceModel::NativeMaceModel(const NNPotParameters &parameters,
                                 const MpiComm &comm)
    : comm_(comm) {
  validateMaceParameters(parameters);
  if (comm_.isMainRank())
    model_ = std::make_unique<MACE>(parameters.modelFileName_);
}

void NativeMaceModel::evaluateModel(
    gmx_enerdata_t *energy, ArrayRef<RVec> forces, ArrayRef<const int> lookup,
    ArrayRef<const int>, ArrayRef<const std::string>, ArrayRef<RVec> positions,
    ArrayRef<int> numbers, ArrayRef<int>, ArrayRef<RVec>, ArrayRef<RVec>,
    ArrayRef<real>, real, ArrayRef<const LinkFrontierAtom>, matrix *box,
    PbcType *periodicity) {
  if (*periodicity != PbcType::Xyz && *periodicity != PbcType::XY &&
      *periodicity != PbcType::No)
    GMX_THROW(
        InconsistentInputError("MACE native supports pbc=xyz, xy or no."));
  GMX_RELEASE_ASSERT(positions.size() == numbers.size() &&
                         numbers.size() == lookup.size(),
                     "MACE input atom arrays must have the same length");
  // GROMACS uses nm and kJ/mol. Native libmace uses Angstrom and eV.
  constexpr double energyFactor = c_electronCharge * c_avogadro / c_kilo;
  constexpr double forceFactor = energyFactor * c_nm2A;
  std::vector<double> globalForces(3 * positions.size(), 0);
  for (auto &row : virial_)
    for (auto &value : row)
      value = 0;
  if (comm_.isMainRank()) {
    std::array<std::array<double, 3>, 3> cell{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        cell[i][j] = (*box)[i][j] * c_nm2A;
    std::array<bool, 3> pbc{*periodicity != PbcType::No,
                            *periodicity != PbcType::No,
                            *periodicity == PbcType::Xyz};
    std::vector<std::array<double, 3>> coordinates(positions.size());
    for (size_t i = 0; i < coordinates.size(); ++i)
      for (int j = 0; j < 3; ++j)
        coordinates[i][j] = positions[i][j] * c_nm2A;
    const auto result = model_->calculate(
        cell, pbc, coordinates,
        std::vector<int>(numbers.begin(), numbers.end()), true);
    energy->term[InteractionFunction::NeuralNetworkPotentialEnergy] =
        result.total_energy * energyFactor;
    for (size_t i = 0; i < coordinates.size(); ++i)
      for (int j = 0; j < 3; ++j)
        globalForces[3 * i + j] = result.forces[i][j] * forceFactor;
    // GROMACS pressure uses 2*(Ekin-Xi)/V: Xi = -W/2, where W=-V*stress
    // is libmace's physical virial. Add it once, on the main rank.
    constexpr int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 0; k < 6; ++k)
      virial_[a[k]][b[k]] = virial_[b[k]][a[k]] =
          -0.5 * energyFactor * result.virial[k];
  }
  if (comm_.isParallel())
    comm_.sumReduce(globalForces.size(), globalForces.data());
  for (size_t i = 0; i < lookup.size(); ++i) {
    const int local = lookup[i];
    if (local < 0)
      continue;
    GMX_RELEASE_ASSERT(local < forces.ssize(),
                       "MACE local atom index exceeds force buffer");
    for (int j = 0; j < 3; ++j)
      forces[local][j] += globalForces[3 * i + j];
  }
}
} // namespace gmx
