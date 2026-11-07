// MPI's standard profiling interface. Time the 3*N double Allreduces used by
// GROMACS's full-system NNP coordinate and force exchange. Waiting for a slower
// rank is included and must not be confused with network transfer time.
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
extern "C" int MPI_Allreduce(const void *send, void *receive, int count,
                             MPI_Datatype datatype, MPI_Op operation,
                             MPI_Comm communicator) {
  const double start = PMPI_Wtime();
  const int result =
      PMPI_Allreduce(send, receive, count, datatype, operation, communicator);
  const char *atoms = std::getenv("MACE_PROFILE_ATOMS");
  int size = 0, rank = 0;
  PMPI_Comm_size(communicator, &size);
  if (atoms && size > 1 && count == 3 * std::atoi(atoms) &&
      datatype == MPI_DOUBLE) {
    PMPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "MACE_MPI %d %d %.9f\n", rank, count,
                 1000 * (PMPI_Wtime() - start));
  }
  return result;
}
