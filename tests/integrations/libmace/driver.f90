! Consumer of libmace's public Fortran module, with MPI broadcast enabled.
program verify_libmace
  use mace, only: MaceModel, dp
  use mpi_f08
  implicit none
  type(MaceModel) :: model
  character(len=4096) :: path, input
  integer :: rank, status, count, flag, reload, unit, ios
  integer :: periodic(3)
  integer, allocatable :: numbers(:)
  real(dp) :: cell(3,3), energy, virial(6)
  real(dp), allocatable :: positions(:,:), forces(:,:), site(:)
  call MPI_Init(status)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, status)
  call get_command_argument(1, path)
  call get_command_argument(2, input)
  model = MaceModel(trim(path), MPI_COMM_WORLD)
  open(newunit=unit, file=trim(input), status='old', action='read')
  do
    read(unit, *, iostat=ios) count, flag, reload
    if (ios /= 0) exit
    allocate(numbers(count), positions(3,count), forces(3,count), site(count))
    ! Each input row is one lattice vector, stored as a Fortran column.
    read(unit, *) cell
    read(unit, *) periodic
    read(unit, *) numbers
    read(unit, *) positions
    if (reload /= 0) call model%reload()
    call model%calculate(flag /= 0, count, cell, periodic /= 0, numbers, positions, energy, site, forces, virial)
    ! Emit every rank to verify that broadcasts preserve all arrays.
    write(*, '(A,I0,1X,*(ES25.17E3,1X))') 'RANK ', rank, energy, site, forces, virial
    deallocate(numbers, positions, forces, site)
  end do
  close(unit)
  call model%deallocate()
  call MPI_Finalize(status)
end program verify_libmace
