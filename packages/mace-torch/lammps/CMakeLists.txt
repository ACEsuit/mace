cmake_minimum_required(VERSION 3.20)
project(mace_native_pair LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Source tree matching the LAMMPS executable")
find_library(LAMMPS_LIBRARY NAMES lammps REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/../native/torch_runtime.cmake")

add_library(mace_native MODULE pair_mace_native.cpp)
target_include_directories(mace_native PRIVATE "${LAMMPS_SOURCE_DIR}/src")
mace_link_torch(mace_native PRIVATE)
target_link_libraries(mace_native PRIVATE "${LAMMPS_LIBRARY}" MPI::MPI_CXX)
set_target_properties(mace_native PROPERTIES PREFIX "" SUFFIX ".so"
  BUILD_RPATH_USE_ORIGIN TRUE)

option(MACE_WITH_KOKKOS "Build mace/native/kk with CUDA neighbour lists" OFF)
if(MACE_WITH_KOKKOS)
  enable_language(CUDA)
  find_package(Kokkos CONFIG REQUIRED COMPONENTS separable_compilation)
  add_library(mace_kokkos_neighbors SHARED kokkos_neighbors.cpp)
  set_source_files_properties(kokkos_neighbors.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(mace_kokkos_neighbors PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CUDA CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(mace_kokkos_neighbors PRIVATE --extended-lambda --fmad=false)
  target_compile_definitions(mace_kokkos_neighbors PRIVATE
    LMP_KOKKOS LMP_KOKKOS_DOUBLE_DOUBLE LMP_KOKKOS_LAYOUT_LEGACY)
  target_include_directories(mace_kokkos_neighbors PRIVATE
    "${LAMMPS_SOURCE_DIR}/src" "${LAMMPS_SOURCE_DIR}/src/KOKKOS")
  # LAMMPS already owns the Kokkos runtime. Linking its static archives here
  # creates a second set of runtime globals and corrupts process teardown.
  foreach(component kokkoscore kokkoscontainers kokkosalgorithms kokkossimd)
    if(TARGET Kokkos::${component})
      target_include_directories(mace_kokkos_neighbors SYSTEM PRIVATE
        "$<TARGET_PROPERTY:Kokkos::${component},INTERFACE_INCLUDE_DIRECTORIES>")
    endif()
  endforeach()
  target_link_libraries(mace_kokkos_neighbors PRIVATE "${LAMMPS_LIBRARY}"
    CUDA::cudart MPI::MPI_CXX)
  target_compile_definitions(mace_native PRIVATE MACE_HAS_KOKKOS)
  target_link_libraries(mace_native PRIVATE mace_kokkos_neighbors)
endif()
