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
set_target_properties(mace_native PROPERTIES PREFIX "" SUFFIX ".so")
