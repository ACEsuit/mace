cmake_minimum_required(VERSION 3.20)
project(mace_native_pair LANGUAGES CXX)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Source tree matching the LAMMPS executable")
find_library(LAMMPS_LIBRARY NAMES lammps REQUIRED)
# Use the exporting interpreter's libtorch and ABI. Linking libtorch directly
# also permits a CPU plugin to be built with a CUDA wheel and no CUDA toolkit.
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib, torch; print(pathlib.Path(torch.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE torch_info OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" torch_info "${torch_info}")
list(GET torch_info 0 torch_root)
list(GET torch_info 1 torch_abi)

add_library(mace_native MODULE pair_mace_native.cpp)
target_compile_features(mace_native PRIVATE cxx_std_20)
target_compile_definitions(mace_native PRIVATE _GLIBCXX_USE_CXX11_ABI=${torch_abi})
target_include_directories(mace_native PRIVATE "${LAMMPS_SOURCE_DIR}/src"
    "${torch_root}/include" "${torch_root}/include/torch/csrc/api/include")
target_link_libraries(mace_native PRIVATE "${LAMMPS_LIBRARY}" MPI::MPI_CXX)
set_target_properties(mace_native PROPERTIES PREFIX "" SUFFIX ".so")

set(torch_libs torch torch_cpu c10)
set(CUEQ_OPS_DIR "" CACHE PATH "Optional cuequivariance_ops 0.10.0 installation")
if(CUEQ_OPS_DIR)
  find_package(CUDAToolkit REQUIRED)
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cueq_schemas.py"
                  "${CMAKE_CURRENT_BINARY_DIR}/cueq_schemas.inc" "${CUEQ_OPS_DIR}"
                  COMMAND_ERROR_IS_FATAL ANY)
  target_sources(mace_native PRIVATE cueq_ops.cpp)
  target_compile_definitions(mace_native PRIVATE MACE_HAS_CUEQ)
  target_include_directories(mace_native PRIVATE "${CUEQ_OPS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(mace_native PRIVATE "${CUEQ_OPS_DIR}/lib/libcue_ops.so" CUDA::cudart)
endif()
if(EXISTS "${torch_root}/lib/libtorch_cuda.so")
  # Load the CUDA runtime registration even when all calls use ATen dispatch.
  target_link_options(mace_native PRIVATE "LINKER:--no-as-needed")
  list(APPEND torch_libs torch_cuda c10_cuda)
endif()
foreach(lib IN LISTS torch_libs)
  find_library(path_${lib} NAMES ${lib} PATHS "${torch_root}/lib" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(mace_native PRIVATE "${path_${lib}}")
endforeach()
