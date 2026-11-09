# Shared ATen/AOTInductor runtime for native deployment consumers.
include_guard(GLOBAL)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import pathlib, torch; print(pathlib.Path(torch.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
  OUTPUT_VARIABLE torch_info OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" torch_info "${torch_info}")
list(GET torch_info 0 torch_root)
list(GET torch_info 1 torch_abi)
set(CUEQ_OPS_DIR "" CACHE PATH "Optional cuequivariance_ops 0.10.0 installation")

function(mace_link_torch target abi_visibility)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_compile_definitions(${target} ${abi_visibility} _GLIBCXX_USE_CXX11_ABI=${torch_abi})
  target_include_directories(${target} PRIVATE
    "${torch_root}/include" "${torch_root}/include/torch/csrc/api/include")
  set(torch_libs torch torch_cpu c10)
  if(CUEQ_OPS_DIR)
    find_package(CUDAToolkit REQUIRED)
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cueq_schemas.py"
      "${CMAKE_CURRENT_BINARY_DIR}/cueq_schemas.inc" "${CUEQ_OPS_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
    target_sources(${target} PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cueq_ops.cpp")
    target_compile_definitions(${target} PRIVATE MACE_HAS_CUEQ)
    target_include_directories(${target} PRIVATE "${CUEQ_OPS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(${target} PRIVATE "${CUEQ_OPS_DIR}/lib/libcue_ops.so" CUDA::cudart)
  endif()
  if(EXISTS "${torch_root}/lib/libtorch_cuda.so")
    # Keep CUDA runtime registration even when every call uses ATen dispatch.
    target_link_options(${target} PRIVATE "LINKER:--no-as-needed")
    list(APPEND torch_libs torch_cuda c10_cuda)
  endif()
  foreach(lib IN LISTS torch_libs)
    find_library(path_${lib} NAMES ${lib} PATHS "${torch_root}/lib" NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(${target} PRIVATE "${path_${lib}}")
  endforeach()
endfunction()
